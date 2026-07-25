from __future__ import annotations

import time
import logging
from typing import List, Tuple, Optional, Union
from datetime import datetime, timedelta
import numpy as np
from scipy.optimize import linear_sum_assignment
import supervision as sv

from config import TRACKER, settings
from core.head_detector import Detection

logger = logging.getLogger("tracker")


def iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Vectorized IoU between two sets of xyxy boxes -> shape (len(a), len(b))."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    a = boxes_a[:, None, :]
    b = boxes_b[None, :, :]

    xx1 = np.maximum(a[..., 0], b[..., 0])
    yy1 = np.maximum(a[..., 1], b[..., 1])
    xx2 = np.minimum(a[..., 2], b[..., 2])
    yy2 = np.minimum(a[..., 3], b[..., 3])

    w = np.clip(xx2 - xx1, 0, None)
    h = np.clip(yy2 - yy1, 0, None)
    inter = w * h

    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter + 1e-6
    return inter / union


class KalmanBoxTracker:
    """Constant-velocity 8-state Kalman filter over state [cx, cy, w, h, vcx, vcy, vw, vh]."""

    _next_id = 1

    def __init__(self, bbox: np.ndarray, score: float, track_thresh: float = 0.5):
        cx, cy, w, h = self._xyxy_to_cxcywh(bbox)

        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)
        self.P = np.eye(8, dtype=np.float32) * 10.0
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = 1.0  # position += velocity per timestep
        self.H = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.H[i, i] = 1.0

        self.Q = np.eye(8, dtype=np.float32) * 0.01  # process noise
        self.R = np.eye(4, dtype=np.float32) * 1.0   # measurement noise

        self.id = KalmanBoxTracker._next_id
        KalmanBoxTracker._next_id += 1

        self.score = score
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.state = "confirmed" if score >= track_thresh else "tentative"

    @staticmethod
    def _xyxy_to_cxcywh(bbox: np.ndarray) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox
        return float((x1 + x2) / 2.0), float((y1 + y2) / 2.0), float(x2 - x1), float(y2 - y1)

    @staticmethod
    def _cxcywh_to_xyxy(cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
        return float(cx - w / 2.0), float(cy - h / 2.0), float(cx + w / 2.0), float(cy + h / 2.0)

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return self.get_state()

    def update(self, bbox: np.ndarray, score: float, track_thresh: float = 0.5) -> None:
        z = np.array(self._xyxy_to_cxcywh(bbox), dtype=np.float32)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8, dtype=np.float32) - K @ self.H) @ self.P

        self.score = score
        self.hits += 1
        self.time_since_update = 0
        if self.hits >= 2 or score >= track_thresh:
            self.state = "confirmed"

    def get_state(self) -> np.ndarray:
        cx, cy, w, h = self.x[:4]
        return np.array(self._cxcywh_to_xyxy(cx, cy, w, h), dtype=np.float32)


class HeadTracker:
    """
    ByteTrack-style multi-object tracker optimized for head bounding boxes.
    Supports prediction-only pass on skipped detection frames for high FPS.
    """

    def __init__(self, cfg: Optional[dict[str, Union[float, int]]] = None):
        self.cfg = cfg or TRACKER
        self.tracks: List[KalmanBoxTracker] = []
        self.frame_count = 0

    def _match(
        self,
        tracks: List[KalmanBoxTracker],
        dets: List[Detection],
        dist_thresh: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if len(tracks) == 0 or len(dets) == 0:
            return [], list(range(len(tracks))), list(range(len(dets)))

        track_boxes = np.array([t.get_state() for t in tracks])
        det_boxes = np.array([d.bbox for d in dets])
        cost = 1.0 - iou_batch(track_boxes, det_boxes)

        row_ind, col_ind = linear_sum_assignment(cost)

        matches, matched_t, matched_d = [], set(), set()
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] <= dist_thresh:
                matches.append((r, c))
                matched_t.add(r)
                matched_d.add(c)

        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_t]
        unmatched_dets = [i for i in range(len(dets)) if i not in matched_d]
        return matches, unmatched_tracks, unmatched_dets

    def _confirmed_results(self) -> List[dict[str, Union[int, float, Tuple[float, float, float, float]]]]:
        results: List[dict[str, Union[int, float, Tuple[float, float, float, float]]]] = []
        for t in self.tracks:
            if t.state == "confirmed":
                x1, y1, x2, y2 = t.get_state()
                results.append({
                    "track_id": t.id,
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),
                    "score": float(t.score),
                    "age": t.age,
                    "time_since_update": t.time_since_update,
                })
        return results

    def update(self, detections: Optional[List[Detection]] = None) -> List[dict[str, Union[int, float, Tuple[float, float, float, float]]]]:
        self.frame_count += 1

        # Advance every track's motion filter once per frame
        for t in self.tracks:
            t.predict()

        buffer_val = self.cfg.get("track_buffer", 30)
        buffer = int(buffer_val) if isinstance(buffer_val, (int, float)) else 30
        self.tracks = [t for t in self.tracks if t.time_since_update <= buffer]

        if detections is None:
            return self._confirmed_results()

        match_thresh_val = self.cfg.get("match_thresh", 0.8)
        match_dist_thresh = float(match_thresh_val) if isinstance(match_thresh_val, (int, float)) else 0.8

        track_thresh_val = self.cfg.get("track_thresh", 0.5)
        track_thresh = float(track_thresh_val) if isinstance(track_thresh_val, (int, float)) else 0.5

        low_thresh_val = self.cfg.get("low_thresh", 0.1)
        low_thresh = float(low_thresh_val) if isinstance(low_thresh_val, (int, float)) else 0.1

        high = [d for d in detections if d.score >= track_thresh]
        low = [d for d in detections if low_thresh <= d.score < track_thresh]

        active = [t for t in self.tracks if t.state in ("tentative", "confirmed")]
        lost = [t for t in self.tracks if t.state == "lost"]

        # 1st pass: active tracks vs high-confidence detections
        matches, unmatched_tracks_idx, unmatched_high_idx = self._match(active, high, match_dist_thresh)
        for t_idx, d_idx in matches:
            active[t_idx].update(np.array(high[d_idx].bbox), high[d_idx].score, track_thresh)

        remaining_tracks = [active[i] for i in unmatched_tracks_idx]

        # 2nd pass: remaining active tracks vs low-confidence detections
        matches2, unmatched_tracks_idx2, _ = self._match(remaining_tracks, low, 0.5)
        for t_idx, d_idx in matches2:
            remaining_tracks[t_idx].update(np.array(low[d_idx].bbox), low[d_idx].score, track_thresh)

        for i in unmatched_tracks_idx2:
            remaining_tracks[i].state = "lost"

        # 3rd pass: lost tracks vs leftover high-conf detections
        leftover_high = [high[i] for i in unmatched_high_idx]
        matches3, _, unmatched_leftover_idx = self._match(lost, leftover_high, match_dist_thresh)
        for t_idx, d_idx in matches3:
            lost[t_idx].update(np.array(leftover_high[d_idx].bbox), leftover_high[d_idx].score, track_thresh)

        # Spawn new tracks for un-matched high-conf detections
        for i in unmatched_leftover_idx:
            d = leftover_high[i]
            self.tracks.append(KalmanBoxTracker(np.array(d.bbox), d.score, track_thresh))

        return self._confirmed_results()


class StreamTrackManager:
    """Track manager providing Supervision annotator support and spatial identity locks."""

    def __init__(self, camera_id: str, frame_rate: int = settings.TRACKING_FPS):
        self.camera_id = camera_id
        self.head_tracker = HeadTracker()
        self.track_states: dict[int, str] = {}
        self.track_identities: dict[int, dict[str, str]] = {}
        self.track_low_conf_counts: dict[int, int] = {}
        self.spatial_memory: List[dict[str, Union[str, float, List[int]]]] = []
        
        self.box_annotator = sv.BoxAnnotator(thickness=2, color_lookup=sv.ColorLookup.INDEX)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, color_lookup=sv.ColorLookup.INDEX)
        self.profile_cooldowns: dict[str, datetime] = {}
        self._stale_grace_seconds: float = 10.0
        self.track_last_seen: dict[int, float] = {}

    def update(
        self,
        head_boxes: Union[List[Tuple[Tuple[int, int, int, int], float]], List[Detection], None],
        frame_shape: Tuple[int, ...]
    ) -> Tuple[sv.Detections, List[Tuple[int, Tuple[int, int, int, int]]]]:
        now = time.time()
        h_img, w_img = frame_shape[:2]

        dets: Optional[List[Detection]] = None
        if head_boxes is not None:
            dets = []
            for item in head_boxes:
                if isinstance(item, Detection):
                    dets.append(item)
                elif isinstance(item, tuple) and len(item) == 2:
                    (x1, y1, x2, y2), conf = item
                    dets.append(Detection(bbox=(float(x1), float(y1), float(x2), float(y2)), score=float(conf)))

        track_results = self.head_tracker.update(dets)

        xyxy_list: List[List[float]] = []
        tracker_ids: List[int] = []
        confidence_list: List[float] = []
        pending_jobs: List[Tuple[int, Tuple[int, int, int, int]]] = []

        for trk in track_results:
            tid = int(trk["track_id"])
            x1, y1, x2, y2 = trk["bbox"]
            ix1, iy1, ix2, iy2 = max(0, int(x1)), max(0, int(y1)), min(w_img, int(x2)), min(h_img, int(y2))
            
            xyxy_list.append([x1, y1, x2, y2])
            tracker_ids.append(tid)
            confidence_list.append(float(trk["score"]))

            if tid not in self.track_states:
                self.track_states[tid] = "PENDING"
                self.track_identities[tid] = {
                    "label": f"Head #{tid}",
                    "profile_id": "Unknown",
                    "status": "pending"
                }

            if self.track_states.get(tid) == "PENDING":
                pending_jobs.append((tid, (ix1, iy1, ix2, iy2)))

            self.track_last_seen[tid] = now

        if len(xyxy_list) == 0:
            sv_dets = sv.Detections.empty()
        else:
            sv_dets = sv.Detections(
                xyxy=np.array(xyxy_list, dtype=np.float32),
                confidence=np.array(confidence_list, dtype=np.float32),
                tracker_id=np.array(tracker_ids, dtype=int)
            )

        # Cleanup stale tracks
        stale_ids = [
            tid for tid, ts in self.track_last_seen.items()
            if (now - ts) > self._stale_grace_seconds
        ]
        for tid in stale_ids:
            self.track_states.pop(tid, None)
            self.track_identities.pop(tid, None)
            self.track_low_conf_counts.pop(tid, None)
            self.track_last_seen.pop(tid, None)

        return sv_dets, pending_jobs

    def set_track_identity(self, track_id: int, profile_id: str, label: str, is_new_visit: bool) -> None:
        self.track_states[track_id] = "RESOLVED"
        self.track_identities[track_id] = {
            "label": label,
            "profile_id": profile_id,
            "status": "resolved"
        }

    def check_visit_cooldown(self, profile_id: str) -> bool:
        now = datetime.now()
        if profile_id in self.profile_cooldowns:
            elapsed = now - self.profile_cooldowns[profile_id]
            if elapsed < timedelta(minutes=settings.VISIT_COOLDOWN_MINS):
                return True
        self.profile_cooldowns[profile_id] = now
        return False
