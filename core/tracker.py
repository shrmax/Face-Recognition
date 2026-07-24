import time
import logging
import numpy as np
import supervision as sv
from typing import Dict, List, Tuple, Optional, Protocol, Sequence
from datetime import datetime, timedelta
from config import settings

logger = logging.getLogger("tracker")

class FaceObject(Protocol):
    det_score: float
    bbox: np.ndarray
    embedding: np.ndarray

def _compute_iou(box1: np.ndarray, box2: List[int]) -> float:
    x1 = max(int(box1[0]), box2[0])
    y1 = max(int(box1[1]), box2[1])
    x2 = min(int(box1[2]), box2[2])
    y2 = min(int(box1[3]), box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(1, int((box1[2] - box1[0]) * (box1[3] - box1[1])))
    area2 = max(1, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    return float(inter_area) / float(area1 + area2 - inter_area)

class TrackState:
    PENDING = "PENDING"
    RESOLVED = "RESOLVED"  # Name is locked, no more recognition jobs needed

class StreamTrackManager:
    def __init__(self, camera_id: str, frame_rate: int = settings.TRACKING_FPS):
        self.camera_id = camera_id
        # sv.ByteTrack instance for persistent object tracking
        self.byte_tracker = sv.ByteTrack(
            track_activation_threshold=0.15,
            lost_track_buffer=180,
            minimum_matching_threshold=0.2,
            frame_rate=frame_rate
        )
        # track_id -> status dict (PENDING vs RESOLVED)
        self.track_states: Dict[int, str] = {}
        # track_id -> assigned profile_id & label info
        self.track_identities: Dict[int, Dict[str, str]] = {}
        # track_id -> count of consecutive low-confidence evaluations
        self.track_low_conf_counts: Dict[int, int] = {}
        # track_id -> timestamp of last queued recognition job for rate-limiting
        self.last_job_times: Dict[int, float] = {}
        
        # Spatial Memory for persistent Re-ID across track ID churn
        self.spatial_memory: List[Dict[str, object]] = []
        
        # Annotators for visualization overlay
        self.box_annotator = sv.BoxAnnotator(thickness=2, color_lookup=sv.ColorLookup.INDEX)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, color_lookup=sv.ColorLookup.INDEX)
        
        # Profile ID -> Last Event Timestamp (Visit Cooldown)
        self.profile_cooldowns: Dict[str, datetime] = {}

    def _spatial_reid_match(self, bbox: List[int]) -> Tuple[Optional[str], Optional[str]]:
        """Matches a new track bounding box against recently active resolved identities in spatial memory."""
        now = time.time()
        valid_memory: List[Dict[str, object]] = []
        for m in self.spatial_memory:
            ts = m.get("timestamp")
            if isinstance(ts, (int, float)) and (now - float(ts)) < 10.0:
                valid_memory.append(m)
        self.spatial_memory = valid_memory
        
        best_iou = 0.0
        matched_pid: Optional[str] = None
        matched_lbl: Optional[str] = None
        
        for mem in self.spatial_memory:
            m_box = mem.get("bbox")
            if isinstance(m_box, list) and len(m_box) == 4:
                iou = _compute_iou(np.array(m_box), bbox)
                if iou > 0.35 and iou > best_iou:
                    best_iou = iou
                    matched_pid = str(mem.get("profile_id", ""))
                    matched_lbl = str(mem.get("label", ""))
                    
        if best_iou > 0.35 and matched_pid and matched_pid not in ("Unknown", "Pending"):
            return matched_pid, matched_lbl
        return None, None

    def update(
        self,
        head_boxes: List[Tuple[Tuple[int, int, int, int], float]],
        frame_shape: Tuple[int, ...]
    ) -> Tuple[sv.Detections, List[Tuple[int, Tuple[int, int, int, int]]]]:
        """
        Updates ByteTrack with current frame's detected head bounding boxes.
        Returns:
            - sv.Detections object for annotation overlay
            - List of pending jobs for unresolved tracks: [(track_id, (x1, y1, x2, y2))]
        """
        now = time.time()
        job_interval = 1.0 / settings.SAMPLE_FPS
        h_img, w_img = frame_shape[:2]

        if not head_boxes:
            empty_detections = sv.Detections.empty()
            tracked_detections = self.byte_tracker.update_with_detections(empty_detections)
            return tracked_detections, []

        xyxy_list = []
        confidence_list = []

        for (x1, y1, x2, y2), conf in head_boxes:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            if (x2 - x1) < settings.MIN_FACE_SIZE or (y2 - y1) < settings.MIN_FACE_SIZE:
                continue
            xyxy_list.append([x1, y1, x2, y2])
            confidence_list.append(conf)

        if not xyxy_list:
            empty_detections = sv.Detections.empty()
            tracked_detections = self.byte_tracker.update_with_detections(empty_detections)
            return tracked_detections, []

        detections = sv.Detections(
            xyxy=np.array(xyxy_list),
            confidence=np.array(confidence_list)
        )

        tracked_detections = self.byte_tracker.update_with_detections(detections)

        pending_jobs = []

        if tracked_detections.tracker_id is not None:
            for idx, tracker_id in enumerate(tracked_detections.tracker_id):
                t_id = int(tracker_id)
                bbox = tracked_detections.xyxy[idx].astype(int)
                x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(w_img, bbox[2]), min(h_img, bbox[3])

                if t_id not in self.track_states:
                    matched_pid, matched_lbl = self._spatial_reid_match([x1, y1, x2, y2])
                    if matched_pid and matched_lbl:
                        self.track_states[t_id] = TrackState.RESOLVED
                        clean_lbl = matched_lbl.split(' ', 1)[-1] if ' ' in matched_lbl else matched_lbl
                        self.track_identities[t_id] = {
                            "label": f"#{t_id} {clean_lbl}",
                            "profile_id": matched_pid,
                            "status": "resolved"
                        }
                    else:
                        self.track_states[t_id] = TrackState.PENDING
                        self.track_identities[t_id] = {
                            "label": f"Track #{t_id}",
                            "profile_id": "Unknown",
                            "status": "pending"
                        }
                else:
                    ident = self.track_identities.get(t_id, {})
                    if ident.get("status") == "resolved" and ident.get("profile_id") not in ("Unknown", "Pending"):
                        self.spatial_memory.append({
                            "bbox": [x1, y1, x2, y2],
                            "profile_id": ident.get("profile_id", ""),
                            "label": ident.get("label", ""),
                            "timestamp": now
                        })

                # Throttle recognition jobs per PENDING track to SAMPLE_FPS rate
                if self.track_states[t_id] == TrackState.PENDING:
                    last_job = self.last_job_times.get(t_id, 0.0)
                    if (now - last_job) >= job_interval:
                        self.last_job_times[t_id] = now
                        pending_jobs.append((t_id, (x1, y1, x2, y2)))

        # Clean up stale track IDs
        active_ids = set(tracked_detections.tracker_id.tolist()) if tracked_detections.tracker_id is not None else set()
        stale_ids = [tid for tid in list(self.track_states.keys()) if tid not in active_ids]
        for tid in stale_ids:
            self.track_states.pop(tid, None)
            self.track_identities.pop(tid, None)
            self.track_low_conf_counts.pop(tid, None)
            self.last_job_times.pop(tid, None)

        return tracked_detections, pending_jobs

    def set_track_identity(self, track_id: int, profile_id: str, label: str, is_new_visit: bool):
        """
        Marks a track_id as RESOLVED and locks its label overlay for the track's lifetime.
        """
        self.track_states[track_id] = TrackState.RESOLVED
        self.track_identities[track_id] = {
            "label": label,
            "profile_id": profile_id,
            "status": "resolved"
        }

    def check_visit_cooldown(self, profile_id: str) -> bool:
        """
        Returns True if profile_id is in cooldown (recent event suppressed).
        Returns False if it's a new visit stay.
        """
        now = datetime.now()
        if profile_id in self.profile_cooldowns:
            elapsed = now - self.profile_cooldowns[profile_id]
            if elapsed < timedelta(minutes=settings.VISIT_COOLDOWN_MINS):
                return True  # Cooldown active, suppress duplicate event alert
        
        # Update cooldown timestamp
        self.profile_cooldowns[profile_id] = now
        return False
