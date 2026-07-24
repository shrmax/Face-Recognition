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
    PROCESSED = "PROCESSED"

class StreamTrackManager:
    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        # sv.ByteTrack instance for persistent object tracking
        self.byte_tracker = sv.ByteTrack(
            track_activation_threshold=0.15,
            lost_track_buffer=60,
            minimum_matching_threshold=0.3,
            frame_rate=settings.SAMPLE_FPS
        )
        # track_id -> status dict
        self.track_states: Dict[int, str] = {}
        # track_id -> assigned profile_id & label info
        self.track_identities: Dict[int, Dict[str, str]] = {}
        # track_id -> count of consecutive low-confidence evaluations for debounced auto-enrollment
        self.track_low_conf_counts: Dict[int, int] = {}
        
        # Annotators for visualization overlay
        self.box_annotator = sv.BoxAnnotator(thickness=2, color_lookup=sv.ColorLookup.INDEX)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, color_lookup=sv.ColorLookup.INDEX)
        
        # Profile ID -> Last Event Timestamp (Visit Cooldown)
        self.profile_cooldowns: Dict[str, datetime] = {}

    def update(self, faces: Sequence[FaceObject], frame_shape: Tuple[int, ...]) -> Tuple[sv.Detections, List[Tuple[int, Tuple[int, int, int, int], np.ndarray]]]:
        """
        Updates ByteTrack with current frame's detected faces.
        Returns:
            - sv.Detections object for annotation overlay
            - List of pending jobs: [(track_id, (x1, y1, x2, y2), embedding)]
        """
        if not faces:
            # Update tracker with empty detections to maintain Kalman filter updates
            empty_detections = sv.Detections.empty()
            tracked_detections = self.byte_tracker.update_with_detections(empty_detections)
            return tracked_detections, []

        xyxy_list = []
        confidence_list = []
        embeddings_list = []

        h_img, w_img = frame_shape[:2]

        for face in faces:
            det_score = getattr(face, 'det_score', 0.0)
            if det_score < settings.DET_THRESH:
                continue
            bbox_arr = getattr(face, 'bbox', None)
            if bbox_arr is None:
                continue
            bbox = bbox_arr.astype(int)
            x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(w_img, bbox[2]), min(h_img, bbox[3])
            
            if (x2 - x1) < settings.MIN_FACE_SIZE or (y2 - y1) < settings.MIN_FACE_SIZE:
                continue
                
            xyxy_list.append([x1, y1, x2, y2])
            confidence_list.append(det_score)
            embeddings_list.append(face.embedding)

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
                
                # Check track state machine
                if t_id not in self.track_states:
                    self.track_states[t_id] = TrackState.PENDING
                    self.track_identities[t_id] = {
                        "label": f"Track #{t_id} (Processing...)",
                        "profile_id": "Unknown",
                        "status": "pending"
                    }
                    
                    bbox = tracked_detections.xyxy[idx].astype(int)
                    x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(w_img, bbox[2]), min(h_img, bbox[3])
                    
                    # Match tracked box back to nearest detection embedding via IoU
                    ious = [_compute_iou(bbox, dbox) for dbox in xyxy_list]
                    best_idx = int(np.argmax(ious)) if ious else 0
                    matched_embedding = embeddings_list[best_idx]
                    
                    pending_jobs.append((t_id, (x1, y1, x2, y2), matched_embedding))

        # Clean up stale track IDs
        active_ids = set(tracked_detections.tracker_id.tolist()) if tracked_detections.tracker_id is not None else set()
        stale_ids = [tid for tid in list(self.track_states.keys()) if tid not in active_ids]
        for tid in stale_ids:
            self.track_states.pop(tid, None)
            self.track_identities.pop(tid, None)
            self.track_low_conf_counts.pop(tid, None)

        return tracked_detections, pending_jobs

    def set_track_identity(self, track_id: int, profile_id: str, label: str, is_new_visit: bool):
        """
        Marks a track_id as PROCESSED and updates its label overlay.
        """
        self.track_states[track_id] = TrackState.PROCESSED
        self.track_identities[track_id] = {
            "label": label,
            "profile_id": profile_id,
            "status": "processed"
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
