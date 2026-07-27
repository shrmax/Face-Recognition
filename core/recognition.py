import os
import faiss
import pickle
import numpy as np
import asyncio
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, TypedDict, Union, TYPE_CHECKING
from config import settings
from database.mongo import mongo_db
from database.storage import crop_storage
from core.quality import quality_filter
from insightface.app import FaceAnalysis

if TYPE_CHECKING:
    from core.stream_worker import StreamWorker

class JobData(TypedDict):
    camera_id: str
    track_id: int
    crop: np.ndarray
    bbox: List[int]
    stream_worker: 'StreamWorker'

logger = logging.getLogger("recognition")

_face_detector_instance: Optional[FaceAnalysis] = None
_face_detector_lock = threading.Lock()


def get_face_detector() -> FaceAnalysis:
    """Returns singleton InsightFace FaceAnalysis instance for face detection & embedding extraction."""
    global _face_detector_instance
    if _face_detector_instance is None:
        with _face_detector_lock:
            if _face_detector_instance is None:
                app = FaceAnalysis(
                    name='buffalo_m',
                    providers=['CUDAExecutionProvider', 'CoreMLExecutionProvider', 'CPUExecutionProvider'],
                    allowed_modules=['detection', 'recognition']
                )
                app.prepare(
                    ctx_id=0,
                    det_size=(settings.DET_WIDTH, settings.DET_HEIGHT),
                    det_thresh=settings.DET_THRESH
                )
                _face_detector_instance = app
                logger.info("InsightFace detector initialized with det_size=(%d,%d)", settings.DET_WIDTH, settings.DET_HEIGHT)
    return _face_detector_instance


class FAISSIndexManager:
    def __init__(self) -> None:
        self.dimension = 512
        self.index: Optional[faiss.Index] = None
        self.faiss_ids: List[str] = []
        self.profile_names: Dict[str, str] = {}
        self.lock = threading.Lock()

    def initialize(self) -> None:
        with self.lock:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.faiss_ids = []
            self.profile_names = {}

    def add_vector(self, embedding: np.ndarray, profile_id: str, name: Optional[str] = None) -> None:
        """Adds a single normalized vector to FAISS index."""
        with self.lock:
            if self.index is None:
                self.index = faiss.IndexFlatIP(self.dimension)
            norm_vector = embedding / (np.linalg.norm(embedding) + 1e-6)
            self.index.add(norm_vector.reshape(1, -1).astype(np.float32))
            self.faiss_ids.append(profile_id)
            if name:
                self.profile_names[profile_id] = name

    def get_name(self, profile_id: str) -> str:
        with self.lock:
            return self.profile_names.get(profile_id, profile_id)

    def search(self, query_embedding: np.ndarray) -> Tuple[float, str]:
        """Searches query vector against FAISS index. Returns (similarity, profile_id)."""
        with self.lock:
            if self.index is None or self.index.ntotal == 0 or not self.faiss_ids:
                return 0.0, "Unknown"

            qnorm = query_embedding / (np.linalg.norm(query_embedding) + 1e-6)
            distances, indices = self.index.search(qnorm.reshape(1, -1).astype(np.float32), k=1)

            best_sim = float(distances[0][0])
            best_idx = int(indices[0][0])

            if 0 <= best_idx < len(self.faiss_ids):
                return best_sim, self.faiss_ids[best_idx]
            return 0.0, "Unknown"

    def save_to_disk(self) -> None:
        with self.lock:
            if self.index is not None:
                faiss.write_index(self.index, settings.FAISS_INDEX_PATH)
                with open(settings.KNOWN_IDS_PATH, 'wb') as f:
                    pickle.dump({"ids": self.faiss_ids, "names": self.profile_names}, f)

    def load_from_disk(self) -> bool:
        with self.lock:
            if os.path.exists(settings.FAISS_INDEX_PATH) and os.path.exists(settings.KNOWN_IDS_PATH):
                try:
                    self.index = faiss.read_index(settings.FAISS_INDEX_PATH)
                    with open(settings.KNOWN_IDS_PATH, 'rb') as f:
                        data = pickle.load(f)
                        if isinstance(data, dict):
                            self.faiss_ids = data.get("ids", [])
                            self.profile_names = data.get("names", {})
                        else:
                            self.faiss_ids = data
                            self.profile_names = {}
                    logger.info("Loaded FAISS index from disk (%d total vectors).", len(self.faiss_ids))
                    return True
                except Exception as e:
                    logger.error("Failed to load FAISS from disk: %s", e)
            return False


faiss_manager = FAISSIndexManager()


class RecognitionWorker:
    def __init__(self, job_queue: asyncio.Queue[JobData], app_detector: Optional[FaceAnalysis] = None) -> None:
        self.job_queue = job_queue
        self.detector = app_detector or get_face_detector()
        self.running = True

    async def start_worker_pool(self, num_workers: int = 2) -> None:
        logger.info("Starting %d async face recognition worker tasks...", num_workers)
        for i in range(num_workers):
            asyncio.create_task(self._worker_loop(i))

    async def _worker_loop(self, worker_id: int) -> None:
        loop = asyncio.get_running_loop()
        while self.running:
            try:
                job = await self.job_queue.get()
                # Execute heavy CPU ONNX model inference off the main asyncio thread loop!
                await loop.run_in_executor(None, self._process_job_sync, job)
                self.job_queue.task_done()
            except Exception as e:
                logger.error("Error in recognition worker %d: %s", worker_id, e)
                await asyncio.sleep(0.05)

    def _process_job_sync(self, job: JobData) -> None:
        camera_id = job["camera_id"]
        track_id = job["track_id"]
        head_crop = job["crop"]
        bbox = job["bbox"]
        stream_worker = job["stream_worker"]

        if head_crop is None or head_crop.size == 0 or head_crop.shape[0] == 0 or head_crop.shape[1] == 0:
            return

        if self.detector is None:
            return

        try:
            faces = self.detector.get(head_crop)
        except Exception as e:
            logger.error("[%s] InsightFace analysis error on track #%d: %s", camera_id, track_id, e)
            return

        if not faces:
            return  # Face not clearly visible yet — stays PENDING, retries next cycle

        face = max(faces, key=lambda f: float(getattr(f, 'det_score', 0.0)))
        det_score = float(getattr(face, 'det_score', 0.0))

        if det_score < settings.MIN_DET_SCORE:
            logger.debug("[%s] Track #%d face det_score too low (%.2f < %.2f), skipping",
                         camera_id, track_id, det_score, settings.MIN_DET_SCORE)
            return  # Face turned away/back of head — stays PENDING, retries when face turns toward camera

        # Validate full frontal face landmark geometry (requires 5 keypoints: eyes, nose, mouth)
        kps = getattr(face, 'kps', None)
        if kps is None or len(kps) < 5:
            logger.debug("[%s] Track #%d face missing full keypoints, skipping", camera_id, track_id)
            return

        left_eye, right_eye, nose, left_mouth, right_mouth = kps[:5]
        eye_dist = float(np.linalg.norm(left_eye - right_eye))
        
        # 1. Eye Distance Check (at least 12px for full face resolution)
        if eye_dist < 12.0:
            logger.debug("[%s] Track #%d eye distance too small (%.1fpx < 12.0px), skipping", camera_id, track_id, eye_dist)
            return

        # 2. Roll Tilt Check (sideways head tilt)
        eye_diff_y = abs(float(left_eye[1] - right_eye[1]))
        if (eye_diff_y / (eye_dist + 1e-5)) > 0.35:
            logger.debug("[%s] Track #%d sideways roll tilt rejected, skipping", camera_id, track_id)
            return

        # 3. Vertical Landmark Alignment & Pitch Check (eyes above nose, nose above mouth)
        eye_center_y = (float(left_eye[1]) + float(right_eye[1])) / 2.0
        mouth_center_y = (float(left_mouth[1]) + float(right_mouth[1])) / 2.0
        if not (eye_center_y < float(nose[1]) < mouth_center_y):
            logger.debug("[%s] Track #%d vertical landmark misalignment (tilted/profile face), skipping", camera_id, track_id)
            return

        d_eye_nose = float(nose[1]) - eye_center_y
        d_nose_mouth = mouth_center_y - float(nose[1])
        d_eye_mouth = mouth_center_y - eye_center_y

        # Pitch Down check (head bowed down): nose is too close to mouth or pitch ratio > 1.45
        if (d_nose_mouth / (d_eye_mouth + 1e-5)) < 0.35 or (d_eye_nose / (d_nose_mouth + 1e-5)) > 1.45:
            logger.debug("[%s] Track #%d downward pitch tilt (looking down), skipping", camera_id, track_id)
            return

        # Pitch Up check (head tilted up): nose too close to eyes
        if (d_eye_nose / (d_eye_mouth + 1e-5)) < 0.25 or (d_eye_nose / (d_nose_mouth + 1e-5)) < 0.55:
            logger.debug("[%s] Track #%d upward pitch tilt (looking up), skipping", camera_id, track_id)
            return

        # 4. Horizontal Pose Symmetry Check (Nose centered relative to eyes, max 0.45 offset)
        eye_center_x = (float(left_eye[0]) + float(right_eye[0])) / 2.0
        eye_span_x = abs(float(right_eye[0]) - float(left_eye[0])) + 1e-5
        nose_offset_x = abs(float(nose[0]) - eye_center_x) / eye_span_x
        if nose_offset_x > 0.45:
            logger.debug("[%s] Track #%d side-profile pose rejected (nose_offset=%.2f > 0.45), skipping", camera_id, track_id, nose_offset_x)
            return

        fx1, fy1, fx2, fy2 = face.bbox.astype(int)
        fx1, fy1 = max(0, fx1), max(0, fy1)
        face_crop = head_crop[fy1:fy2, fx1:fx2]

        is_passed, blur_score, reason = quality_filter.evaluate_quality(face_crop)
        if not is_passed:
            logger.debug("[%s] Track #%d face quality rejected: %s", camera_id, track_id, reason)
            return  # stays PENDING, retries next cycle when posture improves

        emb = face.embedding
        sim, best_profile_id = faiss_manager.search(emb)
        now = datetime.now(timezone.utc)
        logger.info("[%s] Track #%d FAISS match: profile='%s', sim=%.3f (thresh=%.2f)",
                    camera_id, track_id, best_profile_id, sim, settings.HIGH_CONF_THRESH)

        if sim >= settings.HIGH_CONF_THRESH:
            name = faiss_manager.get_name(best_profile_id)
            label = f"#{track_id} {name} ({sim:.2f})"
            stream_worker.track_manager.set_track_identity(track_id, best_profile_id, label, is_new_visit=True)

            # Save face crop
            crop_file_path = crop_storage.save_crop(face_crop, best_profile_id, track_id)
            event_doc = {
                "camera_id": camera_id,
                "track_id": track_id,
                "profile_id": best_profile_id,
                "confidence": sim,
                "event_type": "KNOWN_IDENTITY",
                "timestamp": now,
                "bbox": [int(x) for x in bbox],
                "crop_path": crop_file_path,
                "review_required": False
            }
            if stream_worker.loop and stream_worker.loop.is_running():
                asyncio.run_coroutine_threadsafe(mongo_db.save_detection_event(event_doc), stream_worker.loop)

            logger.info("[%s] KNOWN IDENTITY RESOLVED: Track #%d -> %s (sim=%.2f)", camera_id, track_id, name, sim)
            return

        # Low confidence match: count evaluation attempts
        manager = stream_worker.track_manager
        current_low_count = manager.track_low_conf_counts.get(track_id, 0) + 1
        manager.track_low_conf_counts[track_id] = current_low_count

        if current_low_count >= settings.MAX_EVAL_ATTEMPTS:
            # Resolve as Unknown after MAX_EVAL_ATTEMPTS clear quality evaluations fail to match known profiles
            label = f"#{track_id} Unknown"
            manager.set_track_identity(track_id, "Unknown", label, is_new_visit=False)
            logger.info("[%s] Track #%d RESOLVED as Unknown after %d evaluations", camera_id, track_id, current_low_count)
