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
        while self.running:
            try:
                job = await self.job_queue.get()
                await self._process_job(job)
                self.job_queue.task_done()
            except Exception as e:
                logger.error("Error in recognition worker %d: %s", worker_id, e)
                await asyncio.sleep(0.05)

    async def _process_job(self, job: JobData) -> None:
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

            # Persist crop and detection log in MongoDB
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
            await mongo_db.save_detection_event(event_doc)

            # Progressive vector gallery addition for moderate confidence matches
            if settings.HIGH_CONF_THRESH <= sim < 0.85:
                faiss_manager.add_vector(emb, best_profile_id, name)
                await mongo_db.save_or_update_profile(best_profile_id, emb.tolist(), name)
                faiss_manager.save_to_disk()

            logger.info("[%s] KNOWN IDENTITY RESOLVED: Track #%d -> %s (sim=%.2f)", camera_id, track_id, name, sim)
            return

        # Low confidence match: count evaluation attempts
        manager = stream_worker.track_manager
        current_low_count = manager.track_low_conf_counts.get(track_id, 0) + 1
        manager.track_low_conf_counts[track_id] = current_low_count

        if current_low_count >= 5:
            # Resolve as Unknown after 5 clear quality evaluations fail to match known profiles
            label = f"#{track_id} Unknown"
            manager.set_track_identity(track_id, "Unknown", label, is_new_visit=False)
            logger.info("[%s] Track #%d RESOLVED as Unknown after %d evaluations", camera_id, track_id, current_low_count)
