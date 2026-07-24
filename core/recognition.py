import os
import faiss
import pickle
import numpy as np
import asyncio
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, TypedDict, TYPE_CHECKING
from config import settings
from database.mongo import mongo_db
from database.storage import crop_storage
from insightface.app import FaceAnalysis

if TYPE_CHECKING:
    from core.stream_worker import StreamWorker

class JobData(TypedDict):
    camera_id: str
    track_id: int
    crop: np.ndarray
    bbox: List[int]
    embedding: np.ndarray
    stream_worker: 'StreamWorker'

logger = logging.getLogger("recognition")

class FAISSIndexManager:
    def __init__(self):
        self.dimension = 512
        self.index: Optional[faiss.Index] = None
        self.faiss_ids: List[str] = []
        self.profile_names: Dict[str, str] = {}
        self.lock = threading.Lock()

    def initialize(self):
        with self.lock:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.faiss_ids = []
            self.profile_names = {}

    def add_vector(self, embedding: np.ndarray, profile_id: str, name: Optional[str] = None):
        """Adds a single normalized vector to FAISS index"""
        with self.lock:
            if self.index is None:
                self.index = faiss.IndexFlatIP(self.dimension)
            norm_vector = embedding / np.linalg.norm(embedding)
            self.index.add(norm_vector.reshape(1, -1))
            self.faiss_ids.append(profile_id)
            if name:
                self.profile_names[profile_id] = name

    def get_name(self, profile_id: str) -> str:
        with self.lock:
            return self.profile_names.get(profile_id, profile_id)

    def search(self, query_embedding: np.ndarray) -> Tuple[float, str]:
        """Searches query vector against FAISS index. Returns (similarity, profile_id)"""
        with self.lock:
            if self.index is None or self.index.ntotal == 0 or not self.faiss_ids:
                return 0.0, "Unknown"
            
            qnorm = query_embedding / np.linalg.norm(query_embedding)
            distances, indices = self.index.search(qnorm.reshape(1, -1), k=1)
            
            best_sim = float(distances[0][0])
            best_idx = indices[0][0]
            
            if best_idx >= 0 and best_idx < len(self.faiss_ids):
                return best_sim, self.faiss_ids[best_idx]
            return 0.0, "Unknown"

    def save_to_disk(self):
        with self.lock:
            if self.index is not None:
                faiss.write_index(self.index, settings.FAISS_INDEX_PATH)
                with open(settings.KNOWN_IDS_PATH, 'wb') as f:
                    pickle.dump({"ids": self.faiss_ids, "names": self.profile_names}, f)

    def load_from_disk(self):
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
                    logger.info(f"Loaded FAISS index from disk ({len(self.faiss_ids)} total vectors).")
                    return True
                except Exception as e:
                    logger.error(f"Failed to load FAISS from disk: {e}")
            return False

faiss_manager = FAISSIndexManager()

class RecognitionWorker:
    def __init__(self, job_queue: asyncio.Queue, app_detector: Optional[FaceAnalysis] = None):
        self.job_queue = job_queue
        self.detector = app_detector
        self.running = True
        self.counter = 1000

    async def start_worker_pool(self, num_workers: int = 2):
        logger.info(f"Starting {num_workers} async recognition worker tasks...")
        for i in range(num_workers):
            asyncio.create_task(self._worker_loop(i))

    async def _worker_loop(self, worker_id: int):
        while self.running:
            try:
                job = await self.job_queue.get()
                await self._process_job(job)
                self.job_queue.task_done()
            except Exception as e:
                logger.error(f"Error in recognition worker {worker_id}: {e}")
                await asyncio.sleep(0.05)

    async def _process_job(self, job: JobData):
        camera_id = job["camera_id"]
        track_id = job["track_id"]
        crop = job["crop"]
        bbox = job["bbox"]
        stream_worker = job["stream_worker"]
        raw_emb = job.get("embedding")

        if raw_emb is None:
            return

        emb = np.array(raw_emb, dtype=np.float32)
        sim, best_profile_id = faiss_manager.search(emb)

        now = datetime.now(timezone.utc)
        crop_file_path = crop_storage.save_crop(crop, best_profile_id, track_id)

        if sim >= settings.HIGH_CONF_THRESH:
            # Clear low confidence count for high confidence match
            stream_worker.track_manager.track_low_conf_counts.pop(track_id, None)
            
            # Match Known Profile
            in_cooldown = stream_worker.track_manager.check_visit_cooldown(best_profile_id)
            name = faiss_manager.get_name(best_profile_id)
            label = f"#{track_id} {name} ({sim:.2f})"
            stream_worker.track_manager.set_track_identity(track_id, best_profile_id, label, is_new_visit=not in_cooldown)

            # Progressive Learning: If sim is between 0.55 and 0.85, add vector to multi-vector gallery
            if settings.HIGH_CONF_THRESH <= sim < 0.85:
                faiss_manager.add_vector(emb, best_profile_id, name)
                await mongo_db.save_or_update_profile(best_profile_id, emb.tolist(), name)

            if not in_cooldown:
                event_doc = {
                    "camera_id": str(camera_id),
                    "track_id": int(track_id),
                    "profile_id": str(best_profile_id),
                    "confidence": float(sim),
                    "event_type": "KNOWN_IDENTITY",
                    "timestamp": now,
                    "bbox": [int(x) for x in bbox],
                    "crop_path": str(crop_file_path),
                    "review_required": False
                }
                await mongo_db.save_detection_event(event_doc)
                logger.info(f"[{camera_id}] KNOWN_IDENTITY Logged: Track #{track_id} {name} ({sim:.2f})")

        elif settings.LOW_CONF_THRESH <= sim < settings.HIGH_CONF_THRESH:
            # Clear low confidence count for candidate match
            stream_worker.track_manager.track_low_conf_counts.pop(track_id, None)

            # Uncertain / Review Candidate
            name = faiss_manager.get_name(best_profile_id)
            label = f"#{track_id} {name}? ({sim:.2f})"
            stream_worker.track_manager.set_track_identity(track_id, "Review_Required", label, is_new_visit=True)

            event_doc = {
                "camera_id": str(camera_id),
                "track_id": int(track_id),
                "profile_id": str(best_profile_id),
                "confidence": float(sim),
                "event_type": "UNCERTAIN_CANDIDATE",
                "timestamp": now,
                "bbox": [int(x) for x in bbox],
                "crop_path": str(crop_file_path),
                "review_required": True
            }
            await mongo_db.save_detection_event(event_doc)
            logger.info(f"[{camera_id}] UNCERTAIN Logged: Track #{track_id} {name} ({sim:.2f})")

        else:
            # Debounce auto-enrollment: Require 2 consecutive low-confidence attempts before creating a new visitor profile
            curr_count = stream_worker.track_manager.track_low_conf_counts.get(track_id, 0) + 1
            stream_worker.track_manager.track_low_conf_counts[track_id] = curr_count

            if curr_count < 2:
                label = f"#{track_id} Evaluating..."
                stream_worker.track_manager.set_track_identity(track_id, "Pending", label, is_new_visit=False)
                # Reset track state to PENDING so next frame retries evaluation
                stream_worker.track_manager.track_states[track_id] = "PENDING"
                logger.debug(f"[{camera_id}] Track #{track_id} low confidence attempt {curr_count}/2 (sim {sim:.2f}). Debouncing auto-enroll.")
                return

            # Auto-Enroll New Identity (sim < LOW_CONF_THRESH after 2+ consistent attempts)
            stream_worker.track_manager.track_low_conf_counts.pop(track_id, None)
            self.counter += 1
            new_profile_id = f"VISITOR_{now.strftime('%m%d')}_{self.counter}"
            label = f"#{track_id} {new_profile_id} ({sim:.2f})"
            
            # Save new vector to FAISS and MongoDB
            faiss_manager.add_vector(emb, new_profile_id, new_profile_id)
            await mongo_db.save_or_update_profile(new_profile_id, emb.tolist(), new_profile_id)

            stream_worker.track_manager.set_track_identity(track_id, new_profile_id, label, is_new_visit=True)
            
            # Save crop with new profile_id folder name
            new_crop_path = crop_storage.save_crop(crop, new_profile_id, track_id)

            event_doc = {
                "camera_id": str(camera_id),
                "track_id": int(track_id),
                "profile_id": str(new_profile_id),
                "confidence": float(sim),
                "event_type": "NEW_IDENTITY",
                "timestamp": now,
                "bbox": [int(x) for x in bbox],
                "crop_path": str(new_crop_path),
                "review_required": False
            }
            await mongo_db.save_detection_event(event_doc)
            logger.info(f"[{camera_id}] NEW_IDENTITY Auto-Enrolled & Logged: Track #{track_id} {new_profile_id}")

        # Persist FAISS index periodically
        faiss_manager.save_to_disk()
