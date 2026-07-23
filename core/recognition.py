import os
import faiss
import pickle
import numpy as np
import asyncio
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any
from config import settings
from database.mongo import mongo_db
from database.storage import crop_storage
from insightface.app import FaceAnalysis

logger = logging.getLogger("recognition")

class FAISSIndexManager:
    def __init__(self):
        self.dimension = 512
        self.index: Optional[faiss.Index] = None
        self.faiss_ids: List[str] = []
        self.lock = threading.Lock()

    def initialize(self):
        with self.lock:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.faiss_ids = []

    def add_vector(self, embedding: np.ndarray, profile_id: str):
        """Adds a single normalized vector to FAISS index"""
        with self.lock:
            if self.index is None:
                self.index = faiss.IndexFlatIP(self.dimension)
            norm_vector = embedding / np.linalg.norm(embedding)
            self.index.add(norm_vector.reshape(1, -1))
            self.faiss_ids.append(profile_id)

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
                    pickle.dump(self.faiss_ids, f)

    def load_from_disk(self):
        with self.lock:
            if os.path.exists(settings.FAISS_INDEX_PATH) and os.path.exists(settings.KNOWN_IDS_PATH):
                try:
                    self.index = faiss.read_index(settings.FAISS_INDEX_PATH)
                    with open(settings.KNOWN_IDS_PATH, 'rb') as f:
                        self.faiss_ids = pickle.load(f)
                    logger.info(f"Loaded FAISS index from disk ({len(self.faiss_ids)} total vectors).")
                    return True
                except Exception as e:
                    logger.error(f"Failed to load FAISS from disk: {e}")
            return False

faiss_manager = FAISSIndexManager()

class RecognitionWorker:
    def __init__(self, job_queue: asyncio.Queue, app_detector: FaceAnalysis):
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

    async def _process_job(self, job: Dict[str, Any]):
        camera_id = job["camera_id"]
        track_id = job["track_id"]
        crop = job["crop"]
        bbox = job["bbox"]
        stream_worker = job["stream_worker"]

        # Run ArcFace Feature Extraction
        faces = self.detector.get(crop, max_num=1)
        if not faces or len(faces) == 0:
            return

        emb = faces[0].embedding
        sim, best_profile_id = faiss_manager.search(emb)

        now = datetime.now(timezone.utc)
        crop_file_path = crop_storage.save_crop(crop, best_profile_id, track_id)

        if sim >= settings.HIGH_CONF_THRESH:
            # Match Known Profile
            in_cooldown = stream_worker.track_manager.check_visit_cooldown(best_profile_id)
            label = f"{best_profile_id} ({sim:.2f})"
            stream_worker.track_manager.set_track_identity(track_id, best_profile_id, label, is_new_visit=not in_cooldown)

            # Progressive Learning: If sim is between 0.55 and 0.85, add vector to multi-vector gallery
            if settings.HIGH_CONF_THRESH <= sim < 0.85:
                faiss_manager.add_vector(emb, best_profile_id)
                await mongo_db.save_or_update_profile(best_profile_id, emb.tolist())

            if not in_cooldown:
                event_doc = {
                    "camera_id": camera_id,
                    "track_id": int(track_id),
                    "profile_id": best_profile_id,
                    "confidence": sim,
                    "event_type": "KNOWN_IDENTITY",
                    "timestamp": now,
                    "bbox": [int(x) for x in bbox],
                    "crop_path": crop_file_path,
                    "review_required": False
                }
                await mongo_db.save_detection_event(event_doc)
                logger.info(f"[{camera_id}] KNOWN_IDENTITY Logged: {best_profile_id} ({sim:.2f})")

        elif settings.LOW_CONF_THRESH <= sim < settings.HIGH_CONF_THRESH:
            # Uncertain / Review Candidate
            label = f"Review ({sim:.2f})"
            stream_worker.track_manager.set_track_identity(track_id, "Review_Required", label, is_new_visit=True)

            event_doc = {
                "camera_id": camera_id,
                "track_id": int(track_id),
                "profile_id": best_profile_id,
                "confidence": sim,
                "event_type": "UNCERTAIN_CANDIDATE",
                "timestamp": now,
                "bbox": [int(x) for x in bbox],
                "crop_path": crop_file_path,
                "review_required": True
            }
            await mongo_db.save_detection_event(event_doc)
            logger.info(f"[{camera_id}] UNCERTAIN Logged: {best_profile_id} ({sim:.2f})")

        else:
            # Auto-Enroll New Identity (sim < 0.40)
            self.counter += 1
            new_profile_id = f"VISITOR_{now.strftime('%m%d')}_{self.counter}"
            label = f"{new_profile_id} ({sim:.2f})"
            
            # Save new vector to FAISS and MongoDB
            faiss_manager.add_vector(emb, new_profile_id)
            await mongo_db.save_or_update_profile(new_profile_id, emb.tolist())

            stream_worker.track_manager.set_track_identity(track_id, new_profile_id, label, is_new_visit=True)
            
            # Save crop with new profile_id folder name
            new_crop_path = crop_storage.save_crop(crop, new_profile_id, track_id)

            event_doc = {
                "camera_id": camera_id,
                "track_id": int(track_id),
                "profile_id": new_profile_id,
                "confidence": sim,
                "event_type": "NEW_IDENTITY",
                "timestamp": now,
                "bbox": [int(x) for x in bbox],
                "crop_path": new_crop_path,
                "review_required": False
            }
            await mongo_db.save_detection_event(event_doc)
            logger.info(f"[{camera_id}] NEW_IDENTITY Auto-Enrolled & Logged: {new_profile_id}")

        # Persist FAISS index periodically
        faiss_manager.save_to_disk()
