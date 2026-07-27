import os
import cv2
import shutil
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
import faiss

from config import settings
from database.mongo import mongo_db
from core.recognition import faiss_manager, get_face_detector

logger = logging.getLogger("profile_controller")


class ProfileController:
    """
    Controller handling business logic for person profiles and face image enrollment.
    Enforces strict single-face frame validation and embedding deduplication rules.
    """

    def __init__(self, uploads_dir: str = "./uploads"):
        self.uploads_dir = uploads_dir
        os.makedirs(self.uploads_dir, exist_ok=True)

    async def get_all_profiles(self) -> List[Dict[str, Union[str, int, List[str]]]]:
        """
        Retrieves all person profiles from disk and MongoDB with image URLs and embedding stats.
        """
        db_profiles: Dict[str, Dict[str, Union[str, int, List[List[float]]]]] = {}
        try:
            profiles_list = await mongo_db.load_all_profiles()
            for p in profiles_list:
                pid = str(p.get("profile_id", "")).lower().strip()
                if pid:
                    db_profiles[pid] = p
        except Exception as e:
            logger.warning(f"Error reading profiles from MongoDB: {e}")

        # Scan uploads directory for profile folders
        folders = [f for f in sorted(os.listdir(self.uploads_dir)) if os.path.isdir(os.path.join(self.uploads_dir, f))]
        results: List[Dict[str, Union[str, int, List[str]]]] = []

        # Combine folder profiles and DB profiles
        all_pids = sorted(list(set(folders).union(db_profiles.keys())))

        for pid in all_pids:
            folder_path = os.path.join(self.uploads_dir, pid)
            img_urls: List[str] = []
            if os.path.exists(folder_path):
                img_files = [f for f in sorted(os.listdir(folder_path)) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
                img_urls = [f"/uploads/{pid}/{f}" for f in img_files]

            p_doc = db_profiles.get(pid) or {}
            name = str(p_doc.get("name") or pid.capitalize())
            embeddings = p_doc.get("embeddings") or []
            raw_sample_count = p_doc.get("sample_count")
            sample_count = int(raw_sample_count) if raw_sample_count is not None else len(embeddings)
            last_seen = str(p_doc.get("last_seen") or "")

            results.append({
                "profile_id": pid,
                "name": name,
                "img_count": len(img_urls),
                "sample_count": sample_count,
                "images": img_urls,
                "last_seen": last_seen,
            })

        return results

    async def create_profile(self, profile_id: str, name: Optional[str] = None) -> Dict[str, Union[bool, str]]:
        """
        Creates a new profile directory and database record.
        """
        clean_id = profile_id.lower().strip()
        if not clean_id:
            return {"success": False, "message": "Profile ID cannot be empty."}

        display_name = (name or clean_id.capitalize()).strip()
        folder_path = os.path.join(self.uploads_dir, clean_id)
        os.makedirs(folder_path, exist_ok=True)

        return {
            "success": True,
            "profile_id": clean_id,
            "name": display_name,
            "message": f"Profile '{display_name}' created successfully."
        }

    async def enroll_face_image(
        self,
        profile_id: str,
        file_bytes: bytes,
        filename: str,
        name: Optional[str] = None
    ) -> Dict[str, Union[bool, str, float]]:
        """
        Validates uploaded image and enrolls face embedding:
          1. Decodes BGR frame.
          2. Validates face count:
             - 0 faces -> REJECTED (NO_FACE)
             - > 1 face -> REJECTED (MULTIPLE_FACES)
             - 1 face -> Proceeds to embedding extraction.
          3. Checks cosine similarity against existing profile embeddings:
             - > DEDUP_SIM_THRESH (0.95) -> SKIPPED (DUPLICATE)
             - <= 0.95 -> ENROLLED (FAISS + MongoDB + Disk).
        """
        clean_id = profile_id.lower().strip()
        if not clean_id:
            return {"success": False, "code": "INVALID_ID", "message": "Profile ID cannot be empty."}

        display_name = (name or clean_id.capitalize()).strip()

        # 1. Decode image bytes into OpenCV frame
        np_arr = np.frombuffer(file_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            return {"success": False, "code": "INVALID_IMAGE", "message": "Failed to decode image file."}

        # 2. InsightFace detection
        detector = get_face_detector()
        faces = detector.get(frame)

        num_faces = len(faces)
        if num_faces == 0:
            return {
                "success": False,
                "code": "NO_FACE_DETECTED",
                "message": "No face detected in the image. Please upload a clear photo with your face visible."
            }
        elif num_faces > 1:
            return {
                "success": False,
                "code": "MULTIPLE_FACES_DETECTED",
                "message": f"Multiple faces detected ({num_faces} faces found)! Please ensure exactly ONE person is in the frame."
            }

        # Select the single face
        face = faces[0]
        det_score = float(getattr(face, "det_score", 0.0))
        if det_score < 0.40:
            return {
                "success": False,
                "code": "LOW_CONFIDENCE",
                "message": f"Face detection confidence too low ({det_score:.2f} < 0.40). Please upload a clearer face photo."
            }

        # 3. Extract 512D ArcFace embedding & normalize
        emb = face.embedding
        norm_vec = (emb / (np.linalg.norm(emb) + 1e-6)).astype(np.float32)

        # 4. Save photo file to uploads folder
        folder_path = os.path.join(self.uploads_dir, clean_id)
        os.makedirs(folder_path, exist_ok=True)
        safe_filename = os.path.basename(filename)
        file_path = os.path.join(folder_path, safe_filename)
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        image_url = f"/uploads/{clean_id}/{safe_filename}"

        # 5. Check vector similarity deduplication against existing embeddings for this profile
        p_doc = await mongo_db.get_profile(clean_id)
        existing_embeddings: List[List[float]] = p_doc.get("embeddings", []) if p_doc else []

        max_sim = 0.0
        for vec_list in existing_embeddings:
            existing_vec = np.array(vec_list, dtype=np.float32)
            sim = float(np.dot(norm_vec, existing_vec))
            if sim > max_sim:
                max_sim = sim

        if max_sim > settings.DEDUP_SIM_THRESH:
            logger.info("Image '%s' saved to disk, but vector enrollment skipped (similarity=%.3f > %.2f)",
                        safe_filename, max_sim, settings.DEDUP_SIM_THRESH)
            return {
                "success": True,
                "code": "DUPLICATE_SKIPPED",
                "message": f"Image saved, but vector enrollment was skipped because a near-identical face embedding already exists (similarity = {max_sim:.3f} > {settings.DEDUP_SIM_THRESH:.2f}).",
                "image_url": image_url,
                "similarity": round(max_sim, 3)
            }

        # 6. Save unique vector to MongoDB & FAISS
        await mongo_db.save_or_update_profile(clean_id, norm_vec.tolist(), display_name)
        faiss_manager.add_vector(norm_vec, clean_id, display_name)
        faiss_manager.save_to_disk()

        logger.info("Enrolled unique face vector for profile '%s' (%s) from '%s'", clean_id, display_name, safe_filename)
        return {
            "success": True,
            "code": "ENROLLED",
            "message": f"Face successfully enrolled for '{display_name}'!",
            "image_url": image_url,
            "profile_id": clean_id,
            "name": display_name
        }

    async def delete_profile(self, profile_id: str) -> Dict[str, Union[bool, str]]:
        """
        Deletes profile directory and MongoDB record, then rebuilds FAISS index.
        """
        clean_id = profile_id.lower().strip()
        if not clean_id:
            return {"success": False, "message": "Profile ID cannot be empty."}

        # Remove uploads folder
        folder_path = os.path.join(self.uploads_dir, clean_id)
        if os.path.exists(folder_path):
            try:
                shutil.rmtree(folder_path)
            except Exception as e:
                logger.error(f"Failed to delete directory {folder_path}: {e}")

        # Remove from MongoDB
        await mongo_db.delete_profile(clean_id)

        # Rebuild FAISS index from remaining profiles
        faiss_manager.initialize()
        all_profiles = await mongo_db.load_all_profiles()
        for p in all_profiles:
            pid = str(p.get("profile_id", ""))
            name = str(p.get("name", pid))
            for vec in p.get("embeddings", []):
                faiss_manager.add_vector(np.array(vec, dtype=np.float32), pid, name)

        faiss_manager.save_to_disk()

        return {
            "success": True,
            "profile_id": clean_id,
            "message": f"Profile '{clean_id}' deleted successfully."
        }


profile_controller = ProfileController()
