import os
import cv2
import asyncio
import logging
import numpy as np
import faiss
import pickle
from insightface.app import FaceAnalysis
from motor.motor_asyncio import AsyncIOMotorClient
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("enroll_uploads")

async def enroll_uploads(uploads_dir: str = "./uploads"):
    """
    Scans the uploads/ directory for user folders, extracts InsightFace ArcFace
    embeddings, updates MongoDB, and rebuilds the FAISS vector index.
    """
    if not os.path.exists(uploads_dir):
        logger.warning(f"Uploads directory '{uploads_dir}' not found.")
        return

    logger.info(f"Scanning '{uploads_dir}' for profile enrollment...")
    client = AsyncIOMotorClient(settings.MONGO_URI, serverSelectionTimeoutMS=3000)
    db = client[settings.MONGO_DB_NAME]
    profiles_coll = db['face_profiles']

    # Check existing enrolled profiles & sample counts from MongoDB
    db_profiles = {}
    try:
        cursor = profiles_coll.find({}, {"profile_id": 1, "sample_count": 1, "img_count": 1, "embeddings": 1, "name": 1})
        async for p_doc in cursor:
            db_profiles[p_doc.get("profile_id")] = p_doc
    except Exception as e:
        logger.warning(f"Could not check existing profiles: {e}")

    folders = [f for f in sorted(os.listdir(uploads_dir)) if os.path.isdir(os.path.join(uploads_dir, f))]
    
    # Detect folders requiring embedding generation (new folder OR new images added)
    pending_folders = []
    for folder in folders:
        folder_path = os.path.join(uploads_dir, folder)
        profile_id = folder.lower().strip()
        img_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        p_doc = db_profiles.get(profile_id)
        if not p_doc or p_doc.get("img_count", 0) != len(img_files):
            pending_folders.append(profile_id)

    faiss_exists = os.path.exists(settings.FAISS_INDEX_PATH) and os.path.exists(settings.KNOWN_IDS_PATH)

    if faiss_exists and not pending_folders:
        logger.info("All upload folders and images are up-to-date in MongoDB & FAISS. Skipping enrollment.")
        client.close()
        return

    logger.info(f"Scanning '{uploads_dir}' for profile enrollment (pending profiles: {len(pending_folders)})...")
    app = FaceAnalysis(
        name='buffalo_s',
        providers=['CUDAExecutionProvider', 'CoreMLExecutionProvider', 'CPUExecutionProvider'],
        allowed_modules=['detection', 'recognition']
    )
    app.prepare(ctx_id=0, det_size=(settings.DET_WIDTH, settings.DET_HEIGHT), det_thresh=settings.DET_THRESH)

    faiss_vectors = []
    faiss_ids = []
    faiss_names = {}

    for folder in folders:
        folder_path = os.path.join(uploads_dir, folder)
        profile_id = folder.lower().strip()
        display_name = folder.capitalize().strip()
        img_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]

        p_doc = db_profiles.get(profile_id)
        if p_doc and faiss_exists and profile_id not in pending_folders:
            logger.info(f"Profile '{display_name}' is up-to-date. Loading existing embeddings...")
            for vec in p_doc.get("embeddings", []):
                vec_np = np.array(vec, dtype=np.float32)
                faiss_vectors.append(vec_np / np.linalg.norm(vec_np))
                faiss_ids.append(profile_id)
                faiss_names[profile_id] = p_doc.get("name", display_name)
            continue

        logger.info(f"Processing profile folder: '{folder}' -> Name: '{display_name}'")

        embeddings = []
        for file in sorted(img_files):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            faces = app.get(img)
            if not faces:
                logger.warning(f"  No face detected in {file}")
                continue

            face = max(faces, key=lambda f: float(getattr(f, 'det_score', 0.0)))
            det_score = float(getattr(face, 'det_score', 0.0))
            if det_score < 0.40:
                logger.warning(f"  Low confidence face detection in {file} (det_score={det_score:.2f} < 0.40), skipping")
                continue

            emb = face.embedding
            norm_vec = (emb / (np.linalg.norm(emb) + 1e-6)).astype(np.float32)

            # Deduplication check against already enrolled vectors for this profile
            is_duplicate = False
            for existing_vec in embeddings:
                existing_np = np.array(existing_vec, dtype=np.float32)
                sim = float(np.dot(norm_vec, existing_np))
                if sim > settings.DEDUP_SIM_THRESH:
                    logger.info(f"  Skipping near-duplicate image {file} (similarity={sim:.3f} > {settings.DEDUP_SIM_THRESH})")
                    is_duplicate = True
                    break

            if not is_duplicate:
                embeddings.append(norm_vec.tolist())
                faiss_vectors.append(norm_vec)
                faiss_ids.append(profile_id)
                faiss_names[profile_id] = display_name
                logger.info(f"  Enrolled unique face embedding from: {file}")

        if embeddings:
            doc = {
                'profile_id': profile_id,
                'name': display_name,
                'embeddings': embeddings,
                'sample_count': len(embeddings),
                'img_count': len(img_files),
                'status': 'active'
            }
            await profiles_coll.replace_one(
                {'profile_id': profile_id},
                doc,
                upsert=True
            )
            logger.info(f"  Saved profile '{display_name}' to MongoDB!")

    if faiss_vectors:
        index = faiss.IndexFlatIP(512)
        vectors_np = np.array(faiss_vectors, dtype=np.float32)
        index.add(vectors_np)
        faiss.write_index(index, settings.FAISS_INDEX_PATH)
        with open(settings.KNOWN_IDS_PATH, 'wb') as f:
            pickle.dump({'ids': faiss_ids, 'names': faiss_names}, f)
        logger.info(f"SUCCESS: Enrolled {len(faiss_names)} users into FAISS index & MongoDB!")

    client.close()

if __name__ == "__main__":
    asyncio.run(enroll_uploads())
