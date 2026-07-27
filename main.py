import os
import asyncio
import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import settings
from database.mongo import mongo_db
from database.storage import crop_storage
from core.head_detector import get_global_head_detector
from core.recognition import faiss_manager, RecognitionWorker, get_face_detector
from controllers.stream_controller import stream_controller

from api.profile_router import router as profile_router
from api.stream_router import router as stream_router
from api.log_router import router as log_router
from views.profile_view import router as profile_view_router
from views.stream_view import router as stream_view_router
from views.log_view import router as log_view_router

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("main_service")

recognition_worker: float = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global recognition_worker
    logger.info("Initializing Real-Time CCTV Head Detection & Face Recognition Service (MVC Architecture)...")

    # 1. Connect MongoDB
    try:
        await mongo_db.connect()
    except Exception as e:
        logger.warning(f"MongoDB connection skipped/warning: {e}")

    # 2. Sync uploads folder & initialize FAISS Vector Index
    if os.path.exists("./uploads"):
        try:
            from scripts.enroll_uploads import enroll_uploads
            await enroll_uploads("./uploads")
        except Exception as e:
            logger.error(f"Error syncing uploads folder: {e}")

    faiss_manager.initialize()
    if not faiss_manager.load_from_disk():
        try:
            db_profiles = await mongo_db.load_all_profiles()
            for p in db_profiles:
                pid = str(p.get("profile_id", ""))
                name = str(p.get("name", pid))
                embeddings = p.get("embeddings")
                if isinstance(embeddings, list):
                    for vec in embeddings:
                        faiss_manager.add_vector(np.array(vec, dtype=np.float32), pid, name)
            faiss_manager.save_to_disk()
        except Exception as e:
            logger.warning(f"Could not load profiles from MongoDB into FAISS: {e}")

    # 3. Pre-load ONNX Head Detector session & InsightFace detector
    get_global_head_detector()
    face_app = get_face_detector()

    # 4. Create Async Job Queue & Start Recognition Workers
    recognition_job_queue = asyncio.Queue(maxsize=100)
    stream_controller.set_job_queue(recognition_job_queue)

    recognition_worker = RecognitionWorker(recognition_job_queue, face_app)
    await recognition_worker.start_worker_pool(num_workers=2)

    # 5. Schedule daily retention cleanup task
    asyncio.create_task(periodic_retention_cleanup())

    # 6. Auto-start RTSP streams configured in settings / .env
    if settings.RTSP_STREAMS.strip():
        loop = asyncio.get_event_loop()
        for idx, rtsp_url in enumerate(settings.RTSP_STREAMS.split(","), start=1):
            rtsp_url = rtsp_url.strip()
            if not rtsp_url:
                continue
            camera_id = f"cam_{idx}"
            try:
                stream_controller.add_stream(camera_id, rtsp_url, loop)
                logger.info(f"Auto-started stream worker: {camera_id} -> {rtsp_url}")
            except Exception as e:
                logger.error(f"Error starting stream {camera_id}: {e}")

    logger.info("Service initialized successfully.")
    logger.info(f"Live Monitor UI: http://{settings.HOST}:{settings.PORT}/stream")
    logger.info(f"Enrollment Management UI: http://{settings.HOST}:{settings.PORT}/profiles-ui")
    logger.info(f"API Swagger Docs: http://{settings.HOST}:{settings.PORT}/docs")
    yield

    # Shutdown logic
    logger.info("Shutting down service...")
    if recognition_worker is not None:
        recognition_worker.running = False

    stream_controller.stop_all_streams()
    await mongo_db.close()
    logger.info("Shutdown complete.")


app = FastAPI(title="Real-Time Head Detection & Face Recognition Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file mounts
os.makedirs(settings.CROP_DIR, exist_ok=True)
app.mount("/crops", StaticFiles(directory=settings.CROP_DIR), name="crops")

os.makedirs("./uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="./uploads"), name="uploads")


async def periodic_retention_cleanup():
    while True:
        try:
            crop_storage.prune_old_crops(settings.RETENTION_DAYS)
        except Exception as e:
            logger.error(f"Error in retention cleanup: {e}")
        await asyncio.sleep(86400)


# Register MVC Routers
app.include_router(profile_router)
app.include_router(stream_router)
app.include_router(log_router)
app.include_router(profile_view_router)
app.include_router(stream_view_router)
app.include_router(log_view_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT, workers=1)
