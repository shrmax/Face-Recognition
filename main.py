import asyncio
import logging
import time
from typing import Dict, Optional, List
import numpy as np
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import settings
from database.mongo import mongo_db
from database.storage import crop_storage
from core.stream_worker import StreamWorker, get_face_detector
from core.recognition import faiss_manager, RecognitionWorker

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("main_service")

# Global Stream & Worker Registry
active_streams: Dict[str, StreamWorker] = {}
recognition_job_queue: Optional[asyncio.Queue] = None
recognition_worker: Optional[RecognitionWorker] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global recognition_job_queue, recognition_worker
    logger.info("Initializing Real-Time Face Recognition Service...")
    
    # 1. Connect MongoDB
    await mongo_db.connect()
    
    # 2. Sync uploads folder & initialize FAISS Vector Index FIRST
    if os.path.exists("./uploads"):
        try:
            from scripts.enroll_uploads import enroll_uploads
            await enroll_uploads("./uploads")
        except Exception as e:
            logger.error(f"Error syncing uploads folder: {e}")

    faiss_manager.initialize()
    db_profiles = await mongo_db.load_all_profiles()
    for p in db_profiles:
        pid = str(p.get("profile_id", ""))
        name = str(p.get("name", pid))
        embeddings = p.get("embeddings")
        if isinstance(embeddings, list):
            for vec in embeddings:
                faiss_manager.add_vector(np.array(vec, dtype=np.float32), pid, name)
    faiss_manager.save_to_disk()
    logger.info(f"Successfully loaded FAISS index with {len(faiss_manager.faiss_ids)} vectors from MongoDB ({len(db_profiles)} profiles).")
        
    # 3. Create Async Job Queue & Start Recognition Workers
    recognition_job_queue = asyncio.Queue(maxsize=100)
    detector = get_face_detector()
    from core.person_detector import get_person_detector
    get_person_detector()
    recognition_worker = RecognitionWorker(recognition_job_queue, detector)
    await recognition_worker.start_worker_pool(num_workers=2)
    
    # 4. Schedule daily retention cleanup task
    asyncio.create_task(periodic_retention_cleanup())

    # 5. Auto-start RTSP streams from .env config
    if settings.RTSP_STREAMS.strip():
        loop = asyncio.get_event_loop()
        for idx, rtsp_url in enumerate(settings.RTSP_STREAMS.split(","), start=1):
            rtsp_url = rtsp_url.strip()
            if not rtsp_url:
                continue
            camera_id = f"cam_{idx}"
            worker = StreamWorker(camera_id, rtsp_url, recognition_job_queue, loop)
            active_streams[camera_id] = worker
            logger.info(f"Auto-started stream: {camera_id} -> {rtsp_url}")
            logger.info(f"  WebSocket URL: ws://{settings.HOST}:{settings.PORT}/face/ws?camera_id={camera_id}")

    logger.info("Service initialized successfully.")
    logger.info(f"API docs: http://{settings.HOST}:{settings.PORT}/docs")
    for cam_id in active_streams:
        logger.info(f"Stream '{cam_id}' WebSocket: ws://{settings.HOST}:{settings.PORT}/face/ws?camera_id={cam_id}")
    yield

    # Shutdown logic
    logger.info("Shutting down service...")
    if recognition_worker is not None:
        recognition_worker.running = False
        
    for cam_id, worker in list(active_streams.items()):
        worker.stop()
        
    await mongo_db.close()
    logger.info("Shutdown complete.")

app = FastAPI(title="Real-Time Face Recognition Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve crop images static files
import os
os.makedirs(settings.CROP_DIR, exist_ok=True)
app.mount("/crops", StaticFiles(directory=settings.CROP_DIR), name="crops")

router = APIRouter()

async def periodic_retention_cleanup():
    while True:
        try:
            crop_storage.prune_old_crops(settings.RETENTION_DAYS)
        except Exception as e:
            logger.error(f"Error in retention cleanup: {e}")
        await asyncio.sleep(86400)  # Run once every 24 hours

from fastapi.responses import HTMLResponse

# ======================
# REST API & STREAM ENDPOINTS
# ======================

@router.get("/stream", response_class=HTMLResponse)
async def get_stream_page(camera_id: str = "cam_1"):
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Real-Time Face Recognition Stream - {camera_id}</title>
        <style>
            body {{
                margin: 0;
                padding: 20px;
                background-color: #0f172a;
                color: #f8fafc;
                font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
                display: flex;
                flex-direction: column;
                align-items: center;
                min-height: 100vh;
            }}
            h1 {{
                margin-bottom: 5px;
                font-size: 1.8rem;
                font-weight: 600;
                color: #38bdf8;
            }}
            .subtitle {{
                color: #94a3b8;
                font-size: 0.95rem;
                margin-bottom: 20px;
            }}
            .video-container {{
                position: relative;
                max-width: 960px;
                width: 100%;
                background: #1e293b;
                border-radius: 12px;
                box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
                overflow: hidden;
                border: 1px solid #334155;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 480px;
            }}
            img {{
                width: 100%;
                height: auto;
                display: block;
            }}
            .status-badge {{
                position: absolute;
                top: 15px;
                left: 15px;
                background: rgba(15, 23, 42, 0.8);
                backdrop-filter: blur(8px);
                padding: 6px 14px;
                border-radius: 20px;
                font-size: 0.85rem;
                font-weight: 500;
                display: flex;
                align-items: center;
                gap: 8px;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }}
            .dot {{
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background-color: #ef4444;
            }}
            .dot.connected {{
                background-color: #10b981;
                box-shadow: 0 0 8px #10b981;
            }}
        </style>
    </head>
    <body>
        <h1>🎯 Real-Time Face Recognition</h1>
        <div class="subtitle">Camera ID: <strong>{camera_id}</strong></div>
        
        <div class="video-container">
            <div class="status-badge">
                <div class="dot" id="statusDot"></div>
                <span id="statusText">Connecting...</span>
            </div>
            <img id="streamFrame" src="" alt="Stream Feed" />
        </div>

        <script>
            const camera_id = "{camera_id}";
            const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
            const wsUrl = `${{wsProtocol}}//${{window.location.host}}/face/ws?camera_id=${{camera_id}}`;
            
            const imgEl = document.getElementById("streamFrame");
            const statusDot = document.getElementById("statusDot");
            const statusText = document.getElementById("statusText");

            function connect() {{
                const ws = new WebSocket(wsUrl);
                
                ws.onopen = () => {{
                    statusDot.classList.add("connected");
                    statusText.textContent = "LIVE";
                }};

                ws.onmessage = (event) => {{
                    try {{
                        const data = JSON.parse(event.data);
                        if (data.type === "frame" && data.frame) {{
                            imgEl.src = "data:image/jpeg;base64," + data.frame;
                        }}
                    }} catch (e) {{
                        console.error("Error parsing message", e);
                    }}
                }};

                ws.onclose = () => {{
                    statusDot.classList.remove("connected");
                    statusText.textContent = "Disconnected - Retrying...";
                    setTimeout(connect, 2000);
                }};

                ws.onerror = (err) => {{
                    console.error("WebSocket error:", err);
                    ws.close();
                }};
            }}

            connect();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@router.get("/health")
async def health_check():
    return {
        "status": "ok",
        "active_streams": len(active_streams),
        "stream_ids": list(active_streams.keys()),
        "faiss_registered_vectors": len(faiss_manager.faiss_ids) if faiss_manager.index else 0,
        "queue_size": recognition_job_queue.qsize() if recognition_job_queue else 0,
        "mongo_connected": mongo_db.db is not None
    }

@router.post("/streams/add")
async def add_stream(camera_id: str = Query(...), rtsp_url: str = Query(...)):
    if camera_id in active_streams:
        raise HTTPException(status_code=400, detail=f"Stream '{camera_id}' is already active.")
    
    if recognition_job_queue is None:
        raise HTTPException(status_code=503, detail="Service not initialized yet.")
    
    loop = asyncio.get_event_loop()
    worker = StreamWorker(camera_id, rtsp_url, recognition_job_queue, loop)
    active_streams[camera_id] = worker
    logger.info(f"Added and started stream worker for camera: {camera_id}")
    
    return {
        "status": "success",
        "camera_id": camera_id,
        "message": f"Stream worker started for {rtsp_url}"
    }

@router.delete("/streams/{camera_id}")
async def remove_stream(camera_id: str):
    if camera_id not in active_streams:
        raise HTTPException(status_code=404, detail=f"Stream '{camera_id}' not found.")
    
    worker = active_streams.pop(camera_id)
    worker.stop()
    logger.info(f"Stopped stream worker for camera: {camera_id}")
    
    return {
        "status": "success",
        "camera_id": camera_id,
        "message": "Stream worker stopped"
    }

@router.get("/logs")
async def get_logs(limit: int = 50, camera_id: Optional[str] = None):
    events = await mongo_db.get_recent_events(limit=limit, camera_id=camera_id)
    return {"status": "success", "count": len(events), "events": events}

# ======================
# WEBSOCKET STREAMING
# ======================

@router.websocket("/face/ws")
async def face_websocket(websocket: WebSocket, camera_id: str = Query(...), rtsp_url: Optional[str] = Query(None)):
    await websocket.accept()
    
    # Auto-register stream if missing and rtsp_url provided
    if camera_id not in active_streams:
        if not rtsp_url:
            await websocket.send_json({"type": "error", "message": f"Camera '{camera_id}' not active and no rtsp_url provided."})
            await websocket.close()
            return
        
        if recognition_job_queue is None:
            await websocket.send_json({"type": "error", "message": "Service not initialized yet."})
            await websocket.close()
            return
        
        loop = asyncio.get_event_loop()
        worker = StreamWorker(camera_id, rtsp_url, recognition_job_queue, loop)
        active_streams[camera_id] = worker
        logger.info(f"Auto-registered stream via WebSocket for: {camera_id}")
    else:
        worker = active_streams[camera_id]

    try:
        frame_count = 0
        while True:
            frame_b64 = worker.get_latest_frame_b64()
            if frame_b64:
                await websocket.send_json({
                    "type": "frame",
                    "camera_id": camera_id,
                    "frame": frame_b64
                })
                frame_count += 1
                
            await asyncio.sleep(0.033)  # ~30 FPS WebSocket stream rate
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for camera: {camera_id}")
    except Exception as e:
        logger.error(f"WebSocket error on camera {camera_id}: {e}")

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT, workers=1)
