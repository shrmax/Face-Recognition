import os
import asyncio
import logging
import time
from typing import Dict, Optional, List, Union, Tuple
import numpy as np
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from config import settings
from database.mongo import mongo_db
from database.storage import crop_storage
from core.stream_worker import StreamWorker
from core.head_detector import get_global_head_detector
from core.recognition import faiss_manager, RecognitionWorker, get_face_detector

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("main_service")

# Global Stream & Worker Registry
active_streams: Dict[str, StreamWorker] = {}
recognition_job_queue: Optional[asyncio.Queue[dict[str, Union[str, int, np.ndarray, float, List[int], StreamWorker]]]] = None
recognition_worker: Optional[RecognitionWorker] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global recognition_job_queue, recognition_worker
    logger.info("Initializing Real-Time CCTV Head Detection & Face Recognition Service...")

    # 1. Connect MongoDB
    try:
        await mongo_db.connect()
    except Exception as e:
        logger.warning("MongoDB connection skipped/warning: %s", e)

    # 2. Sync uploads folder & initialize FAISS Vector Index
    if os.path.exists("./uploads"):
        try:
            from scripts.enroll_uploads import enroll_uploads
            await enroll_uploads("./uploads")
        except Exception as e:
            logger.error("Error syncing uploads folder: %s", e)

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
            logger.warning("Could not load profiles from MongoDB into FAISS: %s", e)

    # 3. Pre-load ONNX Head Detector session & InsightFace detector
    get_global_head_detector()
    face_app = get_face_detector()

    # 4. Create Async Job Queue & Start Recognition Workers
    recognition_job_queue = asyncio.Queue(maxsize=100)
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
            worker = StreamWorker(camera_id, rtsp_url, job_queue=recognition_job_queue, loop=loop)
            worker.start()
            active_streams[camera_id] = worker
            logger.info("Auto-started stream worker: %s -> %s", camera_id, rtsp_url)

    logger.info("Service initialized successfully.")
    logger.info("API docs: http://%s:%d/docs", settings.HOST, settings.PORT)
    yield

    # Shutdown logic
    logger.info("Shutting down service...")
    if recognition_worker is not None:
        recognition_worker.running = False

    for cam_id, worker in list(active_streams.items()):
        worker.stop()

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

os.makedirs(settings.CROP_DIR, exist_ok=True)
app.mount("/crops", StaticFiles(directory=settings.CROP_DIR), name="crops")

router = APIRouter()


async def periodic_retention_cleanup():
    while True:
        try:
            crop_storage.prune_old_crops(settings.RETENTION_DAYS)
        except Exception as e:
            logger.error("Error in retention cleanup: %s", e)
        await asyncio.sleep(86400)


@router.get("/stream", response_class=HTMLResponse)
async def get_stream_page(camera_id: str = "cam_1"):
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Real-Time CCTV Head Detection & Face Recognition - {camera_id}</title>
        <style>
            * {{ box-sizing: border-box; margin: 0; padding: 0; }}
            body {{
                background-color: #0b0f19;
                color: #f1f5f9;
                font-family: 'Inter', system-ui, -apple-system, sans-serif;
                display: flex;
                flex-direction: column;
                align-items: center;
                min-height: 100vh;
                padding: 24px;
            }}
            .header {{
                text-align: center;
                margin-bottom: 20px;
            }}
            h1 {{
                font-size: 2rem;
                font-weight: 700;
                color: #38bdf8;
                letter-spacing: -0.5px;
            }}
            .subtitle {{
                color: #94a3b8;
                font-size: 0.95rem;
                margin-top: 4px;
            }}
            .main-layout {{
                display: flex;
                gap: 20px;
                max-width: 1280px;
                width: 100%;
                flex-wrap: wrap;
            }}
            .video-card {{
                flex: 2;
                min-width: 600px;
                position: relative;
                background: #1e293b;
                border-radius: 16px;
                box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
                overflow: hidden;
                border: 1px solid #334155;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 480px;
            }}
            img#streamFrame {{
                width: 100%;
                height: auto;
                display: block;
                border-radius: 16px;
            }}
            .overlay-badge {{
                position: absolute;
                top: 16px;
                left: 16px;
                background: rgba(15, 23, 42, 0.85);
                backdrop-filter: blur(8px);
                padding: 8px 16px;
                border-radius: 30px;
                font-size: 0.85rem;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 10px;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }}
            .dot {{
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background-color: #ef4444;
            }}
            .dot.connected {{
                background-color: #10b981;
                box-shadow: 0 0 10px #10b981;
            }}
            .metrics-panel {{
                flex: 1;
                min-width: 320px;
                background: #1e293b;
                border-radius: 16px;
                padding: 20px;
                border: 1px solid #334155;
                display: flex;
                flex-direction: column;
                gap: 16px;
            }}
            .stat-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 12px;
            }}
            .stat-box {{
                background: #0f172a;
                padding: 14px;
                border-radius: 12px;
                border: 1px solid #1e293b;
            }}
            .stat-title {{
                font-size: 0.75rem;
                color: #64748b;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .stat-value {{
                font-size: 1.5rem;
                font-weight: 700;
                color: #f8fafc;
                margin-top: 4px;
            }}
            .tracks-container {{
                flex: 1;
                overflow-y: auto;
                max-height: 360px;
            }}
            .track-item {{
                background: #0f172a;
                padding: 10px 14px;
                border-radius: 8px;
                margin-bottom: 8px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-size: 0.85rem;
                border: 1px solid #1e293b;
            }}
            .track-id {{ font-weight: 600; color: #38bdf8; }}
            .track-status {{ color: #10b981; font-weight: 500; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎥 Real-Time CCTV Head Detection & Face Recognition</h1>
            <div class="subtitle">Stream Camera ID: <strong>{camera_id}</strong></div>
        </div>

        <div class="main-layout">
            <div class="video-card">
                <div class="overlay-badge">
                    <div class="dot" id="statusDot"></div>
                    <span id="statusText">CONNECTING</span>
                </div>
                <img id="streamFrame" src="" alt="Live RTSP Feed" />
            </div>

            <div class="metrics-panel">
                <h3 style="font-size: 1.1rem; color: #f8fafc;">Pipeline Telemetry</h3>
                <div class="stat-grid">
                    <div class="stat-box">
                        <div class="stat-title">Processing FPS</div>
                        <div class="stat-value" id="fpsVal">0.0</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-title">Active Heads</div>
                        <div class="stat-value" id="headsVal">0</div>
                    </div>
                </div>

                <h4 style="font-size: 0.9rem; color: #94a3b8; margin-top: 8px;">Active Head Identities</h4>
                <div class="tracks-container" id="tracksList">
                    <div style="color: #64748b; font-size: 0.85rem;">Waiting for tracking & identity data...</div>
                </div>
            </div>
        </div>

        <script>
            const camera_id = "{camera_id}";
            const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
            const wsUrl = `${{wsProtocol}}//${{window.location.host}}/face/ws?camera_id=${{camera_id}}`;
            
            const imgEl = document.getElementById("streamFrame");
            const statusDot = document.getElementById("statusDot");
            const statusText = document.getElementById("statusText");
            const fpsVal = document.getElementById("fpsVal");
            const headsVal = document.getElementById("headsVal");
            const tracksList = document.getElementById("tracksList");

            function connect() {{
                const ws = new WebSocket(wsUrl);
                
                ws.onopen = () => {{
                    statusDot.classList.add("connected");
                    statusText.textContent = "LIVE STREAM";
                }};

                ws.onmessage = (event) => {{
                    try {{
                        const data = JSON.parse(event.data);
                        if (data.type === "frame") {{
                            if (data.frame) {{
                                imgEl.src = "data:image/jpeg;base64," + data.frame;
                            }}
                            if (data.fps !== undefined) {{
                                fpsVal.textContent = data.fps.toFixed(1);
                            }}
                            if (data.tracks) {{
                                headsVal.textContent = data.tracks.length;
                                if (data.tracks.length === 0) {{
                                    tracksList.innerHTML = '<div style="color: #64748b; font-size: 0.85rem;">No heads detected in frame</div>';
                                }} else {{
                                    tracksList.innerHTML = data.tracks.map(t => `
                                        <div class="track-item">
                                            <span class="track-id">${{t.label || ('Head #' + t.track_id)}}</span>
                                            <span class="track-status">${{t.status === 'resolved' ? 'LOCKED' : 'PENDING'}}</span>
                                        </div>
                                    `).join('');
                                }}
                            }}
                        }}
                    }} catch (e) {{
                        console.error("Error parsing socket payload", e);
                    }}
                }};

                ws.onclose = () => {{
                    statusDot.classList.remove("connected");
                    statusText.textContent = "DISCONNECTED - RETRYING";
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
        "faiss_vectors": len(faiss_manager.faiss_ids) if faiss_manager.index else 0,
        "queue_size": recognition_job_queue.qsize() if recognition_job_queue else 0,
        "mongo_connected": mongo_db.db is not None
    }


@router.post("/streams/add")
async def add_stream(camera_id: str = Query(...), rtsp_url: str = Query(...)):
    if camera_id in active_streams:
        raise HTTPException(status_code=400, detail=f"Stream '{camera_id}' is already active.")

    loop = asyncio.get_event_loop()
    worker = StreamWorker(camera_id, rtsp_url, job_queue=recognition_job_queue, loop=loop)
    worker.start()
    active_streams[camera_id] = worker
    logger.info("Added and started stream worker for camera: %s", camera_id)

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
    logger.info("Stopped stream worker for camera: %s", camera_id)

    return {
        "status": "success",
        "camera_id": camera_id,
        "message": "Stream worker stopped"
    }


# ======================
# WEBSOCKET STREAMING
# ======================

@router.websocket("/face/ws")
async def face_websocket(websocket: WebSocket, camera_id: str = Query(...), rtsp_url: Optional[str] = Query(None)):
    await websocket.accept()

    if camera_id not in active_streams:
        if not rtsp_url:
            await websocket.send_json({"type": "error", "message": f"Camera '{camera_id}' not active and no rtsp_url provided."})
            await websocket.close()
            return

        loop = asyncio.get_event_loop()
        worker = StreamWorker(camera_id, rtsp_url, job_queue=recognition_job_queue, loop=loop)
        worker.start()
        active_streams[camera_id] = worker
        logger.info("Auto-registered RTSP stream via WebSocket: %s", camera_id)
    else:
        worker = active_streams[camera_id]

    try:
        last_timestamp = 0.0
        while True:
            payload = worker.get_latest_payload()
            if payload and payload.get("timestamp", 0) != last_timestamp:
                last_timestamp = float(payload.get("timestamp", 0))
                await websocket.send_json(payload)

            await asyncio.sleep(0.033)  # ~30 FPS polling loop
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected for camera: %s", camera_id)
    except Exception as e:
        logger.error("WebSocket streaming error on camera %s: %s", camera_id, e)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT, workers=1)
