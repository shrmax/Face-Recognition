import asyncio
import logging
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, HTTPException
from controllers.stream_controller import stream_controller

logger = logging.getLogger("stream_router")
router = APIRouter(tags=["Streams"])


@router.get("/health")
async def health_check():
    """Returns service health status, active streams, and FAISS index stats."""
    return stream_controller.get_health_status()


@router.post("/streams/add")
async def add_stream(camera_id: str = Query(...), rtsp_url: str = Query(...)):
    """Dynamically registers and starts a new RTSP camera worker stream."""
    try:
        loop = asyncio.get_event_loop()
        stream_controller.add_stream(camera_id, rtsp_url, loop)
        return {
            "status": "success",
            "camera_id": camera_id,
            "message": f"Stream worker started for {rtsp_url}"
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start stream: {e}")


@router.delete("/streams/{camera_id}")
async def remove_stream(camera_id: str):
    """Stops and unregisters an active RTSP camera worker stream."""
    stopped = stream_controller.remove_stream(camera_id)
    if not stopped:
        raise HTTPException(status_code=404, detail=f"Stream '{camera_id}' not found.")
    return {
        "status": "success",
        "camera_id": camera_id,
        "message": "Stream worker stopped"
    }


@router.websocket("/face/ws")
async def face_websocket(
    websocket: WebSocket,
    camera_id: str = Query(...),
    rtsp_url: Optional[str] = Query(None)
):
    """
    High-throughput WebSocket streaming endpoint.
    Pushes Base64 JPEG frame buffers & tracking telemetry JSON.
    """
    await websocket.accept()

    worker = stream_controller.get_stream(camera_id)
    if worker is None:
        if not rtsp_url:
            await websocket.send_json({"type": "error", "message": f"Camera '{camera_id}' not active and no rtsp_url provided."})
            await websocket.close()
            return

        loop = asyncio.get_event_loop()
        worker = stream_controller.add_stream(camera_id, rtsp_url, loop)
        logger.info(f"Auto-registered RTSP stream via WebSocket: {camera_id}")

    try:
        last_timestamp = 0.0
        while True:
            payload = worker.get_latest_payload()
            if payload and payload.get("timestamp", 0) != last_timestamp:
                last_timestamp = float(payload.get("timestamp", 0))
                await websocket.send_json(payload)

            await asyncio.sleep(0.033)  # ~30 FPS polling loop
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected for camera: {camera_id}")
    except Exception as e:
        logger.error(f"WebSocket streaming error on camera {camera_id}: {e}")
