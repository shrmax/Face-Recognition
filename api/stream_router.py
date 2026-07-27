import asyncio
import logging
from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, HTTPException

from controllers.stream_controller import stream_controller
from database.mongo import mongo_db

logger = logging.getLogger("stream_router")
router = APIRouter(tags=["Streams"])


class StreamAddRequest(BaseModel):
    camera_id: str
    rtsp_url: str
    name: Optional[str] = None


@router.get("/health")
async def health_check():
    """Returns service health status, active streams count (max 4 limit), and FAISS stats."""
    return stream_controller.get_health_status()


@router.get("/api/streams")
async def list_streams():
    """Lists all configured RTSP streams saved in MongoDB with their active status."""
    db_streams = await mongo_db.load_all_rtsp_streams()
    active_cids = set(stream_controller.active_streams.keys())

    result = []
    for doc in db_streams:
        cid = str(doc.get("camera_id", ""))
        result.append({
            "camera_id": cid,
            "rtsp_url": doc.get("rtsp_url", ""),
            "name": doc.get("name", f"Camera {cid}"),
            "is_active": cid in active_cids,
            "created_at": str(doc.get("created_at", ""))
        })

    return {
        "status": "success",
        "count": len(result),
        "max_streams": 4,
        "streams": result
    }


@router.post("/streams/add")
@router.post("/api/streams/add")
async def add_stream(
    req: Optional[StreamAddRequest] = None,
    camera_id: Optional[str] = Query(None),
    rtsp_url: Optional[str] = Query(None),
    name: Optional[str] = Query(None)
):
    """
    Dynamically registers and starts a new RTSP camera worker stream.
    Strictly capped at a maximum limit of 4 RTSP camera streams.
    """
    cid = req.camera_id if req and req.camera_id else camera_id
    url = req.rtsp_url if req and req.rtsp_url else rtsp_url
    display_name = req.name if req and req.name else name

    if not cid or not url:
        raise HTTPException(status_code=400, detail="Both 'camera_id' and 'rtsp_url' are required.")

    try:
        loop = asyncio.get_event_loop()
        await stream_controller.add_stream(cid, url, loop, name=display_name)
        return {
            "status": "success",
            "camera_id": cid,
            "message": f"Stream worker started for {cid} ({url})"
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start stream: {e}")


@router.delete("/streams/{camera_id}")
@router.delete("/api/streams/{camera_id}")
async def remove_stream(camera_id: str):
    """Stops worker and deletes RTSP stream configuration from MongoDB."""
    stopped = await stream_controller.remove_stream(camera_id)
    return {
        "status": "success",
        "camera_id": camera_id,
        "message": f"Stream '{camera_id}' stopped and removed from system."
    }


@router.websocket("/face/ws")
async def face_websocket(
    websocket: WebSocket,
    camera_id: str = Query(...),
    rtsp_url: Optional[str] = Query(None)
):
    """
    High-throughput WebSocket streaming endpoint.
    Pushes Base64 JPEG frame buffers & tracking telemetry JSON for active streams.
    """
    await websocket.accept()

    worker = stream_controller.get_stream(camera_id)
    if worker is None:
        if not rtsp_url:
            await websocket.send_json({"type": "error", "message": f"Camera '{camera_id}' is not active and no rtsp_url provided."})
            await websocket.close()
            return

        try:
            loop = asyncio.get_event_loop()
            worker = await stream_controller.add_stream(camera_id, rtsp_url, loop)
            logger.info(f"Auto-registered RTSP stream via WebSocket: {camera_id}")
        except ValueError as ve:
            await websocket.send_json({"type": "error", "message": str(ve)})
            await websocket.close()
            return

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
