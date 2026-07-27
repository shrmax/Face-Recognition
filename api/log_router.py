from typing import Optional
from fastapi import APIRouter, Query
from controllers.log_controller import log_controller

router = APIRouter(prefix="/api/logs", tags=["Logs"])


@router.get("")
@router.get("/")
async def get_detection_logs(
    profile_id: Optional[str] = Query(None, description="Filter logs by profile ID"),
    date: Optional[str] = Query(None, description="Filter logs by date (YYYY-MM-DD)"),
    camera_id: Optional[str] = Query(None, description="Filter logs by camera ID"),
    limit: int = Query(50, ge=1, le=500, description="Max logs to return")
):
    """
    Returns filtered real-time face detection activity logs from MongoDB.
    Resolves Method A pre-cropped face snapshot static web URLs.
    """
    logs = await log_controller.get_filtered_logs(
        profile_id=profile_id,
        date_str=date,
        camera_id=camera_id,
        limit=limit
    )
    return {
        "status": "success",
        "count": len(logs),
        "filters": {
            "profile_id": profile_id,
            "date": date,
            "camera_id": camera_id,
            "limit": limit
        },
        "logs": logs
    }
