import os
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime

from database.mongo import mongo_db
from core.recognition import faiss_manager

logger = logging.getLogger("log_controller")


class LogController:
    """
    Controller handling business logic for detection event log queries and Method A snapshot mapping.
    """

    async def get_filtered_logs(
        self,
        profile_id: Optional[str] = None,
        date_str: Optional[str] = None,
        camera_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Union[str, int, float, List[int]]]]:
        """
        Retrieves detection events from MongoDB filtered by profile_id, date_str (YYYY-MM-DD), and camera_id.
        Formats timestamps and converts crop file paths to Method A static web URLs.
        """
        events = await mongo_db.get_recent_events(
            limit=limit,
            camera_id=camera_id,
            profile_id=profile_id,
            date_str=date_str
        )

        formatted_logs: List[Dict[str, Union[str, int, float, List[int]]]] = []

        for doc in events:
            pid = str(doc.get("profile_id", "Unknown")).strip()
            name = faiss_manager.get_name(pid) if pid != "Unknown" else "Unknown Identity"
            if name == pid and pid != "Unknown":
                name = pid.capitalize()

            raw_ts = doc.get("timestamp")
            if isinstance(raw_ts, datetime):
                from datetime import timezone, timedelta
                ist_tz = timezone(timedelta(hours=5, minutes=30))
                if raw_ts.tzinfo is None:
                    utc_ts = raw_ts.replace(tzinfo=timezone.utc)
                else:
                    utc_ts = raw_ts.astimezone(timezone.utc)
                ist_ts = utc_ts.astimezone(ist_tz)
                formatted_ts = ist_ts.strftime("%d/%m/%Y, %I:%M:%S %p IST")
            else:
                formatted_ts = str(raw_ts or "")

            conf = float(doc.get("confidence", 0.0))
            conf_pct = f"{round(conf * 100, 1)}%"

            # Method A: Map local crop file path to static web URL (/crops/...)
            raw_crop_path = str(doc.get("crop_path", "")).strip()
            crop_url = ""
            if raw_crop_path:
                norm_path = raw_crop_path.replace("\\", "/")
                if norm_path.startswith("./crops/"):
                    crop_url = norm_path.replace("./crops/", "/crops/")
                elif norm_path.startswith("crops/"):
                    crop_url = norm_path.replace("crops/", "/crops/")
                elif norm_path.startswith("/crops/"):
                    crop_url = norm_path
                else:
                    crop_url = f"/crops/{norm_path}"

            bbox = doc.get("bbox", [])
            bbox_list = [int(x) for x in bbox] if isinstance(bbox, (list, tuple)) else []

            formatted_logs.append({
                "id": str(doc.get("_id", "")),
                "camera_id": str(doc.get("camera_id", "cam_1")),
                "track_id": int(doc.get("track_id", 0)),
                "profile_id": pid,
                "name": name,
                "confidence": conf,
                "confidence_pct": conf_pct,
                "timestamp": formatted_ts,
                "bbox": bbox_list,
                "crop_url": crop_url,
                "event_type": str(doc.get("event_type", "KNOWN_IDENTITY"))
            })

        return formatted_logs


log_controller = LogController()
