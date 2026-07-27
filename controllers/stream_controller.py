import asyncio
import logging
from typing import Dict, List, Optional, Union, Any

from core.stream_worker import StreamWorker
from database.mongo import mongo_db
from core.recognition import faiss_manager

logger = logging.getLogger("stream_controller")

MAX_STREAMS = 4


class StreamController:
    """
    Controller handling active RTSP stream workers, database persistence,
    and system health reporting (capped at a maximum of 4 RTSP streams).
    """

    def __init__(self):
        self.active_streams: Dict[str, StreamWorker] = {}
        self.recognition_job_queue: Optional[asyncio.Queue[Any]] = None

    def set_job_queue(self, queue: asyncio.Queue[Any]) -> None:
        self.recognition_job_queue = queue

    async def add_stream(
        self,
        camera_id: str,
        rtsp_url: str,
        loop: asyncio.AbstractEventLoop,
        name: Optional[str] = None,
        save_db: bool = True
    ) -> StreamWorker:
        cid = camera_id.strip()
        url = rtsp_url.strip()

        if len(self.active_streams) >= MAX_STREAMS:
            raise ValueError(f"Maximum limit of {MAX_STREAMS} RTSP camera streams reached. Please delete an existing stream before adding a new one.")

        if cid in self.active_streams:
            raise ValueError(f"Stream '{cid}' is already active.")

        if save_db:
            await mongo_db.save_rtsp_stream(cid, url, name)

        worker = StreamWorker(
            camera_id=cid,
            rtsp_url=url,
            job_queue=self.recognition_job_queue,
            loop=loop
        )
        worker.start()
        self.active_streams[cid] = worker
        logger.info(f"StreamController started worker for camera: {cid} ({url})")
        return worker

    async def remove_stream(self, camera_id: str) -> bool:
        cid = camera_id.strip()
        if cid not in self.active_streams:
            # Also try deleting from DB if present
            await mongo_db.delete_rtsp_stream(cid)
            return False

        worker = self.active_streams.pop(cid)
        worker.stop()
        await mongo_db.delete_rtsp_stream(cid)
        logger.info(f"StreamController stopped & removed worker for camera: {cid}")
        return True

    async def initialize_streams_from_db(self, loop: asyncio.AbstractEventLoop) -> List[str]:
        """
        Loads saved active RTSP streams from MongoDB on server startup
        and initializes stream workers up to the MAX_STREAMS limit (4).
        """
        db_streams = await mongo_db.load_all_rtsp_streams()
        started_cids: List[str] = []

        logger.info(f"Initializing saved RTSP streams from MongoDB (found {len(db_streams)})...")
        for stream_doc in db_streams:
            if len(self.active_streams) >= MAX_STREAMS:
                logger.warning(f"Reached max limit of {MAX_STREAMS} streams. Skipping remaining streams from DB.")
                break

            cid = str(stream_doc.get("camera_id", "")).strip()
            url = str(stream_doc.get("rtsp_url", "")).strip()
            name = str(stream_doc.get("name", f"Camera {cid}")).strip()

            if cid and url and cid not in self.active_streams:
                try:
                    await self.add_stream(cid, url, loop, name=name, save_db=False)
                    started_cids.append(cid)
                except Exception as e:
                    logger.error(f"Failed to auto-start stream {cid} from DB: {e}")

        return started_cids

    def get_stream(self, camera_id: str) -> Optional[StreamWorker]:
        return self.active_streams.get(camera_id.strip())

    def stop_all_streams(self) -> None:
        for cam_id, worker in list(self.active_streams.items()):
            worker.stop()
        self.active_streams.clear()

    def get_health_status(self) -> Dict[str, Union[str, int, List[str], bool]]:
        return {
            "status": "ok",
            "active_streams": len(self.active_streams),
            "max_streams": MAX_STREAMS,
            "stream_ids": list(self.active_streams.keys()),
            "faiss_vectors": len(faiss_manager.faiss_ids) if faiss_manager.index else 0,
            "queue_size": self.recognition_job_queue.qsize() if self.recognition_job_queue else 0,
            "mongo_connected": mongo_db.db is not None
        }


stream_controller = StreamController()
