import asyncio
import logging
from typing import Dict, List, Optional, Union, Any

from core.stream_worker import StreamWorker
from database.mongo import mongo_db
from core.recognition import faiss_manager

logger = logging.getLogger("stream_controller")


class StreamController:
    """
    Controller handling active RTSP stream workers, lifecycle management,
    and system health reporting.
    """

    def __init__(self):
        self.active_streams: Dict[str, StreamWorker] = {}
        self.recognition_job_queue: Optional[asyncio.Queue[Any]] = None

    def set_job_queue(self, queue: asyncio.Queue[Any]) -> None:
        self.recognition_job_queue = queue

    def add_stream(self, camera_id: str, rtsp_url: str, loop: asyncio.AbstractEventLoop) -> StreamWorker:
        if camera_id in self.active_streams:
            raise ValueError(f"Stream '{camera_id}' is already active.")

        worker = StreamWorker(
            camera_id=camera_id,
            rtsp_url=rtsp_url,
            job_queue=self.recognition_job_queue,
            loop=loop
        )
        worker.start()
        self.active_streams[camera_id] = worker
        logger.info(f"StreamController started worker for camera: {camera_id} ({rtsp_url})")
        return worker

    def remove_stream(self, camera_id: str) -> bool:
        if camera_id not in self.active_streams:
            return False

        worker = self.active_streams.pop(camera_id)
        worker.stop()
        logger.info(f"StreamController stopped worker for camera: {camera_id}")
        return True

    def get_stream(self, camera_id: str) -> Optional[StreamWorker]:
        return self.active_streams.get(camera_id)

    def stop_all_streams(self) -> None:
        for cam_id, worker in list(self.active_streams.items()):
            worker.stop()
        self.active_streams.clear()

    def get_health_status(self) -> Dict[str, Union[str, int, List[str], bool]]:
        return {
            "status": "ok",
            "active_streams": len(self.active_streams),
            "stream_ids": list(self.active_streams.keys()),
            "faiss_vectors": len(faiss_manager.faiss_ids) if faiss_manager.index else 0,
            "queue_size": self.recognition_job_queue.qsize() if self.recognition_job_queue else 0,
            "mongo_connected": mongo_db.db is not None
        }


stream_controller = StreamController()
