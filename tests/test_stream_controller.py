import unittest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from controllers.stream_controller import StreamController, MAX_STREAMS


class TestStreamController(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.controller = StreamController()

    @patch("controllers.stream_controller.StreamWorker")
    @patch("controllers.stream_controller.mongo_db")
    async def test_add_stream_max_limit_enforcement(self, mock_mongo, mock_worker_cls):
        mock_mongo.save_rtsp_stream = AsyncMock(return_value=True)
        mock_worker_inst = MagicMock()
        mock_worker_cls.return_value = mock_worker_inst

        loop = asyncio.get_running_loop()

        # Add 4 streams cleanly
        for i in range(1, 5):
            worker = await self.controller.add_stream(f"cam_{i}", f"rtsp://192.168.1.{i}/stream", loop)
            self.assertIsNotNone(worker)

        self.assertEqual(len(self.controller.active_streams), 4)

        # Attempt adding 5th stream -> Must raise ValueError
        with self.assertRaises(ValueError) as ctx:
            await self.controller.add_stream("cam_5", "rtsp://192.168.1.5/stream", loop)

        self.assertIn("Maximum limit of 4 RTSP camera streams reached", str(ctx.exception))

    @patch("controllers.stream_controller.StreamWorker")
    @patch("controllers.stream_controller.mongo_db")
    async def test_remove_stream_frees_slot(self, mock_mongo, mock_worker_cls):
        mock_mongo.save_rtsp_stream = AsyncMock(return_value=True)
        mock_mongo.delete_rtsp_stream = AsyncMock(return_value=True)
        mock_worker_inst = MagicMock()
        mock_worker_cls.return_value = mock_worker_inst

        loop = asyncio.get_running_loop()

        # Add 4 streams
        for i in range(1, 5):
            await self.controller.add_stream(f"cam_{i}", f"rtsp://192.168.1.{i}/stream", loop)

        # Remove cam_1
        removed = await self.controller.remove_stream("cam_1")
        self.assertTrue(removed)
        self.assertEqual(len(self.controller.active_streams), 3)

        # Now adding cam_5 should succeed
        worker = await self.controller.add_stream("cam_5", "rtsp://192.168.1.5/stream", loop)
        self.assertIsNotNone(worker)
        self.assertEqual(len(self.controller.active_streams), 4)


if __name__ == "__main__":
    unittest.main()
