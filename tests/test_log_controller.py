import unittest
import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, AsyncMock

from controllers.log_controller import LogController


class TestLogController(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.controller = LogController()

    @patch("controllers.log_controller.mongo_db")
    @patch("controllers.log_controller.faiss_manager")
    async def test_get_filtered_logs_mapping(self, mock_faiss, mock_mongo):
        now = datetime.now(timezone.utc)
        mock_event = {
            "_id": "66a4bc20d3f2a100",
            "camera_id": "cam_1",
            "track_id": 4,
            "profile_id": "shravan",
            "confidence": 0.885,
            "event_type": "KNOWN_IDENTITY",
            "timestamp": now,
            "bbox": [100, 80, 220, 200],
            "crop_path": "./crops/2026-07-27/shravan/face_4_1785125900.jpg",
            "full_frame_path": "./crops/2026-07-27/shravan/frame_4_1785125900.jpg"
        }

        mock_mongo.get_recent_events = AsyncMock(return_value=[mock_event])
        mock_faiss.get_name.return_value = "Shravan"

        logs = await self.controller.get_filtered_logs(profile_id="shravan", limit=10)

        self.assertEqual(len(logs), 1)
        log = logs[0]
        self.assertEqual(log["profile_id"], "shravan")
        self.assertEqual(log["name"], "Shravan")
        self.assertEqual(log["confidence_pct"], "88.5%")
        self.assertTrue(log["timestamp"].endswith("IST"))
        self.assertEqual(log["crop_url"], "/crops/2026-07-27/shravan/face_4_1785125900.jpg")
        self.assertEqual(log["full_frame_url"], "/crops/2026-07-27/shravan/frame_4_1785125900.jpg")
        self.assertEqual(log["bbox"], [100, 80, 220, 200])

    @patch("controllers.log_controller.mongo_db")
    async def test_get_filtered_logs_empty(self, mock_mongo):
        mock_mongo.get_recent_events = AsyncMock(return_value=[])
        logs = await self.controller.get_filtered_logs(profile_id="nonexistent")
        self.assertEqual(len(logs), 0)


if __name__ == "__main__":
    unittest.main()
