import unittest
import numpy as np
import cv2
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch, AsyncMock

from controllers.profile_controller import ProfileController


class TestProfileController(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.controller = ProfileController(uploads_dir=self.test_dir)

    async def asyncTearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    async def test_invalid_image_bytes(self):
        res = await self.controller.enroll_face_image("test_user", b"invalid_bytes", "test.jpg")
        self.assertFalse(res["success"])
        self.assertEqual(res["code"], "INVALID_IMAGE")

    @patch("controllers.profile_controller.get_face_detector")
    async def test_no_face_detected(self, mock_get_detector):
        mock_detector = MagicMock()
        mock_detector.get.return_value = []
        mock_get_detector.return_value = mock_detector

        # Synthetic image frame
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        _, img_bytes = cv2.imencode(".jpg", img)

        res = await self.controller.enroll_face_image("test_user", img_bytes.tobytes(), "test.jpg")
        self.assertFalse(res["success"])
        self.assertEqual(res["code"], "NO_FACE_DETECTED")
        self.assertIn("No face detected", res["message"])

    @patch("controllers.profile_controller.get_face_detector")
    async def test_multiple_faces_detected(self, mock_get_detector):
        mock_face1 = MagicMock()
        mock_face2 = MagicMock()
        mock_detector = MagicMock()
        mock_detector.get.return_value = [mock_face1, mock_face2]
        mock_get_detector.return_value = mock_detector

        img = np.zeros((300, 300, 3), dtype=np.uint8)
        _, img_bytes = cv2.imencode(".jpg", img)

        res = await self.controller.enroll_face_image("test_user", img_bytes.tobytes(), "test.jpg")
        self.assertFalse(res["success"])
        self.assertEqual(res["code"], "MULTIPLE_FACES_DETECTED")
        self.assertIn("Multiple faces detected", res["message"])

    @patch("controllers.profile_controller.mongo_db")
    @patch("controllers.profile_controller.faiss_manager")
    @patch("controllers.profile_controller.get_face_detector")
    async def test_single_face_enrolled_successfully(self, mock_get_detector, mock_faiss, mock_mongo):
        mock_face = MagicMock()
        mock_face.det_score = 0.95
        # Dummy 512D normalized vector
        vec = np.random.randn(512).astype(np.float32)
        vec /= np.linalg.norm(vec)
        mock_face.embedding = vec

        mock_detector = MagicMock()
        mock_detector.get.return_value = [mock_face]
        mock_get_detector.return_value = mock_detector

        mock_mongo.get_profile = AsyncMock(return_value={"embeddings": []})
        mock_mongo.save_or_update_profile = AsyncMock(return_value=True)

        img = np.zeros((300, 300, 3), dtype=np.uint8)
        _, img_bytes = cv2.imencode(".jpg", img)

        res = await self.controller.enroll_face_image("test_user", img_bytes.tobytes(), "test.jpg", name="Test User")
        self.assertTrue(res["success"])
        self.assertEqual(res["code"], "ENROLLED")
        self.assertIn("Face successfully enrolled", res["message"])

    @patch("controllers.profile_controller.mongo_db")
    @patch("controllers.profile_controller.faiss_manager")
    @patch("controllers.profile_controller.get_face_detector")
    async def test_duplicate_face_embedding_skipped(self, mock_get_detector, mock_faiss, mock_mongo):
        mock_face = MagicMock()
        mock_face.det_score = 0.95

        vec = np.random.randn(512).astype(np.float32)
        vec /= np.linalg.norm(vec)
        mock_face.embedding = vec

        mock_detector = MagicMock()
        mock_detector.get.return_value = [mock_face]
        mock_get_detector.return_value = mock_detector

        # Return exact same vector in existing profile embeddings -> similarity ~ 1.0 > 0.95
        mock_mongo.get_profile = AsyncMock(return_value={"embeddings": [vec.tolist()]})

        img = np.zeros((300, 300, 3), dtype=np.uint8)
        _, img_bytes = cv2.imencode(".jpg", img)

        res = await self.controller.enroll_face_image("test_user", img_bytes.tobytes(), "test.jpg", name="Test User")
        self.assertTrue(res["success"])
        self.assertEqual(res["code"], "DUPLICATE_SKIPPED")
        self.assertIn("vector enrollment was skipped", res["message"])


if __name__ == "__main__":
    unittest.main()
