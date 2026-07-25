import unittest
import numpy as np
import cv2
import asyncio
from config import settings
from core.tracker import StreamTrackManager
from core.recognition import faiss_manager, RecognitionWorker
from core.quality import quality_filter


class TestFaceRecognitionPipeline(unittest.TestCase):

    def test_faiss_vector_indexing_and_search(self):
        faiss_manager.initialize()

        # Generate dummy 512-dim normalized vector for test profile
        vec1 = np.random.randn(512).astype(np.float32)
        vec1 /= np.linalg.norm(vec1)

        faiss_manager.add_vector(vec1, "TEST_EMP_101", "Shravan")

        # Query exact vector -> similarity should be ~1.0
        sim, profile_id = faiss_manager.search(vec1)
        self.assertAlmostEqual(sim, 1.0, places=2)
        self.assertEqual(profile_id, "TEST_EMP_101")
        self.assertEqual(faiss_manager.get_name(profile_id), "Shravan")

    def test_identity_locking_and_job_suppression(self):
        track_manager = StreamTrackManager("CAM_TEST")
        head_boxes = [((100, 100, 200, 200), 0.88)]

        # Initial frame -> track state should be PENDING, 1 pending job
        sv_dets, pending_jobs = track_manager.update(head_boxes, (480, 640, 3))
        self.assertEqual(len(pending_jobs), 1)
        track_id, _ = pending_jobs[0]
        self.assertEqual(track_manager.track_states[track_id], "PENDING")

        # Lock track identity once face recognition succeeds
        track_manager.set_track_identity(track_id, "TEST_EMP_101", "Shravan (0.95)", is_new_visit=True)
        self.assertEqual(track_manager.track_states[track_id], "RESOLVED")

        # Subsequent frames -> pending_jobs should be 0 (no further recognition jobs dispatched!)
        sv_dets2, pending_jobs2 = track_manager.update(head_boxes, (480, 640, 3))
        self.assertEqual(len(pending_jobs2), 0, "RESOLVED track must suppress further recognition jobs")
        self.assertEqual(track_manager.track_identities[track_id]["label"], "Shravan (0.95)")

    def test_quality_filter_evaluation(self):
        sharp_crop = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(sharp_crop, (20, 20), (80, 80), (255, 255, 255), -1)
        is_passed, blur_score, reason = quality_filter.evaluate_quality(sharp_crop)
        self.assertTrue(is_passed, f"Sharp crop should pass quality check. Score: {blur_score}")

        blurry_crop = cv2.GaussianBlur(sharp_crop, (25, 25), 0)
        is_passed_b, blur_score_b, reason_b = quality_filter.evaluate_quality(blurry_crop)
        self.assertFalse(is_passed_b, f"Blurry crop should fail quality check. Score: {blur_score_b}")


if __name__ == "__main__":
    unittest.main()
