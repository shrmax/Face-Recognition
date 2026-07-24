import unittest
import numpy as np
import cv2
import asyncio
from config import settings
from core.quality import quality_filter
from core.tracker import StreamTrackManager
from core.recognition import faiss_manager

class TestVisionPipeline(unittest.TestCase):

    def test_quality_filter(self):
        # Create sharp test image
        sharp_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(sharp_img, (20, 20), (80, 80), (255, 255, 255), -1)
        is_passed, blur_score, reason = quality_filter.evaluate_quality(sharp_img)
        self.assertTrue(is_passed, f"Sharp image should pass quality filter. Score: {blur_score}")

        # Create blurry image
        blurry_img = cv2.GaussianBlur(sharp_img, (25, 25), 0)
        is_passed_b, blur_score_b, reason_b = quality_filter.evaluate_quality(blurry_img)
        self.assertFalse(is_passed_b, f"Blurry image should fail quality filter. Score: {blur_score_b}")

    def test_faiss_multi_vector_search(self):
        faiss_manager.initialize()
        
        # Add dummy 512-dim vector for test profile
        vec1 = np.random.randn(512).astype(np.float32)
        vec1 /= np.linalg.norm(vec1)
        
        faiss_manager.add_vector(vec1, "TEST_PROFILE_01")
        
        # Exact match query
        sim, profile_id = faiss_manager.search(vec1)
        self.assertAlmostEqual(sim, 1.0, places=2)
        self.assertEqual(profile_id, "TEST_PROFILE_01")
        
        # Orthogonal query (no match)
        ortho_vec = np.random.randn(512).astype(np.float32)
        ortho_vec /= np.linalg.norm(ortho_vec)
        sim_ortho, prof_ortho = faiss_manager.search(ortho_vec)
        self.assertLess(sim_ortho, settings.HIGH_CONF_THRESH)

    def test_tracker_state_transitions(self):
        tracker = StreamTrackManager("TEST_CAM")
        
        # Simulate dummy face detections
        class DummyFace:
            def __init__(self, bbox, score):
                self.bbox = np.array(bbox)
                self.det_score = score
                self.embedding = np.random.randn(512).astype(np.float32)

        faces = [DummyFace([50, 50, 150, 150], 0.9)]
        tracked_detections, pending_jobs = tracker.update(faces, (480, 640, 3))
        
        self.assertEqual(len(pending_jobs), 1)
        track_id, bbox, embedding = pending_jobs[0]
        self.assertEqual(tracker.track_states[track_id], "PENDING")
        self.assertEqual(len(embedding), 512)

        # Set track identity to PROCESSED
        tracker.set_track_identity(track_id, "EMP_001", "EMP_001 (0.95)", is_new_visit=True)
        self.assertEqual(tracker.track_states[track_id], "PROCESSED")

        # Next frame with same track should yield 0 pending jobs (Frame Gating)
        tracked_detections2, pending_jobs2 = tracker.update(faces, (480, 640, 3))
        self.assertEqual(len(pending_jobs2), 0, "Frame Gating should suppress re-ID jobs for processed tracks")

if __name__ == "__main__":
    unittest.main()
