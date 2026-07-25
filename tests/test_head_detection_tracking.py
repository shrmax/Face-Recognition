import unittest
import numpy as np
import cv2
import time
from config import HEAD_DETECTOR, TRACKER, PIPELINE, get_available_providers
from core.head_detector import HeadDetector, Detection, get_global_head_detector
from core.tracker import KalmanBoxTracker, HeadTracker, StreamTrackManager, iou_batch
from core.stream_worker import StreamWorker


class TestHeadDetectionAndTracking(unittest.TestCase):

    def test_config_and_providers(self):
        providers = get_available_providers()
        self.assertIsInstance(providers, list)
        self.assertGreater(len(providers), 0)

    def test_head_detector_initialization_and_inference(self):
        detector = get_global_head_detector()
        self.assertIsNotNone(detector.session)
        
        # Test synthetic image input
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        # Draw a synthetic circle to simulate head object
        cv2.circle(frame, (320, 240), 40, (255, 255, 255), -1)
        
        detections = detector.detect(frame)
        self.assertIsInstance(detections, list)

    def test_iou_batch(self):
        boxes_a = np.array([[0, 0, 100, 100], [200, 200, 300, 300]], dtype=np.float32)
        boxes_b = np.array([[0, 0, 100, 100], [50, 50, 150, 150]], dtype=np.float32)
        
        ious = iou_batch(boxes_a, boxes_b)
        self.assertEqual(ious.shape, (2, 2))
        self.assertAlmostEqual(float(ious[0, 0]), 1.0, places=3)
        self.assertGreater(float(ious[0, 1]), 0.0)

    def test_kalman_box_tracker(self):
        bbox = np.array([100, 100, 200, 200], dtype=np.float32)
        tracker = KalmanBoxTracker(bbox, score=0.85)
        
        state_pred = tracker.predict()
        self.assertEqual(state_pred.shape, (4,))
        
        # Update tracker
        new_bbox = np.array([105, 105, 205, 205], dtype=np.float32)
        tracker.update(new_bbox, score=0.90)
        self.assertEqual(tracker.hits, 2)
        self.assertEqual(tracker.state, "confirmed")

    def test_head_tracker_skipping_and_byte_track(self):
        tracker = HeadTracker()
        
        # Frame 1: Detection present
        dets1 = [Detection(bbox=(100.0, 100.0, 200.0, 200.0), score=0.85)]
        tracks1 = tracker.update(dets1)
        self.assertEqual(len(tracks1), 1)
        track_id = tracks1[0]["track_id"]
        
        # Frame 2: Skipped frame (None input) -> Motion prediction mode
        tracks2 = tracker.update(None)
        self.assertEqual(len(tracks2), 1)
        self.assertEqual(tracks2[0]["track_id"], track_id)
        
        # Frame 3: Detection slightly shifted
        dets3 = [Detection(bbox=(102.0, 102.0, 202.0, 202.0), score=0.88)]
        tracks3 = tracker.update(dets3)
        self.assertEqual(len(tracks3), 1)
        self.assertEqual(tracks3[0]["track_id"], track_id)

    def test_stream_track_manager_update(self):
        manager = StreamTrackManager("test_cam")
        head_boxes = [((100, 100, 200, 200), 0.85)]
        sv_dets, tracks = manager.update(head_boxes, (480, 640, 3))
        self.assertEqual(len(tracks), 1)
        self.assertEqual(len(sv_dets), 1)


if __name__ == "__main__":
    unittest.main()
