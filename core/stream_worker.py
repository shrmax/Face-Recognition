import os
import cv2
import time
import asyncio
import logging
import threading
import numpy as np
import supervision as sv
from typing import Optional, Dict, Any, Tuple
from config import settings
from core.tracker import StreamTrackManager
from core.quality import quality_filter
from insightface.app import FaceAnalysis

logger = logging.getLogger("stream_worker")

# Shared InsightFace detector instance for SCRFD detection
face_detector: Optional[FaceAnalysis] = None
detector_lock = threading.Lock()

def get_face_detector() -> FaceAnalysis:
    global face_detector
    if face_detector is None:
        with detector_lock:
            if face_detector is None:
                app = FaceAnalysis(
                    name='buffalo_m',
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                    allowed_modules=['detection', 'recognition']
                )
                app.prepare(
                    ctx_id=0,
                    det_size=(settings.DET_WIDTH, settings.DET_HEIGHT),
                    det_thresh=settings.DET_THRESH
                )
                face_detector = app
                logger.info(f"InsightFace detector initialized with det_size=({settings.DET_WIDTH},{settings.DET_HEIGHT})")
    return face_detector

class StreamWorker:
    def __init__(self, camera_id: str, rtsp_url: str, job_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.job_queue = job_queue
        self.loop = loop
        
        self.running = True
        self.cap: Optional[cv2.VideoCapture] = None
        self.cap_lock = threading.Lock()
        
        self.track_manager = StreamTrackManager(camera_id)
        self.detector = get_face_detector()
        
        self.last_raw_frame: Optional[np.ndarray] = None
        self.last_annotated_frame: Optional[np.ndarray] = None
        self.last_frame_timestamp = time.time()
        self.frame_lock = threading.Lock()
        
        self.reconnect_count = 0
        self._first_frame_received = False
        
        # Start capture thread & watchdog thread
        self.capture_thread = threading.Thread(
            target=self._capture_and_process_loop,
            daemon=True,
            name=f"Stream_{camera_id}"
        )
        self.watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            daemon=True,
            name=f"Watchdog_{camera_id}"
        )
        
        self.capture_thread.start()
        self.watchdog_thread.start()

    def _connect(self) -> bool:
        with self.cap_lock:
            try:
                if self.cap is not None:
                    self.cap.release()
                time.sleep(0.3)
                
                # Try TCP transport first, then fallback to UDP (VLC default)
                transport_modes = ['tcp', 'udp']
                for transport in transport_modes:
                    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
                        f'rtsp_transport;{transport}'
                        '|timeout;5000000'      # 5 second timeout in microseconds
                        '|stimeout;5000000'      # socket timeout in microseconds
                        '|max_delay;500000'      # max demux delay
                    )
                    self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                    
                    if self.cap.isOpened():
                        logger.info(f"[{self.camera_id}] Connected via {transport.upper()} transport")
                        break
                    
                    logger.debug(f"[{self.camera_id}] {transport.upper()} transport failed, trying next...")
                    self.cap.release()
                    self.cap = None
                
                if self.cap is None or not self.cap.isOpened():
                    logger.warning(f"[{self.camera_id}] Failed to open RTSP: {self.rtsp_url}")
                    return False
                
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ret, test_frame = self.cap.read()
                if not ret or test_frame is None:
                    self.cap.release()
                    self.cap = None
                    return False
                
                with self.frame_lock:
                    self.last_raw_frame = test_frame.copy()
                    self.last_annotated_frame = test_frame.copy()
                    self.last_frame_timestamp = time.time()
                    self._first_frame_received = True
                
                logger.info(f"[{self.camera_id}] Connected to RTSP stream.")
                self.reconnect_count = 0
                return True
            except Exception as e:
                logger.error(f"[{self.camera_id}] Connection error: {e}")
                if self.cap:
                    self.cap.release()
                self.cap = None
                return False

    def _watchdog_loop(self):
        """RTSP Watchdog thread monitoring socket stalls"""
        while self.running:
            time.sleep(2.0)
            now = time.time()
            if self._first_frame_received and (now - self.last_frame_timestamp) > settings.WATCHDOG_TIMEOUT_SECONDS:
                logger.warning(f"[{self.camera_id}] Stream stall detected (no frame for {now - self.last_frame_timestamp:.1f}s). Reconnecting...")
                with self.cap_lock:
                    if self.cap:
                        self.cap.release()
                    self.cap = None
                self.last_frame_timestamp = time.time()

    def _capture_and_process_loop(self):
        """Main loop: 30 FPS buffer decode + 2-Tier AI Sampling at SAMPLE_FPS"""
        sample_interval = 1.0 / settings.SAMPLE_FPS
        last_sample_time = 0.0
        
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                logger.info(f"[{self.camera_id}] Reconnecting... attempt {self.reconnect_count + 1}")
                if not self._connect():
                    self.reconnect_count += 1
                    time.sleep(min(2 ** min(self.reconnect_count, 6), 30))
                    continue

            with self.cap_lock:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                else:
                    ret, frame = False, None

            if not ret or frame is None:
                time.sleep(0.01)
                continue

            now = time.time()
            self.last_frame_timestamp = now

            with self.frame_lock:
                self.last_raw_frame = frame.copy()

            # 2-Tier Sampling: Execute SCRFD + ByteTrack at SAMPLE_FPS (6 FPS)
            if now - last_sample_time >= sample_interval:
                last_sample_time = now
                self._run_ai_pipeline(frame)

            time.sleep(0.005)

    def _run_ai_pipeline(self, frame: np.ndarray):
        """Runs SCRFD detection, ByteTrack tracking, and enqueues qualified face crops"""
        try:
            # 1. SCRFD Face Detection
            with detector_lock:
                faces = self.detector.get(frame, max_num=settings.MAX_FACES)

            # 2. ByteTrack Tracking Update
            tracked_detections, pending_jobs = self.track_manager.update(faces, frame.shape)

            # 3. Quality Filtering & Enqueueing Pending Recognition Jobs
            h_img, w_img = frame.shape[:2]
            for track_id, (x1, y1, x2, y2) in pending_jobs:
                crop = frame[y1:y2, x1:x2].copy()
                is_passed, blur_score, reason = quality_filter.evaluate_quality(crop)
                
                if is_passed:
                    # Enqueue crop job into thread-safe asyncio queue
                    job_data = {
                        "camera_id": self.camera_id,
                        "track_id": track_id,
                        "crop": crop,
                        "bbox": [x1, y1, x2, y2],
                        "stream_worker": self
                    }
                    asyncio.run_coroutine_threadsafe(self.job_queue.put(job_data), self.loop)
                else:
                    # Reset state so next frame of track can retry quality check
                    if track_id in self.track_manager.track_states:
                        del self.track_manager.track_states[track_id]

            # 4. Render Annotations Overlay
            annotated_frame = frame.copy()
            if tracked_detections.tracker_id is not None and len(tracked_detections.tracker_id) > 0:
                labels = []
                for tid in tracked_detections.tracker_id:
                    tid_int = int(tid)
                    ident = self.track_manager.track_identities.get(tid_int, {})
                    labels.append(ident.get("label", f"Track #{tid_int}"))

                annotated_frame = self.track_manager.box_annotator.annotate(
                    scene=annotated_frame,
                    detections=tracked_detections
                )
                annotated_frame = self.track_manager.label_annotator.annotate(
                    scene=annotated_frame,
                    detections=tracked_detections,
                    labels=labels
                )

            with self.frame_lock:
                self.last_annotated_frame = np.asarray(annotated_frame)

        except Exception as e:
            logger.error(f"[{self.camera_id}] Error in AI pipeline: {e}")

    def get_latest_frame_b64(self) -> Optional[str]:
        with self.frame_lock:
            frame = self.last_annotated_frame if self.last_annotated_frame is not None else self.last_raw_frame
            if frame is None:
                return None
            frame_copy = frame.copy()

        try:
            _, buffer = cv2.imencode('.jpg', frame_copy, [cv2.IMWRITE_JPEG_QUALITY, 50])
            import base64
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding frame to base64: {e}")
            return None

    def stop(self):
        logger.info(f"[{self.camera_id}] Stopping stream worker...")
        self.running = False
        with self.cap_lock:
            if self.cap:
                self.cap.release()
            self.cap = None
