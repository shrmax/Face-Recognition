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
from core.person_detector import detect_persons
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
                    providers=['CUDAExecutionProvider', 'CoreMLExecutionProvider', 'CPUExecutionProvider'],
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
        self.active_tracked_detections = None
        self.active_labels = []
        self.last_frame_timestamp = time.time()
        self.frame_lock = threading.Lock()
        
        self.reconnect_count = 0
        self._first_frame_received = False
        
        # Start capture thread, AI processing thread, and watchdog thread
        self.capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name=f"Capture_{camera_id}"
        )
        self.ai_thread = threading.Thread(
            target=self._ai_processing_loop,
            daemon=True,
            name=f"AI_{camera_id}"
        )
        self.watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            daemon=True,
            name=f"Watchdog_{camera_id}"
        )
        
        self.capture_thread.start()
        self.ai_thread.start()
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
                        '|fflags;nobuffer'       # Disable FFmpeg stream buffering
                        '|flags;low_delay'       # Enable low-latency decoding
                        '|framedrop;1'           # Drop stale packets
                        '|max_delay;0'           # Set max demux delay to 0
                        '|timeout;5000000'       # 5 sec timeout
                        '|stimeout;5000000'
                    )
                    self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                    
                    if self.cap.isOpened():
                        logger.info(f"[{self.camera_id}] Connected via {transport.upper()} transport (Low Latency Mode)")
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

    def _capture_loop(self):
        """Dedicated RTSP frame reading loop: drains FFmpeg buffer as fast as possible"""
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
                time.sleep(0.005)
                continue

            now = time.time()
            self.last_frame_timestamp = now

            # Store 100% full resolution raw frame for max AI detection & recognition accuracy
            with self.frame_lock:
                self.last_raw_frame = frame.copy()

            time.sleep(0.001)

    def _ai_processing_loop(self):
        """Dedicated AI loop: processes latest frame at SAMPLE_FPS without blocking RTSP capture"""
        sample_interval = 1.0 / settings.SAMPLE_FPS
        
        while self.running:
            time.sleep(sample_interval)
            
            with self.frame_lock:
                if self.last_raw_frame is None:
                    continue
                frame_to_process = self.last_raw_frame.copy()

            self._run_ai_pipeline(frame_to_process)

    def _run_ai_pipeline(self, frame: np.ndarray):
        """Runs YOLOv8 person detection, ByteTrack tracking, and enqueues person crops for face recognition"""
        try:
            person_boxes = detect_persons(frame)

            tracked_detections, pending_jobs, reverify_jobs = self.track_manager.update(person_boxes, frame.shape)

            for track_id, (x1, y1, x2, y2) in (pending_jobs + reverify_jobs):
                crop = frame[y1:y2, x1:x2].copy()
                job_data = {
                    "camera_id": self.camera_id,
                    "track_id": track_id,
                    "crop": crop,
                    "bbox": [x1, y1, x2, y2],
                    "stream_worker": self
                }
                asyncio.run_coroutine_threadsafe(self.job_queue.put(job_data), self.loop)

            labels = []
            if tracked_detections.tracker_id is not None and len(tracked_detections.tracker_id) > 0:
                for tid in tracked_detections.tracker_id:
                    tid_int = int(tid)
                    ident = self.track_manager.track_identities.get(tid_int, {})
                    labels.append(ident.get("label", f"Track #{tid_int}"))

            with self.frame_lock:
                self.active_tracked_detections = tracked_detections
                self.active_labels = labels

        except Exception as e:
            logger.error(f"[{self.camera_id}] Error in AI pipeline: {e}")

    def get_latest_frame_b64(self) -> Optional[str]:
        with self.frame_lock:
            if self.last_raw_frame is None:
                return None
            frame_copy = self.last_raw_frame.copy()
            active_detections = self.active_tracked_detections
            active_labels = self.active_labels

        try:
            # Annotate tracking bounding boxes onto the latest 30 FPS live frame
            if active_detections is not None and len(active_detections) > 0:
                if active_detections.tracker_id is not None and len(active_detections.tracker_id) == len(active_detections):
                    labels_to_render = active_labels
                else:
                    labels_to_render = ["Detecting..."] * len(active_detections)

                frame_copy = self.track_manager.box_annotator.annotate(
                    scene=frame_copy,
                    detections=active_detections
                )
                frame_copy = self.track_manager.label_annotator.annotate(
                    scene=frame_copy,
                    detections=active_detections,
                    labels=labels_to_render
                )
                frame_copy = np.asarray(frame_copy)

            # Downscale ONLY the WebSocket stream output frame to 480p (SOCKET_MAX_WIDTH/HEIGHT)
            if settings.SOCKET_MAX_WIDTH > 0 and settings.SOCKET_MAX_HEIGHT > 0:
                h_f, w_f = frame_copy.shape[:2]
                if w_f > settings.SOCKET_MAX_WIDTH or h_f > settings.SOCKET_MAX_HEIGHT:
                    scale = min(settings.SOCKET_MAX_WIDTH / w_f, settings.SOCKET_MAX_HEIGHT / h_f)
                    frame_copy = cv2.resize(frame_copy, (int(w_f * scale), int(h_f * scale)), interpolation=cv2.INTER_AREA)

            _, buffer = cv2.imencode('.jpg', frame_copy, [cv2.IMWRITE_JPEG_QUALITY, 55])
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
