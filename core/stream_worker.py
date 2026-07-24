import os
import cv2
import time
import base64
import asyncio
import logging
import threading
import numpy as np
import supervision as sv
from typing import Optional, Dict, Any, Tuple
from config import settings
from core.tracker import StreamTrackManager, TrackState
from core.quality import quality_filter
from core.head_detector import detect_heads
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
        
        self.track_manager = StreamTrackManager(camera_id, frame_rate=settings.TRACKING_FPS)
        self.detector = get_face_detector()
        
        self.last_raw_frame: Optional[np.ndarray] = None
        self.active_tracked_detections = None
        self.active_labels = []
        self.latest_encoded_b64: Optional[str] = None
        self.last_frame_timestamp = time.time()
        self.frame_lock = threading.Lock()
        
        self.reconnect_count = 0
        self._first_frame_received = False
        
        # Start capture thread, tracking thread, AI recognition job dispatch thread, and watchdog thread
        self.capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name=f"Capture_{camera_id}"
        )
        self.tracking_thread = threading.Thread(
            target=self._tracking_loop,
            daemon=True,
            name=f"Tracking_{camera_id}"
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
        self.tracking_thread.start()
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

    def _tracking_loop(self):
        """High-frequency tracking loop (~TRACKING_FPS): runs ONNX head detection & ByteTrack on fresh raw frames"""
        interval = 1.0 / max(1, settings.TRACKING_FPS)
        while self.running:
            t0 = time.time()
            with self.frame_lock:
                frame_to_process = self.last_raw_frame.copy() if self.last_raw_frame is not None else None

            if frame_to_process is not None:
                try:
                    head_boxes = detect_heads(frame_to_process)
                    tracked_detections, _ = self.track_manager.update(head_boxes, frame_to_process.shape)

                    labels = []
                    if tracked_detections.tracker_id is not None and len(tracked_detections.tracker_id) > 0:
                        for tid in tracked_detections.tracker_id:
                            tid_int = int(tid)
                            ident = self.track_manager.track_identities.get(tid_int, {})
                            labels.append(ident.get("label", f"Track #{tid_int}"))

                    # Pre-render annotations & encode JPEG to Base64 in background tracking thread
                    frame_to_render = frame_to_process.copy()
                    if tracked_detections is not None and len(tracked_detections) > 0:
                        if tracked_detections.tracker_id is not None and len(tracked_detections.tracker_id) == len(tracked_detections):
                            labels_to_render = labels
                        else:
                            labels_to_render = ["Detecting..."] * len(tracked_detections)

                        frame_to_render = self.track_manager.box_annotator.annotate(
                            scene=frame_to_render,
                            detections=tracked_detections
                        )
                        frame_to_render = self.track_manager.label_annotator.annotate(
                            scene=frame_to_render,
                            detections=tracked_detections,
                            labels=labels_to_render
                        )
                        frame_to_render = np.asarray(frame_to_render)

                    if settings.SOCKET_MAX_WIDTH > 0 and settings.SOCKET_MAX_HEIGHT > 0:
                        h_f, w_f = frame_to_render.shape[:2]
                        if w_f > settings.SOCKET_MAX_WIDTH or h_f > settings.SOCKET_MAX_HEIGHT:
                            scale = min(settings.SOCKET_MAX_WIDTH / w_f, settings.SOCKET_MAX_HEIGHT / h_f)
                            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                            frame_to_render = cv2.resize(frame_to_render, (int(w_f * scale), int(h_f * scale)), interpolation=interp)

                    _, buffer = cv2.imencode('.jpg', frame_to_render, [cv2.IMWRITE_JPEG_QUALITY, 55])
                    b64_encoded = base64.b64encode(buffer).decode('utf-8')

                    with self.frame_lock:
                        self.active_tracked_detections = tracked_detections
                        self.active_labels = labels
                        self.latest_encoded_b64 = b64_encoded

                except Exception as e:
                    logger.error(f"[{self.camera_id}] Error in tracking loop: {e}")

            elapsed = time.time() - t0
            if elapsed > interval:
                logger.debug(f"[{self.camera_id}] Tracking loop iteration took {elapsed * 1000:.1f}ms (budget {interval * 1000:.1f}ms)")
            sleep_time = max(0.001, interval - elapsed)
            time.sleep(sleep_time)

    def _ai_processing_loop(self):
        """Recognition job dispatch loop (~SAMPLE_FPS): checks PENDING tracks and dispatches recognition jobs to async queue"""
        sample_interval = 1.0 / max(1, settings.SAMPLE_FPS)
        while self.running:
            time.sleep(sample_interval)
            
            with self.frame_lock:
                if self.last_raw_frame is None or self.active_tracked_detections is None:
                    continue
                frame = self.last_raw_frame.copy()
                detections = self.active_tracked_detections

            if detections.tracker_id is None or len(detections.tracker_id) == 0:
                continue

            h_img, w_img = frame.shape[:2]
            now = time.time()
            job_interval = 1.0 / max(1, settings.SAMPLE_FPS)

            for idx, tracker_id in enumerate(detections.tracker_id):
                t_id = int(tracker_id)
                if self.track_manager.track_states.get(t_id) == TrackState.PENDING:
                    last_job = self.track_manager.last_job_times.get(t_id, 0.0)
                    if (now - last_job) >= job_interval:
                        self.track_manager.last_job_times[t_id] = now
                        bbox = detections.xyxy[idx].astype(int)
                        x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
                        x2, y2 = min(w_img, int(bbox[2])), min(h_img, int(bbox[3]))
                        if (x2 - x1) < settings.MIN_FACE_SIZE or (y2 - y1) < settings.MIN_FACE_SIZE:
                            continue
                        
                        crop = frame[y1:y2, x1:x2].copy()
                        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                            continue

                        job_data = {
                            "camera_id": self.camera_id,
                            "track_id": t_id,
                            "crop": crop,
                            "bbox": [x1, y1, x2, y2],
                            "stream_worker": self
                        }
                        try:
                            asyncio.run_coroutine_threadsafe(self.job_queue.put(job_data), self.loop)
                        except Exception as err:
                            logger.error(f"[{self.camera_id}] Error enqueueing recognition job for Track #{t_id}: {err}")

    def get_latest_frame_b64(self) -> Optional[str]:
        with self.frame_lock:
            return self.latest_encoded_b64

    def stop(self):
        logger.info(f"[{self.camera_id}] Stopping stream worker...")
        self.running = False
        with self.cap_lock:
            if self.cap:
                self.cap.release()
            self.cap = None
