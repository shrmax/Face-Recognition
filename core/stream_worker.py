from __future__ import annotations

import os
import cv2
import time
import base64
import queue
import asyncio
import logging
import threading
from typing import Callable, Optional, Union, Tuple, List
import numpy as np

from config import PIPELINE, settings
from core.head_detector import HeadDetector, Detection, get_global_head_detector
from core.tracker import HeadTracker, StreamTrackManager

logger = logging.getLogger("stream_worker")


class _CaptureThread(threading.Thread):
    """
    Dedicated RTSP capture thread. Discards stale frames and retains only
    the newest frame in a 1-slot queue to eliminate buffering latency.
    """

    def __init__(self, rtsp_url: str, reconnect_delay: float, max_attempts: int):
        super().__init__(daemon=True)
        self.rtsp_url = rtsp_url
        self.reconnect_delay = reconnect_delay
        self.max_attempts = max_attempts
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self._stop_evt = threading.Event()
        self.connected = False

    def _open(self) -> cv2.VideoCapture:
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
            'rtsp_transport;tcp'
            '|fflags;nobuffer'
            '|flags;low_delay'
            '|framedrop;1'
            '|max_delay;0'
            '|timeout;5000000'
        )
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def run(self) -> None:
        attempts = 0
        cap = self._open()
        while not self._stop_evt.is_set():
            if not cap.isOpened():
                self.connected = False
                attempts += 1
                logger.warning("RTSP stream disconnected (%s), retry %d in %.1fs", self.rtsp_url, attempts, self.reconnect_delay)
                if self.max_attempts > 0 and attempts >= self.max_attempts:
                    logger.error("Max RTSP reconnect attempts reached for %s", self.rtsp_url)
                    break
                time.sleep(self.reconnect_delay)
                cap.release()
                cap = self._open()
                continue

            ok, frame = cap.read()
            if not ok or frame is None:
                self.connected = False
                cap.release()
                time.sleep(self.reconnect_delay)
                cap = self._open()
                continue

            self.connected = True
            attempts = 0

            if self._q.full():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
            self._q.put(frame)

        cap.release()

    def read(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._stop_evt.set()


class StreamWorker:
    """
    RTSP Stream Worker combining high-throughput ONNX Head Detection, ByteTrack Tracking,
    and asynchronous Face Recognition job dispatching.
    """

    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        job_queue: Optional[asyncio.Queue[dict[str, Union[str, int, np.ndarray, float, List[int], StreamWorker]]]] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        on_tracks: Optional[Callable[[str, np.ndarray, List[dict[str, Union[int, float, str, Tuple[float, float, float, float]]]]], None]] = None,
        detector: Optional[HeadDetector] = None,
        tracker: Optional[HeadTracker] = None,
        pipeline_cfg: Optional[dict[str, Union[int, float]]] = None,
    ):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.job_queue = job_queue
        self.loop = loop
        self.on_tracks = on_tracks
        self.cfg = pipeline_cfg or PIPELINE

        self.detector = detector or get_global_head_detector()
        self.tracker = tracker or HeadTracker()
        self.track_manager = StreamTrackManager(camera_id, frame_rate=settings.TRACKING_FPS)

        reconnect_delay = float(self.cfg.get("reconnect_delay_sec", 2.0))
        max_attempts = int(self.cfg.get("max_reconnect_attempts", 0))
        self._capture = _CaptureThread(rtsp_url, reconnect_delay, max_attempts)

        self._proc_thread: Optional[threading.Thread] = None
        self._ai_thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._frame_idx = 0
        self.current_fps: float = 0.0

        self.last_raw_frame: Optional[np.ndarray] = None
        self.active_tracks: List[dict[str, Union[int, float, str, Tuple[float, float, float, float]]]] = []
        self.frame_lock = threading.Lock()

        self.latest_payload: Optional[dict[str, Union[str, int, float, List[dict[str, Union[int, float, str, Tuple[float, float, float, float]]]]]]] = None
        self.latest_b64_frame: Optional[str] = None
        self.payload_lock = threading.Lock()

        self.last_job_times: dict[int, float] = {}

        self._socket_listeners: List[Callable[[dict[str, Union[str, int, float, List[dict[str, Union[int, float, str, Tuple[float, float, float, float]]]]]]], None]] = []

    def register_socket_listener(self, callback: Callable[[dict[str, Union[str, int, float, List[dict[str, Union[int, float, str, Tuple[float, float, float, float]]]]]]], None]) -> None:
        if callback not in self._socket_listeners:
            self._socket_listeners.append(callback)

    def unregister_socket_listener(self, callback: Callable[[dict[str, Union[str, int, float, List[dict[str, Union[int, float, str, Tuple[float, float, float, float]]]]]]], None]) -> None:
        if callback in self._socket_listeners:
            self._socket_listeners.remove(callback)

    def start(self) -> None:
        self._capture.start()

        self._proc_thread = threading.Thread(target=self._run, daemon=True, name=f"Tracking_{self.camera_id}")
        self._proc_thread.start()

        self._ai_thread = threading.Thread(target=self._ai_processing_loop, daemon=True, name=f"AI_{self.camera_id}")
        self._ai_thread.start()

        logger.info("StreamWorker[%s] started for %s", self.camera_id, self.rtsp_url)

    def _run(self) -> None:
        detect_every_val = self.cfg.get("detect_every_n_frames", 3)
        detect_every = max(1, int(detect_every_val) if isinstance(detect_every_val, (int, float)) else 3)

        last_time = time.time()
        fps_alpha = 0.1

        while not self._stop_evt.is_set():
            frame = self._capture.read(timeout=1.0)
            if frame is None:
                continue

            t0 = time.time()
            self._frame_idx += 1
            run_detector = (self._frame_idx % detect_every) == 0

            if run_detector:
                detections = self.detector.detect(frame)
                head_boxes = [(d.bbox, d.score) for d in detections]
                sv_dets, pending_jobs = self.track_manager.update(head_boxes, frame.shape)
            else:
                sv_dets, pending_jobs = self.track_manager.update(None, frame.shape)

            dt = time.time() - last_time
            last_time = time.time()
            if dt > 0:
                instant_fps = 1.0 / dt
                self.current_fps = (1.0 - fps_alpha) * self.current_fps + fps_alpha * instant_fps

            # Build enriched track dicts with resolved identity labels
            enriched_tracks: List[dict[str, Union[int, float, str, Tuple[float, float, float, float]]]] = []
            if sv_dets.tracker_id is not None and len(sv_dets.tracker_id) > 0:
                for idx, tid in enumerate(sv_dets.tracker_id):
                    tid_int = int(tid)
                    box = sv_dets.xyxy[idx]
                    conf = float(sv_dets.confidence[idx]) if sv_dets.confidence is not None else 0.0
                    ident = self.track_manager.track_identities.get(tid_int, {})
                    label_str = ident.get("label", f"Head #{tid_int}")
                    status_str = ident.get("status", "pending")

                    enriched_tracks.append({
                        "track_id": tid_int,
                        "bbox": (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                        "score": conf,
                        "label": label_str,
                        "status": status_str
                    })

            with self.frame_lock:
                self.last_raw_frame = frame.copy()
                self.active_tracks = enriched_tracks

            # Render bounding box overlay canvas
            rendered = frame.copy()
            for trk in enriched_tracks:
                x1, y1, x2, y2 = map(int, trk["bbox"])
                label_text = str(trk["label"])
                cv2.rectangle(rendered, (x1, y1), (x2, y2), (0, 255, 128), 2)
                cv2.putText(rendered, label_text, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 2)

            if settings.SOCKET_MAX_WIDTH > 0 and settings.SOCKET_MAX_HEIGHT > 0:
                h_f, w_f = rendered.shape[:2]
                if w_f > settings.SOCKET_MAX_WIDTH or h_f > settings.SOCKET_MAX_HEIGHT:
                    scale = min(settings.SOCKET_MAX_WIDTH / w_f, settings.SOCKET_MAX_HEIGHT / h_f)
                    rendered = cv2.resize(rendered, (int(w_f * scale), int(h_f * scale)), interpolation=cv2.INTER_AREA)

            _, jpeg_buf = cv2.imencode('.jpg', rendered, [cv2.IMWRITE_JPEG_QUALITY, 60])
            b64_frame = base64.b64encode(jpeg_buf).decode('utf-8')

            payload = {
                "type": "frame",
                "camera_id": self.camera_id,
                "timestamp": round(time.time(), 3),
                "frame_idx": self._frame_idx,
                "fps": round(self.current_fps, 1),
                "tracks": enriched_tracks,
                "frame": b64_frame,
            }

            with self.payload_lock:
                self.latest_payload = payload
                self.latest_b64_frame = b64_frame

            if self.on_tracks is not None:
                try:
                    self.on_tracks(self.camera_id, frame, enriched_tracks)
                except Exception as e:
                    logger.exception("on_tracks callback failed for camera %s: %s", self.camera_id, e)

            for listener in list(self._socket_listeners):
                try:
                    listener(payload)
                except Exception as e:
                    logger.error("Error pushing frame to socket listener: %s", e)

            elapsed = time.time() - t0
            target_period = 1.0 / max(1, settings.TRACKING_FPS)
            if elapsed < target_period:
                time.sleep(target_period - elapsed)

    def _ai_processing_loop(self) -> None:
        """
        Recognition job dispatch loop (~SAMPLE_FPS).
        Dispatches head crops ONLY for active tracks in 'PENDING' state.
        Once a track identity is RESOLVED, zero further recognition jobs are dispatched.
        """
        job_interval = 1.0 / max(1, settings.SAMPLE_FPS)
        while not self._stop_evt.is_set():
            time.sleep(job_interval)

            if self.job_queue is None or self.loop is None:
                continue

            with self.frame_lock:
                raw_frame = self.last_raw_frame.copy() if self.last_raw_frame is not None else None
                tracks_to_check = list(self.active_tracks)

            if raw_frame is None or len(tracks_to_check) == 0:
                continue

            h_img, w_img = raw_frame.shape[:2]
            now = time.time()

            for trk in tracks_to_check:
                t_id = int(trk["track_id"])
                state = self.track_manager.track_states.get(t_id)

                # Skip resolved tracks completely — lock identity indefinitely
                if state == "RESOLVED":
                    continue

                last_job = self.last_job_times.get(t_id, 0.0)
                if (now - last_job) >= job_interval:
                    self.last_job_times[t_id] = now
                    x1, y1, x2, y2 = trk["bbox"]
                    bx1, by1, bx2, by2 = int(x1), int(y1), int(x2), int(y2)
                    bw, bh = bx2 - bx1, by2 - by1

                    if bw < settings.MIN_FACE_SIZE or bh < settings.MIN_FACE_SIZE:
                        continue

                    # Add 35% margin padding around head box for facial context
                    pad_w = int(bw * 0.35)
                    pad_h = int(bh * 0.35)
                    cx1 = max(0, bx1 - pad_w)
                    cy1 = max(0, by1 - pad_h)
                    cx2 = min(w_img, bx2 + pad_w)
                    cy2 = min(h_img, by2 + pad_h)

                    crop = raw_frame[cy1:cy2, cx1:cx2].copy()
                    if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                        continue

                    job_data = {
                        "camera_id": self.camera_id,
                        "track_id": t_id,
                        "crop": crop,
                        "bbox": [cx1, cy1, cx2, cy2],
                        "stream_worker": self
                    }
                    try:
                        asyncio.run_coroutine_threadsafe(self.job_queue.put(job_data), self.loop)
                    except Exception as err:
                        logger.error("[%s] Error enqueueing recognition job for Track #%d: %s", self.camera_id, t_id, err)

    def get_latest_payload(self) -> Optional[dict[str, Union[str, int, float, List[dict[str, Union[int, float, str, Tuple[float, float, float, float]]]]]]]:
        with self.payload_lock:
            return self.latest_payload

    def get_latest_frame_b64(self) -> Optional[str]:
        with self.payload_lock:
            return self.latest_b64_frame

    def stop(self) -> None:
        self._stop_evt.set()
        self._capture.stop()
        if self._proc_thread and self._proc_thread.is_alive():
            self._proc_thread.join(timeout=2.0)
        if self._ai_thread and self._ai_thread.is_alive():
            self._ai_thread.join(timeout=2.0)
        logger.info("StreamWorker[%s] stopped", self.camera_id)
