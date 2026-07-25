from __future__ import annotations

import os
import logging
import threading
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
import numpy as np
import cv2
import onnxruntime as ort

from config import settings, HEAD_DETECTOR, get_available_providers

logger = logging.getLogger("head_detector")

_lock = threading.Lock()
_global_detector_instance: Optional[HeadDetector] = None


@dataclass
class Detection:
    """Detection result with bounding box in original frame coordinates."""
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    score: float
    class_id: int = 1


class HeadDetector:
    """
    ONNX Runtime wrapper around YOLOv11 / YOLOv8 head detection models.
    Optimized for single-box head-only detection:
      - Filters specifically for HEAD class (class_id = 1) to eliminate body boxes
      - Aspect-ratio geometry guards to reject full-body candidate boxes
      - Automatically chooses the fastest execution provider available
      - Letterbox aspect-ratio preserving preprocessing
      - Vectorized NumPy postprocessing and OpenCV DNN NMS
    """

    def __init__(self, model_path: Optional[str] = None, cfg: Optional[dict[str, Union[int, float, str, List[str], Tuple[int, int]]]] = None):
        self.cfg = cfg or HEAD_DETECTOR
        active_model_path = model_path or str(self.cfg.get("model_path", settings.HEAD_MODEL_PATH))

        if not os.path.exists(active_model_path):
            if os.path.exists("yolov8n.onnx"):
                active_model_path = "yolov8n.onnx"
            else:
                pt_path = "yolov8n.pt"
                if not os.path.exists(pt_path):
                    import urllib.request
                    pt_url = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt"
                    logger.info("Downloading default YOLO model for head detection...")
                    urllib.request.urlretrieve(pt_url, pt_path)
                try:
                    from ultralytics import YOLO
                    pt_model = YOLO(pt_path)
                    pt_model.export(format="onnx", imgsz=[640, 640], dynamic=False)
                    active_model_path = "yolov8n.onnx"
                except Exception as e:
                    logger.warning(f"Could not export YOLO model to ONNX: {e}")

        providers = get_available_providers()
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        intra_threads = self.cfg.get("intra_op_threads")
        inter_threads = self.cfg.get("inter_op_threads")
        if isinstance(intra_threads, int):
            sess_options.intra_op_num_threads = intra_threads
        if isinstance(inter_threads, int):
            sess_options.inter_op_num_threads = inter_threads

        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(active_model_path, sess_options=sess_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        input_size_cfg = self.cfg.get("input_size", (640, 640))
        if isinstance(input_size_cfg, tuple) and len(input_size_cfg) == 2:
            self.in_w, self.in_h = int(input_size_cfg[0]), int(input_size_cfg[1])
        else:
            self.in_w, self.in_h = 640, 640

        conf_cfg = self.cfg.get("conf_threshold", 0.35)
        self.conf_thresh = float(conf_cfg) if isinstance(conf_cfg, (int, float)) else 0.35

        nms_cfg = self.cfg.get("nms_threshold", 0.45)
        self.nms_thresh = float(nms_cfg) if isinstance(nms_cfg, (int, float)) else 0.45

        logger.info(
            "HeadDetector loaded | model=%s | providers=%s | input=%dx%d | active_provider=%s",
            active_model_path, providers, self.in_w, self.in_h, self.session.get_providers(),
        )

    # ------------------------------------------------------------------ #
    # Preprocessing
    # ------------------------------------------------------------------ #
    def _letterbox(self, frame: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        h, w = frame.shape[:2]
        scale = min(self.in_w / w, self.in_h / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))

        if (new_w, new_h) != (w, h):
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = frame

        canvas = np.full((self.in_h, self.in_w, 3), 114, dtype=np.uint8)
        pad_x = (self.in_w - new_w) // 2
        pad_y = (self.in_h - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        return canvas, scale, pad_x, pad_y

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        canvas, scale, pad_x, pad_y = self._letterbox(frame)
        img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) * (1.0 / 255.0)
        img = img.transpose(2, 0, 1)[None, ...]  # NCHW layout
        return np.ascontiguousarray(img), scale, pad_x, pad_y

    # ------------------------------------------------------------------ #
    # Postprocessing
    # ------------------------------------------------------------------ #
    def _postprocess(
        self,
        raw_outputs: List[np.ndarray],
        scale: float,
        pad_x: int,
        pad_y: int,
        orig_w: int,
        orig_h: int
    ) -> List[Detection]:
        if not raw_outputs:
            return []

        head_class_id = int(self.cfg.get("head_class_id", 1))

        # Handle 3-tensor output format: [boxes (1, N, 4), scores (1, N, C), classes (1, N, 1)]
        if len(raw_outputs) >= 3:
            boxes_raw = np.asarray(raw_outputs[0])[0]
            scores_raw = np.asarray(raw_outputs[1]).flatten()
            classes_raw = np.asarray(raw_outputs[2]).flatten().astype(int)

            # Filter specifically for head_class_id (1 = Head)
            if np.max(classes_raw) >= head_class_id:
                mask = (scores_raw >= self.conf_thresh) & (classes_raw == head_class_id)
            else:
                mask = (scores_raw >= self.conf_thresh)

            if not np.any(mask):
                return []

            boxes_raw = boxes_raw[mask]
            scores = scores_raw[mask]
            class_ids = classes_raw[mask]

            cx, cy, bw, bh = boxes_raw[:, 0], boxes_raw[:, 1], boxes_raw[:, 2], boxes_raw[:, 3]
            x1, y1 = cx - bw / 2.0, cy - bh / 2.0
            x2, y2 = cx + bw / 2.0, cy + bh / 2.0
        else:
            pred = np.asarray(raw_outputs[0])[0]
            if pred.shape[0] < pred.shape[1]:
                pred = pred.T  # (N, 4 + C)

            num_cols = pred.shape[1]
            if num_cols < 5:
                return []

            boxes_cxcywh = pred[:, :4]
            cls_scores = pred[:, 4:]

            if cls_scores.shape[1] > 1 and head_class_id < cls_scores.shape[1]:
                scores = cls_scores[:, head_class_id]
                class_ids = np.full_like(scores, head_class_id, dtype=int)
            elif cls_scores.shape[1] == 1:
                scores = cls_scores[:, 0]
                class_ids = np.zeros_like(scores, dtype=int)
            else:
                class_ids = np.argmax(cls_scores, axis=1)
                scores = cls_scores[np.arange(len(cls_scores)), class_ids]

            keep = scores >= self.conf_thresh
            if cls_scores.shape[1] > 1 and head_class_id < cls_scores.shape[1]:
                keep = keep & (class_ids == head_class_id)

            if not np.any(keep):
                return []

            boxes_cxcywh = boxes_cxcywh[keep]
            scores = scores[keep]
            class_ids = class_ids[keep]

            cx, cy, bw, bh = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
            x1, y1 = cx - bw / 2.0, cy - bh / 2.0
            x2, y2 = cx + bw / 2.0, cy + bh / 2.0

        # Map back from letterbox to original image dimensions
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        x1 = np.clip(x1, 0, orig_w - 1)
        y1 = np.clip(y1, 0, orig_h - 1)
        x2 = np.clip(x2, 0, orig_w - 1)
        y2 = np.clip(y2, 0, orig_h - 1)

        # Aspect ratio & head geometry guard: filter out full-body boxes (height > 1.8 * width)
        box_w = x2 - x1
        box_h = y2 - y1
        valid_head_shape = (box_h <= 1.8 * box_w) & (box_h >= 0.4 * box_w)

        x1 = x1[valid_head_shape]
        y1 = y1[valid_head_shape]
        x2 = x2[valid_head_shape]
        y2 = y2[valid_head_shape]
        scores = scores[valid_head_shape]
        class_ids = class_ids[valid_head_shape]

        if len(scores) == 0:
            return []

        nms_boxes = np.stack([x1, y1, (x2 - x1), (y2 - y1)], axis=1).tolist()
        idxs = cv2.dnn.NMSBoxes(nms_boxes, scores.tolist(), self.conf_thresh, self.nms_thresh)
        if len(idxs) == 0:
            return []

        flat_idxs = idxs.flatten() if hasattr(idxs, 'flatten') else idxs

        return [
            Detection(
                bbox=(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])),
                score=float(scores[i]),
                class_id=int(class_ids[i]),
            )
            for i in flat_idxs
        ]

    # ------------------------------------------------------------------ #
    def detect(self, frame: np.ndarray) -> List[Detection]:
        if frame is None or frame.size == 0:
            return []
        h, w = frame.shape[:2]
        inp, scale, pad_x, pad_y = self._preprocess(frame)
        raw_outputs = self.session.run(self.output_names, {self.input_name: inp})
        return self._postprocess(raw_outputs, scale, pad_x, pad_y, w, h)


def get_global_head_detector() -> HeadDetector:
    global _global_detector_instance
    if _global_detector_instance is None:
        with _lock:
            if _global_detector_instance is None:
                _global_detector_instance = HeadDetector()
    return _global_detector_instance


def get_head_session() -> Tuple[ort.InferenceSession, str, Tuple[int, int]]:
    detector = get_global_head_detector()
    return detector.session, detector.input_name, (detector.in_h, detector.in_w)


def detect_heads(frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
    """Backward compatible helper function returning list of ((x1, y1, x2, y2), score)."""
    detector = get_global_head_detector()
    detections = detector.detect(frame)
    results: List[Tuple[Tuple[int, int, int, int], float]] = []
    for d in detections:
        x1, y1, x2, y2 = map(int, d.bbox)
        results.append(((x1, y1, x2, y2), d.score))
    return results
