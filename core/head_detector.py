import os
import cv2
import threading
import logging
from typing import List, Tuple, Optional
import numpy as np
import onnxruntime as ort
from config import settings

logger = logging.getLogger("head_detector")

_session: Optional[ort.InferenceSession] = None
_input_name: Optional[str] = None
_input_size: Optional[Tuple[int, int]] = None  # (h, w)
_lock = threading.Lock()

HEAD_CLASS_ID = 1  # 0 = person, 1 = head — per dual-class output

def get_head_session() -> Tuple[ort.InferenceSession, str, Tuple[int, int]]:
    global _session, _input_name, _input_size
    if _session is None or _input_name is None or _input_size is None:
        with _lock:
            if _session is None or _input_name is None or _input_size is None:
                os.makedirs(os.path.dirname(settings.HEAD_MODEL_PATH) or ".", exist_ok=True)
                active_model_path = settings.HEAD_MODEL_PATH
                if not os.path.exists(active_model_path):
                    if os.path.exists("yolov8n.onnx"):
                        active_model_path = "yolov8n.onnx"
                    else:
                        pt_path = "yolov8n.pt"
                        if not os.path.exists(pt_path):
                            import urllib.request
                            pt_url = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt"
                            logger.info("Downloading yolov8n.pt for model initialization...")
                            urllib.request.urlretrieve(pt_url, pt_path)
                        from ultralytics import YOLO
                        pt_model = YOLO(pt_path)
                        pt_model.export(format="onnx", imgsz=[640, 640], dynamic=False)
                        active_model_path = "yolov8n.onnx"

                providers = ['CUDAExecutionProvider', 'CoreMLExecutionProvider', 'CPUExecutionProvider']
                _session = ort.InferenceSession(active_model_path, providers=providers)
                input_meta = _session.get_inputs()[0]
                _input_name = input_meta.name
                
                shape = input_meta.shape
                h_val = shape[2] if len(shape) > 2 and isinstance(shape[2], int) else settings.HEAD_DET_SIZE
                w_val = shape[3] if len(shape) > 3 and isinstance(shape[3], int) else settings.HEAD_DET_SIZE
                _input_size = (int(h_val), int(w_val))
                logger.info(f"Loaded head detector ONNX session from '{active_model_path}' with shape={_input_size} and providers: {_session.get_providers()}")

    return _session, _input_name, _input_size


def _preprocess(image: np.ndarray, input_size: Tuple[int, int]) -> Tuple[np.ndarray, float, int, int]:
    h_in, w_in = input_size
    h_orig, w_orig = image.shape[:2]

    scale = min(float(w_in) / float(w_orig), float(h_in) / float(h_orig))
    new_w, new_h = int(w_orig * scale), int(h_orig * scale)
    resized = cv2.resize(image, (new_w, new_h))

    canvas = np.full((h_in, w_in, 3), 114, dtype=np.uint8)
    pad_top = (h_in - new_h) // 2
    pad_left = (w_in - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    # BGR, no channel swap — model_color_format=0
    img = canvas.astype(np.float32) * float(settings.HEAD_NET_SCALE_FACTOR)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img, scale, pad_top, pad_left


def _postprocess(
    raw_outputs: List[np.ndarray],
    scale: float,
    pad_top: int,
    pad_left: int,
    conf_thresh: float,
    iou_thresh: float
) -> List[Tuple[Tuple[int, int, int, int], float]]:
    if not raw_outputs:
        return []

    # Handle 3-tensor output format: [boxes (1, 8400, 4), scores (1, 8400, 1), classes (1, 8400, 1)]
    if len(raw_outputs) >= 3:
        boxes_tensor = np.asarray(raw_outputs[0])
        scores_tensor = np.asarray(raw_outputs[1])
        classes_tensor = np.asarray(raw_outputs[2])

        boxes_raw = boxes_tensor[0] if boxes_tensor.ndim == 3 else boxes_tensor
        scores = scores_tensor.flatten()
        classes = classes_tensor.flatten()

        target_class_id = float(HEAD_CLASS_ID)
        mask = (scores >= conf_thresh) & (classes == target_class_id)
        boxes_raw = boxes_raw[mask]
        scores = scores[mask]

        if len(scores) == 0:
            return []

        col_a, col_b, col_c, col_d = boxes_raw[:, 0], boxes_raw[:, 1], boxes_raw[:, 2], boxes_raw[:, 3]
        is_xyxy = len(col_a) > 0 and np.mean((col_c > col_a) & (col_d > col_b)) > 0.8
        logger.info(f"Raw box sample: {boxes_raw[0].tolist()}, is_xyxy={is_xyxy}")

        if is_xyxy:
            x1_m, y1_m, x2_m, y2_m = col_a, col_b, col_c, col_d
        else:
            x1_m = col_a - col_c / 2.0
            y1_m = col_b - col_d / 2.0
            x2_m = col_a + col_c / 2.0
            y2_m = col_b + col_d / 2.0

        x1 = (x1_m - float(pad_left)) / scale
        y1 = (y1_m - float(pad_top)) / scale
        x2 = (x2_m - float(pad_left)) / scale
        y2 = (y2_m - float(pad_top)) / scale

        # Aspect-ratio guard: filter out flat boxes (heads are taller than ~60% of their width)
        bw = x2 - x1
        bh = y2 - y1
        valid_mask = (bh >= 0.6 * bw)

        x1, y1, x2, y2, scores = x1[valid_mask], y1[valid_mask], x2[valid_mask], y2[valid_mask], scores[valid_mask]
        if len(scores) == 0:
            return []

        boxes_wh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(boxes_wh, scores.tolist(), conf_thresh, iou_thresh)

        results: List[Tuple[Tuple[int, int, int, int], float]] = []
        if len(indices) > 0:
            flat_indices = indices.flatten() if hasattr(indices, 'flatten') else indices
            for i in flat_indices:
                idx = int(i)
                bx1 = max(0, int(x1[idx]))
                by1 = max(0, int(y1[idx]))
                bx2 = max(bx1 + 1, int(x2[idx]))
                by2 = max(by1 + 1, int(y2[idx]))
                results.append(((bx1, by1, bx2, by2), float(scores[idx])))
        return results

    # Fallback for 1-tensor output format: (1, 6, 8400) or (1, 8400, 6)
    raw_arr = np.asarray(raw_outputs[0])
    if raw_arr.ndim == 3:
        if raw_arr.shape[1] < raw_arr.shape[2]:
            preds = raw_arr[0].T
        else:
            preds = raw_arr[0]
    else:
        preds = raw_arr

    num_cols = preds.shape[1] if preds.ndim > 1 else 0
    if num_cols <= 4:
        return []

    boxes_raw = preds[:, :4]
    class_scores = preds[:, 4:]

    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(class_scores)), class_ids]

    target_class_id = HEAD_CLASS_ID if class_scores.shape[1] > 1 else 0
    mask = (scores >= conf_thresh) & (class_ids == target_class_id)
    boxes_raw = boxes_raw[mask]
    scores = scores[mask]

    if len(scores) == 0:
        return []

    col_a, col_b, col_c, col_d = boxes_raw[:, 0], boxes_raw[:, 1], boxes_raw[:, 2], boxes_raw[:, 3]
    is_xyxy = len(col_a) > 0 and np.mean((col_c > col_a) & (col_d > col_b)) > 0.8

    if is_xyxy:
        x1_m, y1_m, x2_m, y2_m = col_a, col_b, col_c, col_d
    else:
        x1_m = col_a - col_c / 2.0
        y1_m = col_b - col_d / 2.0
        x2_m = col_a + col_c / 2.0
        y2_m = col_b + col_d / 2.0

    x1 = (x1_m - float(pad_left)) / scale
    y1 = (y1_m - float(pad_top)) / scale
    x2 = (x2_m - float(pad_left)) / scale
    y2 = (y2_m - float(pad_top)) / scale

    # Aspect-ratio guard: filter out flat boxes
    bw = x2 - x1
    bh = y2 - y1
    valid_mask = (bh >= 0.6 * bw)

    x1, y1, x2, y2, scores = x1[valid_mask], y1[valid_mask], x2[valid_mask], y2[valid_mask], scores[valid_mask]
    if len(scores) == 0:
        return []

    boxes_wh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    indices = cv2.dnn.NMSBoxes(boxes_wh, scores.tolist(), conf_thresh, iou_thresh)

    results: List[Tuple[Tuple[int, int, int, int], float]] = []
    if len(indices) > 0:
        flat_indices = indices.flatten() if hasattr(indices, 'flatten') else indices
        for i in flat_indices:
            idx = int(i)
            bx1 = max(0, int(x1[idx]))
            by1 = max(0, int(y1[idx]))
            bx2 = max(bx1 + 1, int(x2[idx]))
            by2 = max(by1 + 1, int(y2[idx]))
            results.append(((bx1, by1, bx2, by2), float(scores[idx])))
    return results


def detect_heads(frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
    if frame is None or frame.size == 0:
        return []
    session, input_name, input_size = get_head_session()
    blob, scale, pad_top, pad_left = _preprocess(frame, input_size)
    raw_outputs = session.run(None, {input_name: blob})
    output = [np.asarray(out) for out in raw_outputs]
    return _postprocess(output, scale, pad_top, pad_left, settings.HEAD_DET_CONF, settings.HEAD_DET_IOU)
