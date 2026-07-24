import os
import cv2
import threading
import logging
from typing import List, Tuple, Optional
import numpy as np
import onnxruntime as ort

logger = logging.getLogger("person_detector")

class YOLOONNXPersonDetector:
    def __init__(self, model_path: str = "yolov8n.onnx", input_size: Tuple[int, int] = (640, 640)):
        self.model_path = model_path
        self.input_size = input_size
        
        if not os.path.exists(self.model_path):
            logger.info(f"ONNX model '{self.model_path}' not found. Preparing ONNX model...")
            pt_path = "yolov8n.pt"
            if not os.path.exists(pt_path):
                import urllib.request
                pt_url = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt"
                logger.info(f"Downloading '{pt_path}' from release assets...")
                urllib.request.urlretrieve(pt_url, pt_path)
            from ultralytics import YOLO
            pt_model = YOLO(pt_path)
            pt_model.export(format="onnx", imgsz=list(input_size), dynamic=False)
            
        providers = ['CUDAExecutionProvider', 'CoreMLExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        logger.info(f"Loaded YOLO ONNX session with providers: {self.session.get_providers()}")

    def detect(self, frame: np.ndarray, conf_thresh: float = 0.4) -> List[Tuple[Tuple[int, int, int, int], float]]:
        if frame is None or frame.size == 0:
            return []

        orig_h, orig_w = frame.shape[:2]
        target_w, target_h = self.input_size

        # Preprocessing: BGR -> RGB -> Resize -> Normalize (0-1) -> CHW -> NCHW
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(rgb_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        input_tensor = resized_img.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)

        # Inference via ONNX Runtime session
        raw_outputs = self.session.run(None, {self.input_name: input_tensor})
        out_arr = np.asarray(raw_outputs[0])
        # YOLOv8 output shape: (1, 84, 8400) -> transpose to (8400, 84)
        predictions = out_arr[0].T

        boxes_list: List[List[int]] = []
        confidences: List[float] = []

        scale_x = orig_w / float(target_w)
        scale_y = orig_h / float(target_h)

        # COCO class index 0 = 'person'
        for pred in predictions:
            person_score = float(pred[4])
            if person_score < conf_thresh:
                continue

            cx, cy, w, h = float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3])
            x1 = int((cx - w / 2.0) * scale_x)
            y1 = int((cy - h / 2.0) * scale_y)
            box_w = int(w * scale_x)
            box_h = int(h * scale_y)

            boxes_list.append([x1, y1, box_w, box_h])
            confidences.append(person_score)

        if not boxes_list:
            return []

        # NMS (Non-Maximum Suppression)
        indices = cv2.dnn.NMSBoxes(boxes_list, confidences, conf_thresh, 0.45)
        
        results: List[Tuple[Tuple[int, int, int, int], float]] = []
        if len(indices) > 0:
            flat_indices = indices.flatten() if hasattr(indices, 'flatten') else indices
            for i in flat_indices:
                idx = int(i)
                bx1, by1, bw, bh = boxes_list[idx]
                bx2 = max(0, min(orig_w, bx1 + bw))
                by2 = max(0, min(orig_h, by1 + bh))
                bx1 = max(0, min(orig_w, bx1))
                by1 = max(0, min(orig_h, by1))
                results.append(((bx1, by1, bx2, by2), confidences[idx]))

        return results

_person_detector: Optional[YOLOONNXPersonDetector] = None
_detector_lock = threading.Lock()

def get_person_detector() -> YOLOONNXPersonDetector:
    global _person_detector
    if _person_detector is None:
        with _detector_lock:
            if _person_detector is None:
                _person_detector = YOLOONNXPersonDetector()
    return _person_detector

def detect_persons(frame: np.ndarray, conf_thresh: float = 0.4) -> List[Tuple[Tuple[int, int, int, int], float]]:
    detector = get_person_detector()
    return detector.detect(frame, conf_thresh=conf_thresh)
