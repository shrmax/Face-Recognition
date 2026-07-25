import os
from typing import List, Tuple, TYPE_CHECKING
import onnxruntime as ort

if TYPE_CHECKING:
    from pydantic_settings import BaseSettings
else:
    try:
        from pydantic_settings import BaseSettings
    except ImportError:
        from pydantic import BaseSettings  # Fallback for standard pydantic v1/v2

# ---------------------------------------------------------------------------
# Paths & Default Model Locations
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
HEAD_DETECTOR_MODEL_PATH = os.path.join(MODELS_DIR, "yolov11_phd_s.onnx")

# ---------------------------------------------------------------------------
# Head Detector (YOLOv11 - person head detection, class 1: "head", class 0: "person")
# ---------------------------------------------------------------------------
HEAD_DETECTOR = {
    "model_path": HEAD_DETECTOR_MODEL_PATH,
    "input_size": (640, 640),        # (w, h) - MUST match ONNX export dimensions
    "conf_threshold": 0.35,          # Detection confidence threshold
    "nms_threshold": 0.45,           # IoU threshold for NMS
    "head_class_id": 1,              # 0 = person/body, 1 = HEAD ONLY
    "num_classes": 2,                # Person + Head
    # Preference order — fastest available ONNX runtime execution provider is chosen
    "providers": [
        "OpenVINOExecutionProvider",  # Intel CPU/iGPU hardware acceleration
        "CUDAExecutionProvider",      # NVIDIA GPU acceleration
        "CoreMLExecutionProvider",    # Apple Silicon / macOS hardware acceleration
        "CPUExecutionProvider",      # Universal fallback
    ],
    "intra_op_threads": max(1, (os.cpu_count() or 2) // 2),
    "inter_op_threads": 1,
}

# ---------------------------------------------------------------------------
# Tracker (ByteTrack-style: IoU + Kalman filter)
# ---------------------------------------------------------------------------
TRACKER = {
    "track_thresh": 0.5,     # Detections above this score are high-confidence
    "low_thresh": 0.1,       # Low confidence threshold for 2nd pass matching
    "match_thresh": 0.8,     # Cost ceiling (1 - IoU) for matching (0.8 = IoU >= 0.2)
    "track_buffer": 30,      # Frame buffer to keep lost tracks alive (~1s @ 30fps)
    "min_box_area": 100,     # Discard tiny boxes in px^2
    "frame_rate": 25,        # Nominal camera FPS
}

# ---------------------------------------------------------------------------
# RTSP / Stream Processing Pipeline Performance Tuning
# ---------------------------------------------------------------------------
PIPELINE = {
    "detect_every_n_frames": 3,     # Run YOLO detector every Nth frame; tracker predicts in between
    "reconnect_delay_sec": 2.0,     # RTSP reconnect delay
    "queue_size": 1,                # Drop stale frames, always keep latest frame
    "max_reconnect_attempts": 0,    # 0 = retry indefinitely
    "socket_fps": 30,               # Target WebSocket broadcast FPS
}


def get_available_providers() -> List[str]:
    """Intersection of configured provider preferences with ONNX Runtime available providers."""
    available = ort.get_available_providers()
    chosen = [p for p in HEAD_DETECTOR["providers"] if p in available]
    return chosen or ["CPUExecutionProvider"]


class Settings(BaseSettings):
    # App Config
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # MongoDB Config
    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "face_recognition_db")

    # Dataset & Storage Paths
    VIDEOS_FOLDER: str = os.getenv("VIDEOS_FOLDER", "./Employee")
    CROP_DIR: str = os.getenv("CROP_DIR", "./crops")
    FAISS_INDEX_PATH: str = "faiss_index.bin"
    KNOWN_IDS_PATH: str = "known_ids.pkl"
    EMBEDDINGS_PATH: str = "known_embeddings.pkl"

    # Head Detection Parameters
    HEAD_MODEL_PATH: str = HEAD_DETECTOR_MODEL_PATH
    HEAD_CLASS_ID: int = 1  # 1 = HEAD ONLY
    HEAD_NET_SCALE_FACTOR: float = 0.0039215697906911373
    HEAD_DET_SIZE: int = 640
    HEAD_DET_CONF: float = 0.35
    HEAD_DET_IOU: float = 0.45
    DET_WIDTH: int = 1280
    DET_HEIGHT: int = 1280
    DET_THRESH: float = 0.20
    MAX_FACES: int = 0  # 0 = unlimited

    # Sampling & Performance
    SAMPLE_FPS: int = 12  # AI Recognition Job Dispatch FPS
    TRACKING_FPS: int = 25 # Head Detection & ByteTrack Update FPS
    WATCHDOG_TIMEOUT_SECONDS: float = 5.0
    REVERIFY_INTERVAL_SECONDS: float = 45.0
    SOCKET_MAX_WIDTH: int = 1280
    SOCKET_MAX_HEIGHT: int = 720

    # Quality & Blur Filtering
    MIN_BLUR_VAR: float = 10.0
    MIN_FACE_SIZE: int = 12
    MIN_FACE_CROP_SIZE: int = 15

    # FAISS Dual Thresholds & Multi-Vector Gallery
    HIGH_CONF_THRESH: float = 0.42
    LOW_CONF_THRESH: float = 0.28
    MAX_GALLERY_EMBEDDINGS: int = 5

    # Cooldown & Retention Policy
    VISIT_COOLDOWN_MINS: int = 3
    RETENTION_DAYS: int = 30

    # RTSP Stream URLs (comma-separated camera_id=rtsp_url pairs)
    RTSP_STREAMS: str = ""

    # Pipeline & Models exposure
    HEAD_DETECTOR_CFG: dict[str, object] = HEAD_DETECTOR
    TRACKER_CFG: dict[str, object] = TRACKER
    PIPELINE_CFG: dict[str, object] = PIPELINE

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()
