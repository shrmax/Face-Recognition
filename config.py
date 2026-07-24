import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_settings import BaseSettings
else:
    try:
        from pydantic_settings import BaseSettings
    except ImportError:
        from pydantic import BaseSettings  # Fallback for standard pydantic v1/v2

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
    
    # SCRFD Face Detection Parameters (tuned for wide-angle overhead CCTV camera feeds)
    DET_WIDTH: int = 1280
    DET_HEIGHT: int = 1280
    DET_THRESH: float = 0.20
    MAX_FACES: int = 0  # 0 = unlimited face detection (100+ crowd per frame)
    
    # Sampling & Performance
    SAMPLE_FPS: int = 12  # AI Detection & Tracking FPS
    WATCHDOG_TIMEOUT_SECONDS: float = 5.0
    REVERIFY_INTERVAL_SECONDS: float = 45.0
    SOCKET_MAX_WIDTH: int = 1280  # 720p max width for WebSocket streaming output
    SOCKET_MAX_HEIGHT: int = 720  # 720p max height for WebSocket streaming output
    
    # Quality & Blur Filtering
    MIN_BLUR_VAR: float = 15.0   # Laplacian variance threshold
    MIN_FACE_SIZE: int = 12      # Min bounding box width/height in pixels for small distant faces
    
    # FAISS Dual Thresholds & Multi-Vector Gallery
    HIGH_CONF_THRESH: float = 0.42  # Match threshold for known profile (optimized for RTSP video)
    LOW_CONF_THRESH: float = 0.28   # Below this = Auto-enroll new identity
    MAX_GALLERY_EMBEDDINGS: int = 5 # Max representative vectors per profile in FAISS
    
    # Cooldown & Retention Policy
    VISIT_COOLDOWN_MINS: int = 3   # Suppress duplicate alerts for 3 mins during temporary track loss
    RETENTION_DAYS: int = 30       # Retention period for logs and face crop files
    
    # RTSP Stream URLs (comma-separated camera_id=rtsp_url pairs)
    RTSP_STREAMS: str = ""
    
    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()
