# 🚀 Real-Time Face Recognition System using InsightFace, FAISS & MongoDB

A production-ready, high-performance real-time face recognition system built with **FastAPI**, **InsightFace (SCRFD + ArcFace)**, **FAISS**, **MongoDB**, and **OpenCV**.

The system captures multi-camera RTSP video streams, executes GPU/CPU accelerated face detection and feature extraction, performs vector similarity searches with FAISS, logs events into MongoDB with retention management, and streams real-time recognition results over WebSockets and an interactive HTML web dashboard.

---

## 📌 Overview

This project is tailored for scalable enterprise face recognition applications, including:

- Smart Access Control & Entrance Monitoring
- Automated Employee Attendance Tracking
- Perimeter Security & Real-Time Alerts
- Multi-Camera Surveillance Systems
- Visitor Identification & VIP Recognition

The system includes automatic dataset enrollment, multi-vector gallery profiling, face quality/blur filtering, multi-person tracking with visit cooldowns, and automatic RTSP stream reconnection.

> **Note:** Dataset folders (`uploads/`, `crops/`), FAISS binary index files (`faiss_index.bin`), and identity pkl files are excluded from git tracking for privacy and security.

---

## ✨ Features

- 🎯 **Real-Time Multi-Camera Streaming**: Asynchronous RTSP video ingestion with background Watchdog reconnection.
- 🚀 **InsightFace Deep Learning**: Multi-person face detection (SCRFD) & 512-D embedding extraction (ArcFace / buffalo_m).
- 🔍 **FAISS Vector Search**: Fast vector similarity matching for registered profiles with customizable confidence thresholds.
- 🍃 **MongoDB Integration (Async Motor)**: Profile management, multi-vector galleries per identity, and structured event logging.
- 📸 **Automated Profile Enrollment**: Syncs photo directories (`uploads/<name>/`) to MongoDB and FAISS automatically on startup or via `scripts/enroll_uploads.py`.
- 👁️ **Face Quality & Blur Filtering**: Filters out blurred or sub-pixel face crops using OpenCV Laplacian variance.
- ⏱️ **Tracking & Visit Cooldown**: Bounding-box IoU tracking prevents duplicate event triggers during continuous camera presence.
- 🗑️ **Automatic Retention Cleanup**: TTL index and daily scheduled cleanup for log retention and face crop storage.
- 📺 **Live Web Dashboard & WebSockets**: Built-in dark-mode web viewer (`/stream`) and high-speed WebSocket stream endpoint (`/face/ws`).
- ⚡ **Multi-Worker Architecture**: Threaded video capture decoupled from AI inference worker pools to maintain steady FPS.

---

## 🛠 Tech Stack

| Category               | Technology           | Description                                      |
| ---------------------- | -------------------- | ------------------------------------------------ |
| **Language**           | Python 3.9+          | Core programming language                        |
| **Backend Framework**  | FastAPI              | High-performance async API server                |
| **Face Recognition**   | InsightFace          | SCRFD detection & ArcFace feature extraction     |
| **Vector Search**      | FAISS                | High-speed L2/IP vector similarity search        |
| **Database**           | MongoDB & Motor      | Async profile storage & event logging with TTL   |
| **Computer Vision**    | OpenCV               | Frame processing, blur detection & web streaming |
| **Server & Real-Time** | Uvicorn & WebSockets | ASGI server and WebSocket communication          |
| **Configuration**      | Pydantic Settings    | Environment-driven settings management           |

---

## 📂 Project Structure

```text
real-time-face-recognition-system/
│
├── core/
│   ├── quality.py         # Face quality & Laplacian blur filtering
│   ├── recognition.py     # FAISS vector manager & RecognitionWorker pool
│   ├── stream_worker.py   # Threaded RTSP capture & Watchdog reconnection
│   └── tracker.py         # IoU multi-face tracking & visit cooldown manager
│
├── database/
│   ├── mongo.py           # Async MongoDB client, indexes & event logger
│   └── storage.py         # Face crop storage manager & auto-pruning
│
├── scripts/
│   └── enroll_uploads.py  # Script to enroll photos from uploads/ into MongoDB & FAISS
│
├── docs/                  # API and project documentation
├── uploads/              # Gallery directories containing images for enrollment (e.g. uploads/john_doe/)
├── crops/                # Saved face crop images from detection events
│
├── config.py             # App configuration & environment settings
├── main.py               # FastAPI server, REST routes, HTML stream UI & WebSockets
├── faiss_index.bin       # FAISS vector database file
├── known_ids.pkl         # Mappings for face identity IDs to names
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variables template
└── README.md             # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone Repository & Setup Virtual Environment

```bash
git clone https://github.com/Sahil592003/real-time-face-recognition-system.git
cd real-time-face-recognition-system

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Key configuration parameters in `.env`:

| Parameter                  | Default                     | Description                                      |
| -------------------------- | --------------------------- | ------------------------------------------------ |
| `HOST`                     | `0.0.0.0`                   | Server listen host                               |
| `PORT`                     | `8000`                      | Server listen port                               |
| `MONGO_URI`                | `mongodb://localhost:27017` | MongoDB connection URI                           |
| `MONGO_DB_NAME`            | `face_recognition_db`       | MongoDB database name                            |
| `CROP_DIR`                 | `./crops`                   | Path to store detected face crops                |
| `DET_WIDTH` / `DET_HEIGHT` | `640` / `640`               | Face detector input resolution                   |
| `DET_THRESH`               | `0.3`                       | Minimum face detection threshold                 |
| `SAMPLE_FPS`               | `6`                         | Frame sampling rate for AI inference             |
| `MIN_BLUR_VAR`             | `100.0`                     | Minimum Laplacian variance for blur filter       |
| `HIGH_CONF_THRESH`         | `0.55`                      | Confidence threshold for profile matching        |
| `LOW_CONF_THRESH`          | `0.40`                      | Threshold below which auto-enrollment triggers   |
| `VISIT_COOLDOWN_MINS`      | `3`                         | Duplicate alert suppression window (minutes)     |
| `RETENTION_DAYS`           | `30`                        | Data retention period for logs and crop images   |
| `RTSP_STREAMS`             | `""`                        | Comma-separated list of RTSP URLs for auto-start |

---

## 📸 Face Profile Enrollment

To enroll known individuals into the recognition database:

1. Create a subfolder inside `uploads/` named after the person (e.g. `uploads/john_doe/`).
2. Add clear face photos (`.jpg`, `.png`, `.webp`) into that folder.
3. Run the enrollment script (or restart the FastAPI server, which auto-syncs `uploads/` on startup):

```bash
python scripts/enroll_uploads.py
```

This extracts 512-D ArcFace embeddings, updates the MongoDB `face_profiles` collection, and builds the FAISS vector index (`faiss_index.bin`).

---

## 🚀 Running the System

Start the FastAPI server:

```bash
python main.py
```

Or using Uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

- **Server URL**: `http://localhost:8000`
- **Swagger API Docs**: `http://localhost:8000/docs`
- **Web Stream Dashboard**: `http://localhost:8000/stream`

---

## 📡 API Endpoints

### Health Check

```http
GET /health
```

**Response:**

```json
{
  "status": "ok",
  "active_streams": 1,
  "stream_ids": ["cam_1"],
  "faiss_registered_vectors": 50,
  "queue_size": 0,
  "mongo_connected": true
}
```

### Add RTSP Stream Worker

```http
POST /streams/add?camera_id=cam1&rtsp_url=rtsp://192.168.1.100:554/stream1
```

### Remove RTSP Stream Worker

```http
DELETE /streams/cam1
```

### Fetch Detection Logs

```http
GET /logs?limit=50&camera_id=cam1
```

### Web Live Viewer UI

```http
GET /stream?camera_id=cam_1
```

### WebSocket Streaming

```websocket
ws://localhost:8000/face/ws?camera_id=cam_1&rtsp_url=rtsp://192.168.1.100:554/stream1
```

Streams real-time JPEG frame buffers (`base64`) along with detection metadata and bounding boxes.

---

## 🔄 System Architecture & Data Flow

```text
 RTSP Camera Stream
        │
        ▼
 StreamWorker (Background Thread)
  - Frame Ingestion
  - Reconnection Watchdog
        │
        ▼
 Async Job Queue (Queue maxsize=100)
        │
        ▼
 RecognitionWorker Pool
  - Quality Filter (Laplacian Blur & Box Size)
  - InsightFace SCRFD Detection & ArcFace Embeddings
  - FAISS Vector Similarity Search
  - IoU Bounding Box Tracking & Visit Cooldown
        │
        ├──────────────────────────┐
        ▼                          ▼
 MongoDB (Motor Driver)    WebSocket Streaming
  - Face Profiles           - Base64 Frame Buffer
  - Event Logging           - Dynamic Annotation Overlay
  - TTL Retention          - HTML Web Viewer (/stream)
```

---

## 📄 License

This project is licensed under the MIT License.

---

source venv/bin/activate
python main.py
