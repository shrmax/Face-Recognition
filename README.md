# 🚀 High-Performance Real-Time CCTV Head Detection & Face Recognition System

A production-ready, ultra-low latency real-time head detection, tracking, and face recognition system built with **FastAPI**, **ONNX Runtime (YOLOv11)**, **ByteTrack**, **InsightFace (SCRFD + ArcFace)**, **FAISS**, **MongoDB**, and **OpenCV**.

The system ingests multi-camera RTSP video feeds, executes ONNX hardware-accelerated single-box head detection (`head_class_id = 1`), tracks head motion seamlessly at 25 FPS using Kalman filter predictions, runs strict full-face quality verification, matches 512D ArcFace embeddings using FAISS vector search, and streams live annotated video overlays over WebSockets to an interactive HTML dashboard.

---

## 📌 Technical Highlights

- 👤 **Single Bounding Box Head Tracking**: Filters YOLOv11 ONNX outputs strictly for Class `1` (`head`) with aspect-ratio geometry guards ($0.4 \le h/w \le 1.8$), eliminating torso/upper-body boxes.
- ⚡ **ByteTrack & Kalman Motion Modeling**: 8-state constant-velocity Kalman filter tracking ($cx, cy, w, h, v_{cx}, v_{cy}, v_w, v_h$). Detector runs every 3rd frame; Kalman filter predicts motion on intermediate frames for 30 FPS throughput.
- 🎯 **Identity Locking & 0% Compute Bypass**: Once a head track identity is resolved (e.g. `Shravan (0.92)` or `Head #ID Unknown`), recognition job dispatching drops to **0 compute overhead** for the rest of that person's stay in frame.
- 🔍 **Full-Face & Zero-Blur Quality Filter**: 
  - **Landmark Geometry Verification**: Validates 5 facial keypoints (eyes, nose, mouth), minimum eye-to-eye distance ($\ge 12$px), vertical ordering ($\text{Eye\_Y} < \text{Nose\_Y} < \text{Mouth\_Y}$), and pose symmetry ($\text{nose\_offset} \le 0.45$). Rejects turned-away or back-of-head candidates.
  - **Laplacian Blur Variance Check**: Rejects out-of-focus or motion-blurred face crops ($\text{MIN\_BLUR\_VAR} \ge 35.0$).
- 🧠 **FAISS Multi-Vector Gallery & Diversity Enrollment**: Stores multiple distinct 512D ArcFace feature vectors per employee. Automatically deduplicates near-identical burst photos ($> 0.95$ similarity) during folder enrollment.
- 🖥️ **Apple Neural Engine (ANE) / GPU Acceleration**: ONNX models run on Apple Silicon ANE/GPU via `CoreMLExecutionProvider`.
- 📺 **Multi-Camera WebSocket Streaming**: Zero-latency RTSP frame capture thread (`nobuffer`, `low_delay`) with live annotated WebSocket stream broadcasting on `/face/ws`.

---

## 🛠 Tech Stack

| Category | Technology | Description |
| :--- | :--- | :--- |
| **Language** | Python 3.9+ | Primary runtime environment |
| **Backend Framework** | FastAPI | Async ASGI web server |
| **Head Detection** | YOLOv11 ONNX | Class-filtered head detector (`yolov11_phd_s.onnx`) |
| **Motion Tracking** | ByteTrack | IoU + Kalman filter 8-state motion tracker |
| **Face Analysis** | InsightFace | SCRFD detection & ArcFace 512D embedding extractor (`buffalo_m`) |
| **Vector Index** | FAISS (`IndexFlatIP`) | Cosine similarity vector search |
| **Database** | MongoDB (Motor) | Profile storage & event logging |
| **Computer Vision** | OpenCV | Preprocessing, letterboxing & frame rendering |
| **Hardware Accel.** | CoreML / ANE | Apple Silicon Neural Engine & GPU execution provider |

---

## 📂 Project Structure

```text
real-time-face-recognition-system/
│
├── core/
│   ├── head_detector.py   # YOLOv11 ONNX letterboxed single-box head detector
│   ├── tracker.py         # KalmanBoxTracker, ByteTrack & StreamTrackManager
│   ├── quality.py         # OpenCV Laplacian blur variance filter
│   ├── recognition.py     # InsightFace SCRFD, ArcFace embeddings & FAISS manager
│   └── stream_worker.py   # Zero-latency RTSP capture thread & AI processing loop
│
├── database/
│   ├── mongo.py           # Async MongoDB client & event logger
│   └── storage.py         # Face crop disk storage & auto-pruning
│
├── models/
│   └── yolov11_phd_s.onnx # ONNX head detector model file
│
├── scripts/
│   └── enroll_uploads.py  # Profile enrollment with deduplication (>0.95 sim check)
│
├── uploads/               # Profile folders with employee photos (e.g. uploads/shravan/)
├── crops/                 # Saved face crop snapshot images
├── config.py              # Central application configuration & settings
├── main.py                # FastAPI ASGI server, REST routes & WebSocket endpoint
├── .env                   # Environment variables file
├── .env.example           # Environment template file
└── requirements.txt       # Python dependencies
```

---

## ⚙️ Installation & Requirements

### 1. Prerequisites
- **Python 3.9+** installed
- **MongoDB** running locally (`mongodb://localhost:27017`) or remote URI

### 2. Setup Virtual Environment

```bash
# Clone repository
git clone https://github.com/shrmax/Face-Recognition.git
cd real-time-face-recognition-system

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration (`.env` and `config.py`)

Create your `.env` file from the provided template:

```bash
cp .env.example .env
```

### `.env` File Setup:
```env
# Server Config
HOST=0.0.0.0
PORT=8000

# MongoDB Database Configuration
MONGO_URI=mongodb://localhost:27017/
MONGO_DB_NAME=face_recognition_db

# RTSP Stream URLs (Comma-separated for multi-camera support)
RTSP_STREAMS=rtsp://user:pass@192.168.1.33:554/stream1
```

> **Note**: All AI model parameters, detection thresholds, and performance parameters are managed centrally in [`config.py`](file:///Users/shravan/Documents/learning/real-time-face-recognition-system/config.py).

---

## 📸 Profile Enrollment

To register employees or known individuals into the system:

1. Create a subfolder inside `uploads/` named after the person (e.g. `uploads/shravan/`).
2. Place face photos (`.jpg`, `.png`, `.jpeg`, `.webp`) in that folder (you can add 1, 5, 10, or 20 photos per person).
3. Run the enrollment script:

```bash
source venv/bin/activate
python scripts/enroll_uploads.py
```

> **Auto-Enrollment**: The system also automatically scans and enrolls new folders in `./uploads/` whenever you launch `main.py`.

---

## 🚀 How to Run the Application

### Step 1: Ensure MongoDB is Running
Make sure your local MongoDB instance is started:
```bash
brew services start mongodb-community
```

### Step 2: Activate Virtual Environment
```bash
source venv/bin/activate
```

### Step 3: Launch FastAPI Service
```bash
python main.py
```

### Step 4: Open Live Web Stream Dashboard
Open your browser and navigate to:
* **Interactive Live Dashboard**: [http://localhost:8000/stream?camera_id=cam_1](http://localhost:8000/stream?camera_id=cam_1)
* **Swagger Interactive API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **Service Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## 📡 REST API & WebSocket Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Service health status & active camera count |
| `GET` | `/stream` | Interactive HTML dark-mode live video dashboard |
| `WS` | `/face/ws?camera_id=cam_1` | High-speed WebSocket streaming of base64 frames & JSON telemetry |
| `POST` | `/streams/add` | Dynamically start a new RTSP camera worker feed |
| `DELETE` | `/streams/{camera_id}` | Stop an active RTSP camera worker stream |
| `GET` | `/logs` | Fetch detection event logs from MongoDB |
| `POST` | `/enroll` | Enroll a single image for a specific profile ID |

---

## 🧪 Verification & Unit Tests

Run the complete automated test suite (head detection, Kalman tracker, ByteTrack matching, FAISS vector search, and quality filter):

```bash
source venv/bin/activate
python -m unittest discover -s tests -p "*test*.py"
```

---

## 📄 License
This project is licensed under the MIT License.
