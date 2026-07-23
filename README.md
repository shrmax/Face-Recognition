# 🚀 Real-Time Face Recognition System using InsightFace & FAISS

A production-ready real-time face recognition system built using **FastAPI**, **InsightFace**, **FAISS**, and **OpenCV**. The system performs high-speed face detection and recognition from RTSP camera streams using GPU acceleration and streams live recognition results through WebSocket communication.

---

## 📌 Overview

This project is designed for enterprise-grade face recognition applications including:

- Employee Attendance System
- Smart Access Control
- Office Security
- Visitor Management
- Surveillance Systems
- Multi-Camera Face Recognition

The system automatically generates face embeddings from registered videos, builds a FAISS vector database, and recognizes known individuals in real time with low latency.

> **Note:** Employee videos, images, embeddings, and FAISS index files are intentionally excluded from this repository for privacy and security. Users can create their own dataset by adding videos to the configured directory.

---

# ✨ Features

- 🎯 Real-Time Face Detection & Recognition
- 🚀 GPU Accelerated InsightFace Inference
- 📡 RTSP Camera Streaming
- 🔍 FAISS Vector Similarity Search
- ⚡ FastAPI REST APIs
- 📺 WebSocket Live Streaming
- 🔄 Automatic Face Database Reload
- 👤 Multi-Person Recognition
- 🔁 Automatic RTSP Reconnection
- ⏱ Recognition Cooldown
- 🧵 Multi-threaded Processing
- 📊 Health Check Endpoint
- 💻 Supports both GPU (CUDA) and CPU execution depending on the installed ONNX Runtime provider.

---

# 🛠 Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| Backend | FastAPI |
| Face Recognition | InsightFace |
| Similarity Search | FAISS |
| Computer Vision | OpenCV |
| Numerical Computing | NumPy |
| Data Processing | Pandas |
| GPU Inference | ONNX Runtime |
| Communication | WebSocket |
| Server | Uvicorn |

---

# 📂 Project Structure

```text
Real-Time-Face-Recognition-System/
│
├── main.py
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── LICENSE
└── docs/
    └── API.md
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/Sahil592003/real-time-face-recognition-system.git

cd real-time-face-recognition-system
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure Environment

Rename

```
.env.example
```

to

```
.env
```

Update

```
VIDEOS_FOLDER
```

with your dataset location.

---

## Run

```bash
python main.py
```

Server starts at

```
http://localhost:8000
```

Swagger Documentation

```
http://localhost:8000/docs
```

---

# 📡 API Endpoints

## Health Check

```
GET /health
```

Response

```json
{
  "status":"ok",
  "active_recognizers":1
}
```

---

## Reload Face Database

```
POST /faces/reload
```

Response

```json
{
  "status":"success",
  "num_faces":120,
  "changed":true,
  "duration":"2.34s"
}
```

---

## Live Recognition

```
WebSocket

ws://localhost:8000/face/ws?rtsp_url=<RTSP_URL>
```

Returns

- Live Camera Frames
- Face Recognition Results
- Detection Events

---

# 🔄 Recognition Workflow

```
RTSP Camera
      │
      ▼
OpenCV Video Capture
      │
      ▼
InsightFace Detection
      │
      ▼
Face Embedding Extraction
      │
      ▼
FAISS Similarity Search
      │
      ▼
Identity Recognition
      │
      ▼
WebSocket Streaming
      │
      ▼
Frontend Dashboard
```

---

# 🚀 Performance Optimizations

- GPU Accelerated Inference
- Multi-threaded Processing
- Automatic RTSP Reconnection
- FAISS Vector Indexing
- Embedding Caching
- Low Latency Recognition
- Thread-safe Processing
- Optimized Frame Handling

---

# 🔒 Privacy

This repository does **not** include:

- Employee Images
- Employee Videos
- Face Embeddings
- FAISS Index Files
- Production Datasets

Users can create their own dataset by adding videos/images to the configured dataset directory.

---

# 📌 Future Improvements

- Docker Support
- JWT Authentication
- Face Registration API
- Anti-Spoofing
- Face Mask Detection
- Multi-Camera Dashboard
- PostgreSQL Integration
- Kubernetes Deployment
- Face Analytics Dashboard

---

# 📄 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Sahil Ghadge**

AI/ML Engineer | Computer Vision Engineer | Generative AI Engineer

### Skills

- Computer Vision
- Deep Learning
- Face Recognition
- FastAPI
- Python
- OpenCV
- InsightFace
- FAISS
- YOLO
- Generative AI
- LangChain
- Enterprise AI Automation

---

⭐ If you found this project useful, consider giving it a **Star** on GitHub.

# 1. Activate the virtual environment
source venv/bin/activate

# 2. Start the server
python3 main.py

# 3. Add an RTSP stream (in a separate terminal)
curl -X POST "http://localhost:8000/streams/add?camera_id=cam1&rtsp_url=rtsp://YOUR_CAMERA_IP:554/stream"

# 4. Check health
curl http://localhost:8000/health

# 5. View detection logs
curl http://localhost:8000/logs?limit=20

# 6. Remove a stream
curl -X DELETE http://localhost:8000/streams/cam1

# 7. WebSocket live stream (auto-registers if stream not active)
# Connect to: ws://localhost:8000/face/ws?camera_id=cam1&rtsp_url=rtsp://YOUR_CAMERA_IP:554/stream
