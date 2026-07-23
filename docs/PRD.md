# Product Requirements Document (PRD)
## Real-Time Vision Processing Service ("Black Box")

---

## 1. Executive Summary & Goal

Build a lightweight, asynchronous vision processing service ("Black Box") that consumes live RTSP/RTP camera streams, detects and tracks faces continuously (including dense crowds of 100+ people per frame), logs appearance events without redundant computations (via Frame Gating and Visit Cooldown), and dynamically registers/updates identity embeddings in a vector database.

The system is engineered for enterprise-grade video analytics (employee attendance, smart access control, multi-camera surveillance) with end-to-end processing latency under 200 ms and optimal resource utilization.

---

## 2. Functional Requirements (FR)

### **FR-1: Stream Ingestion & Preprocessing**
- Ingest live RTSP/RTP camera streams with continuous background buffer hygiene.
- Decode video frames at full stream frame rate (30 FPS) while running AI detection on a sampled subset (5–10 FPS).
- Support dynamic multi-stream management (add/stop streams at runtime via REST API).
- Handle network drops and silent socket hangs gracefully via a dedicated RTSP Watchdog with exponential backoff auto-reconnection.

### **FR-2: High-Density Face Detection & Tracking**
- Perform face detection using InsightFace SCRFD configured with unlimited detection count (`max_num=0`) and adaptive high resolution (`det_size=(640, 640)` / `(1280, 1280)`).
- Support tracking of 100+ faces in a single frame simultaneously.
- Assign persistent `track_id`s across consecutive frames using Roboflow `supervision` (`sv.ByteTrack`).

### **FR-3: Identification, Dynamic Auto-Enrollment & Profile Gallery**
- **Known Identity (`sim >= 0.55`):** Match query embedding against the FAISS vector database. Check profile's **Visit Cooldown Window (2–5 mins)**:
  - If within cooldown: Log detection internally, but **suppress duplicate event alerts**.
  - If new visit: Issue a single `KNOWN_IDENTITY` event log.
  - **Progressive Learning:** If the new high-quality embedding offers a distinct angle/lighting (`0.55 <= sim < 0.85`), add it to the profile's multi-vector gallery (up to 5 vectors max per profile in FAISS) to continuously improve future recognition.
- **Uncertain Candidate (`0.40 <= sim < 0.55`):** Log as an unverified/review candidate without auto-enrolling.
- **New Identity (`sim < 0.40`):** Compute ArcFace feature embedding, generate a new `profile_id` (e.g. `VISITOR_1042`), insert vector into FAISS dynamically in memory (`index.add()`), store profile in MongoDB, and issue a `NEW_IDENTITY` log event.

### **FR-4: Motion Gating & Occlusion Cooldown**
- **Frame Gating:** Once a face `track_id` is successfully processed, **freeze heavy ArcFace re-identification calculations** for that specific `track_id` while the person remains continuously in frame.
- **Visit Cooldown:** If a person is temporarily occluded (e.g., walking behind a pillar or door frame for a few seconds) causing ByteTrack to generate a new `track_id`, matching against FAISS recognizes the same `profile_id` and applies the Visit Cooldown to prevent duplicate event logs.

### **FR-5: Production Safeguards & Quality Control**
- **Face Quality & Blur Filter:** Compute OpenCV Laplacian blur variance and minimum bounding box dimensions (>= 60px). Reject blurry or profile-angle crops before submitting to ArcFace.
- **Best-Frame Selection:** Collect crops across a 3–5 frame window per track and select the sharpest crop with highest confidence.
- **Retention & Cleanup:** Purge MongoDB log events older than $N$ days via TTL index and run daily background tasks to prune local face crop files.

---

## 3. Non-Functional Requirements (NFR)

- **Latency:** End-to-end processing delay under **200 ms** per frame.
- **Efficiency:** Minimize GPU/CPU resource usage by 80% through 2-tier frame sampling (30 FPS decode -> 5–10 FPS detection) and 99% through Track Frame Gating (ArcFace executed strictly once per track ID).
- **Scalability:** Single-process, multi-threaded worker pipeline handling multiple camera streams concurrently without inter-process IPC overhead.
- **Reliability:** Auto-reconnect hung RTSP feeds within 5 seconds without manual intervention.

---

## 4. The 10-Stage Pipeline Specification

| Stage | Name | Description |
| :--- | :--- | :--- |
| **1** | **RTSP Ingestion** | Pull CCTV stream, maintain buffer hygiene, auto-reconnect on socket drops. |
| **2** | **Frame Sampling** | Decode at 30 FPS, sample 5–10 FPS for AI detection. |
| **3** | **Face Detection** | SCRFD face detection (`max_num=0`, high-res grid). |
| **4** | **Tracking** | Assign persistent `track_id` using `sv.ByteTrack`. |
| **5** | **Track State Check** | New `track_id` → `PENDING`; existing `track_id` → `PROCESSED` (skip ArcFace). |
| **6** | **Quality Gate & Queue** | Validate crop sharpness (Laplacian variance > 100), enqueue to `asyncio.Queue`. |
| **7** | **Face Recognition** | Worker pool pulls crop from queue, extracts ArcFace 512-dim embedding vector. |
| **8** | **Matching & Gallery Update**| FAISS vector search. If match (`sim >= 0.55`), check Visit Cooldown & update multi-vector gallery. If `sim < 0.40`, auto-enroll new profile. |
| **9** | **Logging & Crop Save** | Write event to MongoDB (`detection_events`) and save face crop JPEG to disk (`/crops/`). |
| **10**| **Track Cleanup** | Drop track from state table after $N$ missing frames. |

---

## 5. Technology Stack Summary

| Layer | Component / Tool |
| :--- | :--- |
| **Stream Ingestion** | OpenCV `VideoCapture` + RTSP TCP transport + Watchdog Thread |
| **Face Detection** | InsightFace **SCRFD** (ONNX Runtime / CUDA) |
| **Face Recognition** | InsightFace **ArcFace** (512-dimensional embeddings) |
| **Tracking & Annotation** | Roboflow **Supervision** (`sv.Detections`, `sv.ByteTrack`, `sv.BoxAnnotator`) |
| **Vector Search** | **FAISS** (`IndexFlatIP`, multi-embedding vector index per profile) |
| **In-Memory Queue** | Python `asyncio.Queue` (zero IPC serialization delay) |
| **Database** | **MongoDB** (`motor` async driver) |
| **Backend & WebSockets** | **FastAPI** + Uvicorn |

---

## 6. Database Schema Design (MongoDB)

### Collection: `detection_events`
```json
{
  "_id": ObjectId("65b1c..."),
  "track_id": 104,
  "profile_id": "VISITOR_1042",
  "camera_id": "CAM_ENTRANCE_01",
  "confidence": 0.88,
  "event_type": "NEW_IDENTITY",
  "timestamp": ISODate("2026-07-23T12:00:00Z"),
  "bbox": [120, 45, 280, 205],
  "crop_path": "./crops/2026-07-23/VISITOR_1042/track_104.jpg",
  "review_required": false,
  "visit_cooldown_active": false
}
```

### Collection: `face_profiles`
```json
{
  "_id": ObjectId("65b1d..."),
  "profile_id": "VISITOR_1042",
  "name": "Unassigned Visitor #1042",
  "created_at": ISODate("2026-07-23T12:00:00Z"),
  "embeddings": [
    [0.024, -0.118, 0.452, "... vector 1 ..."],
    [0.031, -0.105, 0.440, "... vector 2 (new angle) ..."]
  ],
  "sample_count": 2,
  "last_seen": ISODate("2026-07-23T12:05:00Z"),
  "status": "active"
}
```
