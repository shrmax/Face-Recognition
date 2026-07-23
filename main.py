

import os
import cv2
import asyncio
import base64
import numpy as np
import threading
import time
import pandas as pd
import pickle
import random
from queue import Queue, Empty
from datetime import datetime, timedelta
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from insightface.app import FaceAnalysis
import faiss

# ======================
# FASTAPI SETUP
# ======================
app = FastAPI()
router = APIRouter()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================
# OPTIMIZED INSIGHTFACE MODEL
# ======================
face_app = FaceAnalysis(
    name='buffalo_m',
    providers=['CUDAExecutionProvider'],
    allowed_modules=['detection', 'recognition']
)
face_app.prepare(
    ctx_id=0,
    det_size=(320, 320),
    det_thresh=0.3
)
print("InsightFace loaded on GPU")

# ======================
# LOAD KNOWN FACES
# ======================
VIDEOS_FOLDER = os.getenv("VIDEOS_FOLDER", "./Employee")
known_face_embeddings = {}
index = None
known_ids = []
FAISS_INDEX_PATH = "faiss_index.bin"
KNOWN_IDS_PATH = "known_ids.pkl"
EMBEDDINGS_PATH = "known_embeddings.pkl"
index_lock = threading.Lock()

def load_known_faces():
    global known_face_embeddings
    changed = False
    if os.path.exists(VIDEOS_FOLDER):
        if os.path.exists(EMBEDDINGS_PATH):
            with open(EMBEDDINGS_PATH, 'rb') as f:
                stored_data = pickle.load(f)
        else:
            stored_data = {}
       
        temp_embeddings = {}
        for person_folder in sorted(os.listdir(VIDEOS_FOLDER)):
            person_path = os.path.join(VIDEOS_FOLDER, person_folder)
            if not os.path.isdir(person_path):
                continue
            person_key = person_folder
            current_videos = sorted([f for f in os.listdir(person_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))])
           
            if person_key in stored_data:
                processed_videos = stored_data[person_key]['processed_videos']
                new_videos = [v for v in current_videos if v not in processed_videos]
                if new_videos:
                    new_embeddings = []
                    for file in new_videos:
                        vid_path = os.path.join(person_path, file)
                        cap = cv2.VideoCapture(vid_path)
                        frame_count = 0
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            if frame_count % 15 == 0:
                                faces = face_app.get(frame, max_num=1)
                                if faces and len(faces) > 0:
                                    new_embeddings.append(faces[0].embedding)
                            frame_count += 1
                        cap.release()
                    if new_embeddings:
                        changed = True
                        old_avg = stored_data[person_key]['avg']
                        old_count = stored_data[person_key]['count']
                        new_sum = np.sum(new_embeddings, axis=0)
                        updated_avg = (old_avg * old_count + new_sum) / (old_count + len(new_embeddings))
                        stored_data[person_key]['avg'] = updated_avg
                        stored_data[person_key]['count'] += len(new_embeddings)
                        print(f"Updated {person_folder} with {len(new_embeddings)} new samples")
                    stored_data[person_key]['processed_videos'] = current_videos[:]
                else:
                    print(f"{person_folder} up to date")
                temp_embeddings[person_key] = stored_data[person_key]['avg']
            else:
                embeddings = []
                for file in current_videos:
                    vid_path = os.path.join(person_path, file)
                    cap = cv2.VideoCapture(vid_path)
                    frame_count = 0
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if frame_count % 15 == 0:
                            faces = face_app.get(frame, max_num=1)
                            if faces and len(faces) > 0:
                                embeddings.append(faces[0].embedding)
                        frame_count += 1
                    cap.release()
                if len(embeddings) >= 3:
                    changed = True
                    avg = np.mean(embeddings, axis=0)
                    stored_data[person_key] = {
                        'avg': avg,
                        'count': len(embeddings),
                        'processed_videos': current_videos[:]
                    }
                    temp_embeddings[person_key] = avg
                    print(f"Loaded new {person_folder} ({len(embeddings)} samples)")
                else:
                    print(f"Skipped {person_folder} (<3 samples)")
       
        with open(EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump(stored_data, f)
       
        known_face_embeddings = temp_embeddings
    print(f"Total known identities: {len(known_face_embeddings)}")
    return changed

def build_index():
    known_ids_local = list(known_face_embeddings.keys())
    if not known_ids_local:
        return None, []
    db_embeddings = np.stack(list(known_face_embeddings.values()))
    norms = np.linalg.norm(db_embeddings, axis=1, keepdims=True)
    db_embeddings /= norms
    dimension = db_embeddings.shape[1]
    new_index = faiss.IndexFlatIP(dimension)
    new_index.add(db_embeddings)
    print(f"Built new FAISS index with {len(known_ids_local)} known faces")
    return new_index, known_ids_local

def load_or_build_index():
    global index, known_ids
    changed = load_known_faces()
    if not changed and os.path.exists(FAISS_INDEX_PATH) and os.path.exists(KNOWN_IDS_PATH):
        try:
            temp_index = faiss.read_index(FAISS_INDEX_PATH)
            with open(KNOWN_IDS_PATH, 'rb') as f:
                temp_ids = pickle.load(f)
            with index_lock:
                index = temp_index
                known_ids = temp_ids
            print(f"Loaded existing FAISS index with {len(known_ids)} known faces")
            return
        except Exception as e:
            print(f"Error loading FAISS: {e}, rebuilding...")
    
    new_index, new_ids = build_index()
    if new_index is not None:
        with index_lock:
            index = new_index
            known_ids = new_ids
        faiss.write_index(index, FAISS_INDEX_PATH)
        with open(KNOWN_IDS_PATH, 'wb') as f:
            pickle.dump(known_ids, f)
        print(f"Built and saved FAISS index with {len(known_ids)} known faces")
    else:
        with index_lock:
            index = None
            known_ids = []

load_or_build_index()

# ======================
# RELOAD ENDPOINT
# ======================
@app.post("/faces/reload")
async def reload_faces():
    print("Manual reload triggered...")
    start_time = time.time()
    changed = load_known_faces()
    new_index, new_ids = build_index()
    if new_index is None:
        return {"status": "error", "message": "No known faces to index"}
    with index_lock:
        global index, known_ids
        index = new_index
        known_ids = new_ids
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(KNOWN_IDS_PATH, 'wb') as f:
        pickle.dump(known_ids, f)
    duration = time.time() - start_time
    print(f"Reload complete: {len(known_ids)} faces, {duration:.2f}s")
    return {
        "status": "success",
        "num_faces": len(known_ids),
        "changed": changed,
        "duration": f"{duration:.2f}s"
    }

# ======================
# FIXED RTSP CLASS (SINGLE __init__)
# ======================
class RTSPFaceRecognition:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.frame_queue = Queue(maxsize=3)
        self.running = True
        self.cap = None
        self.last_frame = None
        self.last_processed_frame = None
        # Changed: Replaced detected_today and today with cooldown mechanism
        self.detection_cooldown = timedelta(minutes=1)  # 1 minute cooldown
        self.last_detection_time = {}  # Track last detection timestamp per person
        self.pending_messages = []
        self.lock = threading.Lock()
        self.cap_lock = threading.Lock()
        self.reconnect_count = 0
        self.max_reconnects = None
        self.processing = False
        self.last_process_time = 0
        self.target_fps = 5
        self.frame_skip_counter = 0
        self.process_every_n_frames = 2
        self.last_annotations = []
        self.last_known_detected = False
        self.initial_frames_sent = False
        self._first_frame_received = False  # Track if we've received first frame
        
        # Start threads
        threading.Thread(target=self._capture_loop, daemon=True, name=f"CaptureThread_{id(self)}").start()
        threading.Thread(target=self._process_loop, daemon=True, name=f"ProcessThread_{id(self)}").start()
        
    def _connect(self):
        with self.cap_lock:
            try:
                if self.cap is not None:
                    self.cap.release()
                time.sleep(0.5)
                
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
                self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                
                if not self.cap.isOpened():
                    print(f"Failed to open RTSP stream: {self.rtsp_url}")
                    return False
                
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FPS, 15)
                
                ret, test_frame = self.cap.read()
                if not ret or test_frame is None:
                    self.cap.release()
                    return False
                
                with self.lock:
                    self.last_frame = test_frame.copy()
                    self.last_processed_frame = test_frame.copy()
                    self._first_frame_received = True
                
                print(f"✓ RTSP connected: {self.rtsp_url}")
                self.reconnect_count = 0
                return True
                
            except Exception as e:
                print(f"Connection error: {e}")
                if self.cap:
                    self.cap.release()
                self.cap = None
                return False
    
    def _capture_loop(self):
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        while self.running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    if self.max_reconnects is not None and self.reconnect_count >= self.max_reconnects:
                        print("Max reconnection attempts reached. Stopping.")
                        self.running = False
                        break
                    
                    print(f"Reconnecting... (attempt {self.reconnect_count + 1})")
                    if not self._connect():
                        self.reconnect_count += 1
                        sleep_time = min(2 ** min(self.reconnect_count, 10), 60) + random.uniform(0, 1)
                        print(f"Reconnect failed, waiting {sleep_time:.1f}s")
                        time.sleep(sleep_time)
                        continue
                
                with self.cap_lock:
                    if self.cap and self.cap.isOpened():
                        ret, frame = self.cap.read()
                    else:
                        ret, frame = False, None
                
                if ret and frame is not None:
                    consecutive_failures = 0
                    
                    # Always keep the latest frame
                    with self.lock:
                        self.last_frame = frame.copy()
                        if not self._first_frame_received:
                            self._first_frame_received = True
                    
                    # Put in queue for processing
                    if not self.frame_queue.full():
                        try:
                            self.frame_queue.put_nowait(frame)
                        except:
                            pass
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        print("Too many consecutive failures, reconnecting...")
                        with self.cap_lock:
                            if self.cap:
                                self.cap.release()
                            self.cap = None
                        consecutive_failures = 0
                        time.sleep(1)
                
                time.sleep(0.001)
                
            except Exception as e:
                print(f"Error in capture loop: {e}")
                consecutive_failures += 1
                time.sleep(1)
    
    def _process_loop(self):
        while self.running:
            try:
                now = time.time()
                
                if now - self.last_process_time < 1.0 / self.target_fps:
                    time.sleep(0.005)
                    continue
                
                if self.processing:
                    time.sleep(0.005)
                    continue
                
                try:
                    frame = self.frame_queue.get_nowait()
                except Empty:
                    time.sleep(0.01)
                    continue
                
                # Removed: date-based reset logic
                # Now using cooldown mechanism instead
                
                self.processing = True
                try:
                    processed_frame = self._recognize_and_draw(frame)
                    with self.lock:
                        self.last_processed_frame = processed_frame
                finally:
                    self.processing = False
                
                self.last_process_time = now
                
            except Exception as e:
                print(f"Error in process loop: {e}")
                self.processing = False
                time.sleep(0.1)
    
    def _recognize_and_draw(self, frame):
        try:
            processed_frame = frame.copy()
            faces = face_app.get(processed_frame, max_num=3)
            
            annotations = []
            known_detected = False
            
            for face in faces:
                if face.det_score < 0.3:
                    continue
                
                bbox = face.bbox.astype(int)
                face_width = bbox[2] - bbox[0]
                
                if face_width < 60:
                    continue
                
                emb = face.embedding
                qnorm = emb / np.linalg.norm(emb)
                
                best_id = "Unknown"
                best_sim = 0.0
                
                if index is not None and len(known_ids) > 0:
                    with index_lock:
                        local_index = index
                        local_known_ids = known_ids
                        if local_index is not None and len(local_known_ids) > 0:
                            distances, indices = local_index.search(qnorm.reshape(1, -1), k=1)
                            best_sim = distances[0][0]
                            if len(indices[0]) > 0:
                                best_idx = indices[0][0]
                                best_id = local_known_ids[best_idx]
                
                if best_sim >= 0.50:
                    color = (0, 255, 0)
                    label = f"{best_id} ({best_sim:.2f})"
                else:
                    color = (0, 0, 255)
                    label = f"Unknown ({best_sim:.2f})"
                
                cv2.rectangle(processed_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cv2.putText(processed_frame, label, (bbox[0], bbox[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                annotations.append({
                    'bbox': list(bbox),
                    'color': color,
                    'label': label
                })
                
                # Changed: 1-minute cooldown logic instead of daily reset
                if best_sim >= 0.50:
                    known_detected = True
                    current_time = datetime.now()
                    
                    # Check if cooldown period has passed since last detection
                    if best_id not in self.last_detection_time or \
                       (current_time - self.last_detection_time[best_id]) > self.detection_cooldown:
                        
                        self.last_detection_time[best_id] = current_time
                        self._log_detection(best_id)
            
            if known_detected:
                cv2.putText(processed_frame, "KNOWN PERSON DETECTED!", (10, 40),
                           cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 3)
            
            self.last_annotations = annotations
            self.last_known_detected = known_detected
            
            return processed_frame
            
        except Exception as e:
            print(f"Error in recognition: {e}")
            return frame
    
    def _log_detection(self, emp_id):
        message = {"type": "detection", "emp_id": emp_id}
        with self.lock:
            self.pending_messages.append(message)
        print(f"[LOGGED] {emp_id}")
    
    async def get_frame_base64(self):
        """Get frame as base64 - ensures we return something even if processing isn't done"""
        # Wait a bit for first frame if needed
        wait_attempts = 0
        while not self._first_frame_received and wait_attempts < 20:
            await asyncio.sleep(0.1)
            wait_attempts += 1
        
        with self.lock:
            # Try to get processed frame first, fall back to raw frame
            frame_to_send = self.last_processed_frame if self.last_processed_frame is not None else self.last_frame
            if frame_to_send is None:
                return None
            frame = frame_to_send.copy()
        
        try:
            _, buffer = cv2.imencode(
                '.jpg',
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 40, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
            )
            return base64.b64encode(buffer).decode()
        except Exception as e:
            print(f"Error encoding frame: {e}")
            return None
    
    def stop(self):
        """Clean shutdown"""
        print("Stopping recognizer...")
        self.running = False
        time.sleep(0.5)
        with self.cap_lock:
            if self.cap:
                self.cap.release()
            self.cap = None

# ======================
# WEBSOCKET ENDPOINT (FIXED)
# ======================
# Store active recognizers for cleanup
active_recognizers = {}
recognizers_lock = threading.Lock()

@router.websocket("/face/ws")
async def face_websocket(websocket: WebSocket, rtsp_url: str = Query(...)):
    await websocket.accept()
    
    # Create unique ID for this connection
    connection_id = f"{rtsp_url}_{id(websocket)}"
    
    # Check if we already have a recognizer for this RTSP URL
    if rtsp_url in active_recognizers:
        recognizer = active_recognizers[rtsp_url]
        print(f"Reusing existing recognizer for {rtsp_url}")
    else:
        recognizer = RTSPFaceRecognition(rtsp_url)
        active_recognizers[rtsp_url] = recognizer
        print(f"Created new recognizer for {rtsp_url}")
    
    try:
        # Send initial frames quickly to establish stream
        frame_count = 0
        while True:
            frame_b64 = await recognizer.get_frame_base64()
            
            if frame_b64:
                # Send frame
                await websocket.send_json({"frame": frame_b64, "type": "frame"})
                frame_count += 1
                
                # Log occasionally
                if frame_count % 30 == 0:
                    print(f"Sent {frame_count} frames for {rtsp_url}")
            
            # Send pending detection messages
            with recognizer.lock:
                messages = recognizer.pending_messages.copy()
                recognizer.pending_messages.clear()
            
            for msg in messages:
                print(f"[WS SENDING TO CLIENT] {msg} for {rtsp_url}")
                await websocket.send_json(msg)
            
            # Throttle to reasonable FPS
            await asyncio.sleep(0.066)  # ~15 FPS
            
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for {rtsp_url}")
        # Don't stop the recognizer immediately - keep it for potential reconnection
        # This prevents the need to restart the stream
        
        # Optional: Implement timeout cleanup
        # For now, we keep the recognizer alive
        
    except Exception as e:
        print(f"WS Error: {e}")
        import traceback
        traceback.print_exc()

# Optional: Cleanup old recognizers (you can call this periodically)
async def cleanup_old_recognizers():
    """Clean up recognizers that haven't been used recently"""
    # This is optional - implement if needed
    pass

app.include_router(router)

# ======================
# HEALTH CHECK ENDPOINT
# ======================
@app.get("/health")
async def health_check():
    return {"status": "ok", "active_recognizers": len(active_recognizers)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)




