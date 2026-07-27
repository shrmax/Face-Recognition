from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from views.layout import render_layout

router = APIRouter(tags=["Views"])


@router.get("/stream", response_class=HTMLResponse)
async def get_stream_page():
    content = """
    <style>
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
            margin-bottom: 16px;
        }
        .header h1 {
            font-size: 1.8rem;
            font-weight: 700;
            color: #f8fafc;
            letter-spacing: -0.5px;
        }
        .header p {
            color: #94a3b8;
            font-size: 0.95rem;
            margin-top: 4px;
        }

        .header-actions {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .stream-counter-badge {
            background: #0f172a;
            border: 1px solid #334155;
            padding: 8px 16px;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 600;
            color: #38bdf8;
        }

        .btn-add-stream {
            background: #0284c7;
            color: #ffffff;
            border: none;
            padding: 9px 18px;
            border-radius: 10px;
            font-size: 0.9rem;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: background 0.2s ease;
        }
        .btn-add-stream:hover {
            background: #0369a1;
        }
        .btn-add-stream:disabled {
            background: #334155;
            color: #64748b;
            cursor: not-allowed;
        }

        /* Dynamic Grid Layout */
        .dynamic-cam-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
            gap: 24px;
            width: 100%;
            transition: all 0.3s ease;
        }

        .dynamic-cam-grid.single-stream {
            grid-template-columns: 1fr;
        }

        .cam-card {
            background: #1e293b;
            border-radius: 16px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
            overflow: hidden;
            border: 1px solid #334155;
            display: flex;
            flex-direction: column;
            transition: border-color 0.2s ease;
        }
        .cam-card:hover {
            border-color: #475569;
        }

        /* Dedicated Header Bar ABOVE Video */
        .cam-card-bar {
            background: #0f172a;
            border-bottom: 1px solid #334155;
            padding: 12px 18px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 10;
        }

        .cam-bar-left {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .cam-bar-right {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .cam-label {
            font-size: 0.9rem;
            font-weight: 700;
            color: #38bdf8;
            letter-spacing: 0.3px;
        }

        .status-badge {
            background: rgba(30, 41, 59, 0.8);
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.78rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
            border: 1px solid #334155;
        }

        .dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #ef4444;
        }
        .dot.connected {
            background-color: #10b981;
            box-shadow: 0 0 8px #10b981;
        }

        .btn-action-sm {
            background: #1e293b;
            color: #cbd5e1;
            border: 1px solid #334155;
            padding: 6px 12px;
            border-radius: 8px;
            font-size: 0.8rem;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
            transition: all 0.2s ease;
        }
        .btn-action-sm:hover {
            background: #334155;
            color: #ffffff;
        }
        .btn-action-danger {
            background: rgba(239, 68, 68, 0.15);
            color: #ef4444;
            border: 1px solid rgba(239, 68, 68, 0.3);
            padding: 6px 12px;
            border-radius: 8px;
            font-size: 0.8rem;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
            transition: all 0.2s ease;
        }
        .btn-action-danger:hover {
            background: #ef4444;
            color: #ffffff;
        }

        /* Video Frame Box */
        .cam-frame-box {
            position: relative;
            width: 100%;
            background: #0b0f19;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 320px;
        }

        .cam-frame {
            width: 100%;
            height: auto;
            max-height: 720px;
            object-fit: contain;
            display: block;
        }

        .cam-placeholder {
            position: absolute;
            color: #64748b;
            font-size: 0.95rem;
            font-weight: 500;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 8px;
        }

        .no-streams-card {
            grid-column: 1 / -1;
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 16px;
            padding: 48px 24px;
            text-align: center;
            color: #64748b;
            font-size: 1rem;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 16px;
        }

        /* Fullscreen Theater Mode Overrides */
        .cam-card:fullscreen, .cam-card:-webkit-full-screen {
            width: 100vw !important;
            height: 100vh !important;
            max-width: 100vw !important;
            max-height: 100vh !important;
            border-radius: 0 !important;
            background: #000000 !important;
            border: none !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: flex-start !important;
            align-items: stretch !important;
            padding: 0 !important;
            margin: 0 !important;
            box-shadow: none !important;
        }

        .cam-card:fullscreen .cam-card-bar, .cam-card:-webkit-full-screen .cam-card-bar {
            background: rgba(15, 23, 42, 0.95) !important;
            backdrop-filter: blur(10px) !important;
            padding: 12px 24px !important;
            flex-shrink: 0 !important;
            border-bottom: 1px solid #334155 !important;
        }

        .cam-card:fullscreen .cam-frame-box, .cam-card:-webkit-full-screen .cam-frame-box {
            flex: 1 1 auto !important;
            width: 100vw !important;
            height: calc(100vh - 54px) !important;
            min-height: unset !important;
            background: #000000 !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            overflow: hidden !important;
        }

        .cam-card:fullscreen .cam-frame, .cam-card:-webkit-full-screen .cam-frame {
            width: 100% !important;
            height: 100% !important;
            max-width: 100vw !important;
            max-height: calc(100vh - 54px) !important;
            object-fit: contain !important;
            display: block !important;
        }

        /* Modal Overlay */
        .modal-overlay {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(11, 15, 25, 0.85);
            backdrop-filter: blur(8px);
            z-index: 1000;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .modal-card {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 16px;
            width: 100%;
            max-width: 520px;
            padding: 24px;
            box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .modal-header h3 {
            font-size: 1.25rem;
            font-weight: 700;
            color: #f8fafc;
        }
        .form-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        .form-group label {
            font-size: 0.8rem;
            font-weight: 600;
            color: #94a3b8;
            text-transform: uppercase;
        }
        .form-group input {
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 10px 14px;
            color: #f8fafc;
            font-size: 0.9rem;
            outline: none;
        }
        .modal-footer {
            display: flex;
            justify-content: flex-end;
            gap: 12px;
            margin-top: 8px;
        }
    </style>

    <div class="header">
        <div>
            <h1>🎥 Live Camera Stream Monitor</h1>
            <p id="subTitleText">Detecting active camera streams from database...</p>
        </div>
        <div class="header-actions">
            <div class="stream-counter-badge" id="streamCounter">Active Streams: 0 / 4</div>
            <button class="btn-add-stream" id="btnAddStream" onclick="openAddStreamModal()">
                <span>➕ Add Stream</span>
            </button>
        </div>
    </div>

    <!-- Dynamic Camera Grid Container -->
    <div class="dynamic-cam-grid" id="camGrid">
        <div class="no-streams-card">
            <span>📷 Querying saved RTSP camera feeds...</span>
        </div>
    </div>

    <!-- Add Stream Modal -->
    <div class="modal-overlay" id="addStreamModal">
        <div class="modal-card">
            <div class="modal-header">
                <h3>📷 Register New RTSP Camera</h3>
                <button onclick="closeAddStreamModal()" style="background:none; border:none; color:#94a3b8; font-size:1.4rem; cursor:pointer;">&times;</button>
            </div>
            <div style="color: #94a3b8; font-size: 0.85rem;">
                Add a new RTSP camera feed (Maximum 4 camera streams allowed).
            </div>
            <div class="form-group">
                <label for="inputCamId">Camera ID</label>
                <input type="text" id="inputCamId" placeholder="e.g. cam_1" />
            </div>
            <div class="form-group">
                <label for="inputRtspUrl">RTSP Stream URL</label>
                <input type="text" id="inputRtspUrl" placeholder="e.g. rtsp://user:pass@192.168.1.50:554/stream1" />
            </div>
            <div class="form-group">
                <label for="inputCamName">Camera Name (Optional)</label>
                <input type="text" id="inputCamName" placeholder="e.g. Reception Desk Cam" />
            </div>
            <div class="modal-footer">
                <button onclick="closeAddStreamModal()" style="background:#334155; color:#cbd5e1; border:none; padding:10px 16px; border-radius:8px; font-weight:600; cursor:pointer;">Cancel</button>
                <button onclick="submitAddStream()" style="background:#0284c7; color:#fff; border:none; padding:10px 20px; border-radius:8px; font-weight:600; cursor:pointer;">Start Stream</button>
            </div>
        </div>
    </div>

    <script>
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const activeSockets = {};
        let activeCamIds = [];

        async function updateActiveStreams() {
            try {
                const res = await fetch('/api/streams');
                const data = await res.json();
                const streams = data.streams || [];
                const currentStreams = streams.map(s => s.camera_id);

                document.getElementById('streamCounter').textContent = `Active Streams: ${streams.length} / 4`;
                const btnAdd = document.getElementById('btnAddStream');
                if (streams.length >= 4) {
                    btnAdd.disabled = true;
                    btnAdd.title = "Maximum limit of 4 RTSP camera streams reached";
                } else {
                    btnAdd.disabled = false;
                    btnAdd.title = "";
                }

                if (JSON.stringify(currentStreams.sort()) === JSON.stringify(activeCamIds.sort())) {
                    return;
                }

                activeCamIds = currentStreams;
                renderDynamicGrid(streams);
            } catch (e) {
                console.error("Error fetching active streams", e);
            }
        }

        function renderDynamicGrid(streams) {
            const grid = document.getElementById('camGrid');
            const subTitle = document.getElementById('subTitleText');
            const streamIds = streams.map(s => s.camera_id);

            // 1. Close sockets and remove DOM cards for deleted streams ONLY
            Object.keys(activeSockets).forEach(id => {
                if (!streamIds.includes(id)) {
                    if (activeSockets[id]) {
                        activeSockets[id].close();
                        delete activeSockets[id];
                    }
                    const card = document.getElementById(`card_${id}`);
                    if (card) card.remove();
                }
            });

            // Remove empty placeholder card if streams exist
            const noStreamsCard = grid.querySelector('.no-streams-card');
            if (streams.length > 0 && noStreamsCard) {
                noStreamsCard.remove();
            }

            if (!streams || streams.length === 0) {
                subTitle.textContent = "0 Active Camera Streams";
                grid.className = "dynamic-cam-grid";
                if (!grid.querySelector('.no-streams-card')) {
                    grid.innerHTML = `
                        <div class="no-streams-card">
                            <span style="font-size: 2.2rem;">📷</span>
                            <span style="color: #f8fafc; font-weight: 700; font-size: 1.1rem;">No active camera streams configured</span>
                            <span style="max-width: 420px; font-size: 0.9rem;">Click "➕ Add Stream" above to register an RTSP camera feed (Up to 4 streams max).</span>
                            <button class="btn-add-stream" onclick="openAddStreamModal()">➕ Add Camera Stream</button>
                        </div>
                    `;
                }
                return;
            }

            subTitle.textContent = `Streaming ${streams.length} active camera feed(s) from database`;
            grid.className = streams.length === 1 ? "dynamic-cam-grid single-stream" : "dynamic-cam-grid";

            // 2. Dynamically append ONLY NEW camera cards without destroying existing DOM cards or WebSockets
            streams.forEach(s => {
                const camId = s.camera_id;
                const name = s.name || `Camera ${camId}`;
                let card = document.getElementById(`card_${camId}`);

                if (!card) {
                    const cardHtml = `
                        <div class="cam-card" id="card_${camId}">
                            <div class="cam-card-bar">
                                <div class="cam-bar-left">
                                    <div class="cam-label">📷 ${escapeHtml(name)} (${camId})</div>
                                    <div class="status-badge">
                                        <div class="dot" id="dot_${camId}"></div>
                                        <span id="status_${camId}">CONNECTING</span>
                                    </div>
                                </div>
                                <div class="cam-bar-right">
                                    <button class="btn-action-sm" onclick="toggleFullscreen('card_${camId}')">⛶ Fullscreen</button>
                                    <button class="btn-action-danger" onclick="deleteStream('${escapeHtml(camId)}')">🗑️ Delete</button>
                                </div>
                            </div>
                            <div class="cam-frame-box">
                                <div class="cam-placeholder" id="ph_${camId}">
                                    <span>🎥 Camera ${camId} Offline / Connecting...</span>
                                </div>
                                <img class="cam-frame" id="frame_${camId}" src="" alt="${camId} Feed" />
                            </div>
                        </div>
                    `;
                    grid.insertAdjacentHTML('beforeend', cardHtml);
                    connectCameraSocket(camId);
                }
            });
        }

        function connectCameraSocket(camId) {
            if (activeSockets[camId] && activeSockets[camId].readyState === WebSocket.OPEN) {
                return;
            }

            const wsUrl = `${wsProtocol}//${window.location.host}/face/ws?camera_id=${camId}`;
            const imgEl = document.getElementById(`frame_${camId}`);
            const dotEl = document.getElementById(`dot_${camId}`);
            const statusEl = document.getElementById(`status_${camId}`);
            const placeholderEl = document.getElementById(`ph_${camId}`);

            const ws = new WebSocket(wsUrl);
            activeSockets[camId] = ws;

            ws.onopen = () => {
                if (dotEl) dotEl.classList.add('connected');
                if (statusEl) statusEl.textContent = 'LIVE STREAM';
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'frame' && data.frame && imgEl) {
                        if (placeholderEl) placeholderEl.style.display = 'none';
                        imgEl.src = 'data:image/jpeg;base64,' + data.frame;
                    }
                } catch (e) {
                    console.error(`Error parsing frame for ${camId}`, e);
                }
            };

            ws.onclose = () => {
                if (dotEl) dotEl.classList.remove('connected');
                if (statusEl) statusEl.textContent = 'OFFLINE';
                if (placeholderEl) placeholderEl.style.display = 'flex';
            };

            ws.onerror = (err) => {
                console.error(`WebSocket error on ${camId}:`, err);
                ws.close();
            };
        }

        function openAddStreamModal() {
            document.getElementById('addStreamModal').style.display = 'flex';
        }

        function closeAddStreamModal() {
            document.getElementById('addStreamModal').style.display = 'none';
            document.getElementById('inputCamId').value = '';
            document.getElementById('inputRtspUrl').value = '';
            document.getElementById('inputCamName').value = '';
        }

        async function submitAddStream() {
            const camId = document.getElementById('inputCamId').value.trim();
            const url = document.getElementById('inputRtspUrl').value.trim();
            const name = document.getElementById('inputCamName').value.trim();

            if (!camId || !url) {
                alert("Please enter both Camera ID and RTSP Stream URL.");
                return;
            }

            try {
                const res = await fetch('/api/streams/add', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ camera_id: camId, rtsp_url: url, name: name })
                });

                const data = await res.json();
                if (res.ok && data.status === 'success') {
                    closeAddStreamModal();
                    updateActiveStreams();
                } else {
                    alert(data.detail || "Failed to add stream.");
                }
            } catch (e) {
                alert("Error connecting to server to add stream.");
            }
        }

        async function deleteStream(camId) {
            if (!confirm(`Are you sure you want to stop and delete camera stream '${camId}'?`)) {
                return;
            }

            try {
                const res = await fetch(`/api/streams/${encodeURIComponent(camId)}`, {
                    method: 'DELETE'
                });
                const data = await res.json();
                if (res.ok) {
                    updateActiveStreams();
                } else {
                    alert(data.detail || "Failed to delete stream.");
                }
            } catch (e) {
                alert("Error connecting to server to delete stream.");
            }
        }

        function toggleFullscreen(cardId) {
            const card = document.getElementById(cardId);
            if (!card) return;

            if (!document.fullscreenElement && !document.webkitFullscreenElement) {
                if (card.requestFullscreen) {
                    card.requestFullscreen();
                } else if (card.webkitRequestFullscreen) {
                    card.webkitRequestFullscreen();
                }
            } else {
                if (document.exitFullscreen) {
                    document.exitFullscreen();
                } else if (document.webkitExitFullscreen) {
                    document.webkitExitFullscreen();
                }
            }
        }

        function escapeHtml(str) {
            return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
        }

        updateActiveStreams();
        setInterval(updateActiveStreams, 8000);
    </script>
    """
    html_page = render_layout("Dynamic Live Stream Monitor", "stream", content)
    return HTMLResponse(content=html_page)
