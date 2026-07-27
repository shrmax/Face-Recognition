from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from views.layout import render_layout

router = APIRouter(tags=["Views"])


@router.get("/logs-ui", response_class=HTMLResponse)
@router.get("/logs", response_class=HTMLResponse)
async def get_logs_ui_page():
    content = """
    <style>
        .page-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
            margin-bottom: 4px;
        }
        .page-title h1 {
            font-size: 1.8rem;
            font-weight: 700;
            color: #f8fafc;
            letter-spacing: -0.5px;
        }
        .page-title p {
            color: #94a3b8;
            font-size: 0.9rem;
            margin-top: 4px;
        }

        /* Filter Control Bar */
        .filter-bar {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 16px;
            padding: 18px 24px;
            display: flex;
            gap: 16px;
            align-items: flex-end;
            flex-wrap: wrap;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
        }
        .filter-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
            flex: 1;
            min-width: 180px;
        }
        .filter-group label {
            font-size: 0.8rem;
            font-weight: 600;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .filter-group select, .filter-group input {
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 10px 14px;
            color: #f8fafc;
            font-size: 0.9rem;
            outline: none;
            transition: border-color 0.2s ease;
        }
        .filter-group select:focus, .filter-group input:focus {
            border-color: #38bdf8;
        }
        .btn-filter {
            background: #0284c7;
            color: #ffffff;
            border: none;
            border-radius: 10px;
            padding: 10px 20px;
            font-size: 0.9rem;
            font-weight: 600;
            cursor: pointer;
            height: 42px;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: background 0.2s ease;
        }
        .btn-filter:hover {
            background: #0369a1;
        }
        .btn-clear {
            background: #334155;
            color: #cbd5e1;
            border: none;
            border-radius: 10px;
            padding: 10px 16px;
            font-size: 0.9rem;
            font-weight: 600;
            cursor: pointer;
            height: 42px;
        }
        .btn-clear:hover {
            background: #475569;
        }

        /* Logs List Grid */
        .logs-container {
            display: flex;
            flex-direction: column;
            gap: 16px;
            margin-top: 8px;
        }
        .log-card {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 16px;
            padding: 16px 20px;
            display: flex;
            align-items: center;
            gap: 20px;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
            transition: border-color 0.2s ease, transform 0.2s ease;
        }
        .log-card:hover {
            border-color: #475569;
            transform: translateY(-1px);
        }

        /* Full Camera Widescreen Frame Box (240x135) */
        .full-frame-box {
            width: 240px;
            height: 135px;
            border-radius: 12px;
            overflow: hidden;
            background: #0b0f19;
            border: 1px solid #334155;
            flex-shrink: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            position: relative;
            transition: border-color 0.2s ease, transform 0.2s ease;
        }
        .full-frame-box:hover {
            border-color: #38bdf8;
            transform: scale(1.03);
        }

        .frame-img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            background: #0b0f19;
        }
        .no-crop-placeholder {
            color: #64748b;
            font-size: 0.75rem;
            text-align: center;
        }

        .log-main {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .log-header-row {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .log-person-name {
            font-size: 1.15rem;
            font-weight: 700;
            color: #f8fafc;
        }
        .log-conf-badge {
            background: rgba(16, 185, 129, 0.15);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .log-conf-badge.low {
            background: rgba(245, 158, 11, 0.15);
            color: #f59e0b;
            border-color: rgba(245, 158, 11, 0.3);
        }
        .log-meta-row {
            display: flex;
            gap: 20px;
            font-size: 0.85rem;
            color: #94a3b8;
            flex-wrap: wrap;
        }
        .log-meta-item {
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .bbox-tag {
            font-family: monospace;
            background: #0f172a;
            border: 1px solid #334155;
            color: #38bdf8;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.8rem;
        }

        /* Image View Modal */
        .image-modal-overlay {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(11, 15, 25, 0.9);
            backdrop-filter: blur(10px);
            z-index: 2000;
            align-items: center;
            justify-content: center;
            padding: 24px;
        }
        .image-modal-content {
            position: relative;
            max-width: 90vw;
            max-height: 90vh;
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7);
            display: flex;
            flex-direction: column;
        }
        .image-modal-header {
            background: #0f172a;
            padding: 12px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #334155;
            color: #f8fafc;
            font-weight: 700;
        }
        .image-modal-img {
            max-width: 100%;
            max-height: 80vh;
            object-fit: contain;
            background: #000000;
            display: block;
        }
    </style>

    <div class="page-header">
        <div class="page-title">
            <h1>📋 Detection Event Activity Logs</h1>
            <p>Full camera frame snapshots with bounding box overlay and identity telemetry.</p>
        </div>
    </div>

    <!-- Filter Control Bar -->
    <div class="filter-bar">
        <div class="filter-group">
            <label for="profileFilter">Person Profile</label>
            <select id="profileFilter">
                <option value="">All Persons</option>
            </select>
        </div>

        <div class="filter-group">
            <label for="dateFilter">Date Filter</label>
            <input type="date" id="dateFilter" />
        </div>

        <div class="filter-group">
            <label for="cameraFilter">Camera Feed</label>
            <select id="cameraFilter">
                <option value="">All Cameras</option>
                <option value="cam_1">cam_1</option>
                <option value="cam_2">cam_2</option>
            </select>
        </div>

        <button class="btn-filter" onclick="applyFilters()">
            <span>🔍 Search Logs</span>
        </button>

        <button class="btn-clear" onclick="clearFilters()">Clear</button>
    </div>

    <!-- Logs Container -->
    <div class="logs-container" id="logsContainer">
        <div style="color: #64748b; font-size: 0.95rem;">Loading activity logs...</div>
    </div>

    <!-- Image Modal -->
    <div class="image-modal-overlay" id="imageModal" onclick="closeImageModal()">
        <div class="image-modal-content" onclick="event.stopPropagation()">
            <div class="image-modal-header">
                <span id="imageModalTitle">Full Camera Frame Snapshot</span>
                <button onclick="closeImageModal()" style="background:none; border:none; color:#94a3b8; font-size:1.5rem; cursor:pointer;">&times;</button>
            </div>
            <img class="image-modal-img" id="modalImg" src="" alt="Full Camera Frame Snapshot" />
        </div>
    </div>

    <script>
        const urlParams = new URLSearchParams(window.location.search);
        const initProfileId = urlParams.get('profile_id') || '';
        const initDate = urlParams.get('date') || '';
        const initCameraId = urlParams.get('camera_id') || '';

        async function initPage() {
            if (initDate) {
                document.getElementById('dateFilter').value = initDate;
            }
            if (initCameraId) {
                document.getElementById('cameraFilter').value = initCameraId;
            }

            await loadProfileOptions();
            if (initProfileId) {
                document.getElementById('profileFilter').value = initProfileId;
            }

            fetchLogs();
        }

        async function loadProfileOptions() {
            try {
                const res = await fetch('/api/profiles');
                const data = await res.json();
                if (data.status === 'success') {
                    const select = document.getElementById('profileFilter');
                    const optionsHtml = ['<option value="">All Persons</option>']
                        .concat(data.profiles.map(p => `<option value="${p.profile_id}">${p.name} (${p.profile_id})</option>`))
                        .concat(['<option value="Unknown">Unknown Identity</option>']);
                    select.innerHTML = optionsHtml.join('');
                }
            } catch (e) {
                console.error("Failed to load profile filter options", e);
            }
        }

        async function fetchLogs() {
            const pid = document.getElementById('profileFilter').value.trim();
            const dt = document.getElementById('dateFilter').value.trim();
            const cam = document.getElementById('cameraFilter').value.trim();

            const params = new URLSearchParams();
            if (pid) params.append('profile_id', pid);
            if (dt) params.append('date', dt);
            if (cam) params.append('camera_id', cam);
            params.append('limit', '100');

            const container = document.getElementById('logsContainer');
            container.innerHTML = '<div style="color: #64748b; font-size: 0.95rem;">Fetching matching detection logs...</div>';

            try {
                const res = await fetch(`/api/logs?${params.toString()}`);
                const data = await res.json();

                if (data.status === 'success') {
                    renderLogs(data.logs);
                } else {
                    container.innerHTML = '<div style="color: #ef4444; font-size: 0.95rem;">Failed to fetch activity logs.</div>';
                }
            } catch (e) {
                container.innerHTML = '<div style="color: #ef4444; font-size: 0.95rem;">Server error while querying detection logs.</div>';
            }
        }

        function renderLogs(logs) {
            const container = document.getElementById('logsContainer');
            if (!logs || logs.length === 0) {
                container.innerHTML = '<div style="color: #64748b; font-size: 0.95rem; background: #1e293b; padding: 24px; border-radius: 16px; border: 1px solid #334155; text-align: center;">No matching detection event logs found. Try adjusting your filters.</div>';
                return;
            }

            container.innerHTML = logs.map(l => {
                const fullFrameUrl = l.full_frame_url || l.crop_url;
                const frameImgHtml = fullFrameUrl
                    ? `<img src="${fullFrameUrl}" class="frame-img" alt="${l.name} Full Frame" />`
                    : '<div class="no-crop-placeholder">No Snapshot</div>';

                const isLow = l.confidence < 0.60;
                const bboxStr = l.bbox && l.bbox.length === 4 ? `[${l.bbox.join(', ')}]` : 'N/A';

                return `
                    <div class="log-card">
                        <!-- Full Widescreen Camera Frame Snapshot -->
                        <div class="full-frame-box" title="Click to view full resolution camera frame" onclick="openImageModal('${fullFrameUrl}', '${escapeHtml(l.name)} - ${l.timestamp}')">
                            ${frameImgHtml}
                        </div>

                        <div class="log-main">
                            <div class="log-header-row">
                                <span class="log-person-name">${escapeHtml(l.name)}</span>
                                <span class="log-conf-badge ${isLow ? 'low' : ''}">${l.confidence_pct} Match</span>
                            </div>
                            <div class="log-meta-row">
                                <div class="log-meta-item">⏰ <strong>${l.timestamp}</strong></div>
                                <div class="log-meta-item">🎥 <strong>${l.camera_id}</strong> (Track #${l.track_id})</div>
                                <div class="log-meta-item">📍 BBox: <span class="bbox-tag">${bboxStr}</span></div>
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
        }

        function openImageModal(imgUrl, title) {
            document.getElementById('modalImg').src = imgUrl;
            document.getElementById('imageModalTitle').textContent = title || 'Full Camera Frame Snapshot';
            document.getElementById('imageModal').style.display = 'flex';
        }

        function closeImageModal() {
            document.getElementById('imageModal').style.display = 'none';
            document.getElementById('modalImg').src = '';
        }

        function applyFilters() {
            fetchLogs();
        }

        function clearFilters() {
            document.getElementById('profileFilter').value = '';
            document.getElementById('dateFilter').value = '';
            document.getElementById('cameraFilter').value = '';
            fetchLogs();
        }

        function escapeHtml(str) {
            return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
        }

        initPage();
    </script>
    """
    html_page = render_layout("Detection Activity Logs", "logs", content)
    return HTMLResponse(content=html_page)
