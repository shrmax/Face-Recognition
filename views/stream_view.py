from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from views.layout import render_layout

router = APIRouter(tags=["Views"])


@router.get("/stream", response_class=HTMLResponse)
async def get_stream_page(camera_id: str = "cam_1"):
    content = f"""
    <style>
        .header {{
            text-align: center;
            margin-bottom: 8px;
        }}
        h1 {{
            font-size: 1.8rem;
            font-weight: 700;
            color: #f8fafc;
            letter-spacing: -0.5px;
        }}
        .subtitle {{
            color: #94a3b8;
            font-size: 0.95rem;
            margin-top: 4px;
        }}
        .main-layout {{
            display: flex;
            gap: 20px;
            width: 100%;
            flex-wrap: wrap;
        }}
        .video-card {{
            flex: 2;
            min-width: 600px;
            position: relative;
            background: #1e293b;
            border-radius: 16px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
            overflow: hidden;
            border: 1px solid #334155;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 480px;
        }}
        img#streamFrame {{
            width: 100%;
            height: auto;
            display: block;
            border-radius: 16px;
        }}
        .overlay-badge {{
            position: absolute;
            top: 16px;
            left: 16px;
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(8px);
            padding: 8px 16px;
            border-radius: 30px;
            font-size: 0.85rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background-color: #ef4444;
        }}
        .dot.connected {{
            background-color: #10b981;
            box-shadow: 0 0 10px #10b981;
        }}
        .metrics-panel {{
            flex: 1;
            min-width: 320px;
            background: #1e293b;
            border-radius: 16px;
            padding: 20px;
            border: 1px solid #334155;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}
        .stat-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }}
        .stat-box {{
            background: #0f172a;
            padding: 14px;
            border-radius: 12px;
            border: 1px solid #1e293b;
        }}
        .stat-title {{
            font-size: 0.75rem;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .stat-value {{
            font-size: 1.5rem;
            font-weight: 700;
            color: #f8fafc;
            margin-top: 4px;
        }}
        .tracks-container {{
            flex: 1;
            overflow-y: auto;
            max-height: 360px;
        }}
        .track-item {{
            background: #0f172a;
            padding: 10px 14px;
            border-radius: 8px;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.85rem;
            border: 1px solid #1e293b;
        }}
        .track-id {{ font-weight: 600; color: #38bdf8; }}
        .track-status {{ color: #10b981; font-weight: 500; }}
    </style>

    <div class="header">
        <h1>🎥 Real-Time CCTV Feed Monitor</h1>
        <div class="subtitle">Active Camera ID: <strong>{camera_id}</strong></div>
    </div>

    <div class="main-layout">
        <div class="video-card">
            <div class="overlay-badge">
                <div class="dot" id="statusDot"></div>
                <span id="statusText">CONNECTING</span>
            </div>
            <img id="streamFrame" src="" alt="Live RTSP Feed" />
        </div>

        <div class="metrics-panel">
            <h3 style="font-size: 1.1rem; color: #f8fafc;">Pipeline Telemetry</h3>
            <div class="stat-grid">
                <div class="stat-box">
                    <div class="stat-title">Processing FPS</div>
                    <div class="stat-value" id="fpsVal">0.0</div>
                </div>
                <div class="stat-box">
                    <div class="stat-title">Active Heads</div>
                    <div class="stat-value" id="headsVal">0</div>
                </div>
            </div>

            <h4 style="font-size: 0.9rem; color: #94a3b8; margin-top: 8px;">Active Head Identities</h4>
            <div class="tracks-container" id="tracksList">
                <div style="color: #64748b; font-size: 0.85rem;">Waiting for tracking & identity data...</div>
            </div>
        </div>
    </div>

    <script>
        const camera_id = "{camera_id}";
        const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${{wsProtocol}}//${{window.location.host}}/face/ws?camera_id=${{camera_id}}`;
        
        const imgEl = document.getElementById("streamFrame");
        const statusDot = document.getElementById("statusDot");
        const statusText = document.getElementById("statusText");
        const fpsVal = document.getElementById("fpsVal");
        const headsVal = document.getElementById("headsVal");
        const tracksList = document.getElementById("tracksList");

        function connect() {{
            const ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {{
                statusDot.classList.add("connected");
                statusText.textContent = "LIVE STREAM";
            }};

            ws.onmessage = (event) => {{
                try {{
                    const data = JSON.parse(event.data);
                    if (data.type === "frame") {{
                        if (data.frame) {{
                            imgEl.src = "data:image/jpeg;base64," + data.frame;
                        }}
                        if (data.fps !== undefined) {{
                            fpsVal.textContent = data.fps.toFixed(1);
                        }}
                        if (data.tracks) {{
                            headsVal.textContent = data.tracks.length;
                            if (data.tracks.length === 0) {{
                                tracksList.innerHTML = '<div style="color: #64748b; font-size: 0.85rem;">No heads detected in frame</div>';
                            }} else {{
                                tracksList.innerHTML = data.tracks.map(t => `
                                    <div class="track-item">
                                        <span class="track-id">${{t.label || ('Head #' + t.track_id)}}</span>
                                        <span class="track-status">${{t.status === 'resolved' ? 'LOCKED' : 'PENDING'}}</span>
                                    </div>
                                `).join('');
                            }}
                        }}
                    }}
                }} catch (e) {{
                    console.error("Error parsing socket payload", e);
                }}
            }};

            ws.onclose = () => {{
                statusDot.classList.remove("connected");
                statusText.textContent = "DISCONNECTED - RETRYING";
                setTimeout(connect, 2000);
            }};

            ws.onerror = (err) => {{
                console.error("WebSocket error:", err);
                ws.close();
            }};
        }}

        connect();
    </script>
    """
    html_page = render_layout("Live Feed Monitor", "stream", content)
    return HTMLResponse(content=html_page)
