"""
Shared MVC View Layout module providing consistent navigation, branding, dark-mode CSS variables, and layout wrapper.
"""

def render_layout(title: str, active_tab: str, content: str) -> str:
    stream_active = "active" if active_tab == "stream" else "inactive"
    profiles_active = "active" if active_tab == "profiles" else "inactive"
    logs_active = "active" if active_tab == "logs" else "inactive"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Real-Time CCTV AI System</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-main: #0b0f19;
            --bg-card: #1e293b;
            --bg-dark: #0f172a;
            --border-color: #334155;
            --border-hover: #475569;
            --accent-primary: #0284c7;
            --accent-hover: #0369a1;
            --accent-cyan: #38bdf8;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --text-dim: #64748b;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            background-color: var(--bg-main);
            color: var(--text-main);
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding-bottom: 50px;
        }}

        /* Navigation Bar */
        .app-navbar {{
            width: 100%;
            background: var(--bg-card);
            border-bottom: 1px solid var(--border-color);
            padding: 14px 32px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
            position: sticky;
            top: 0;
            z-index: 900;
            backdrop-filter: blur(12px);
        }}

        .nav-brand {{
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--accent-cyan);
            display: flex;
            align-items: center;
            gap: 12px;
            text-decoration: none;
            letter-spacing: -0.3px;
        }}

        .nav-brand-logo {{
            background: linear-gradient(135deg, #38bdf8 0%, #0284c7 100%);
            color: #ffffff;
            width: 36px;
            height: 36px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
            box-shadow: 0 4px 12px rgba(56, 189, 248, 0.3);
        }}

        .nav-items {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .nav-link {{
            padding: 9px 18px;
            border-radius: 10px;
            font-size: 0.9rem;
            font-weight: 600;
            text-decoration: none;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s ease;
        }}

        .nav-link.active {{
            background: var(--accent-primary);
            color: #ffffff;
            box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35);
        }}

        .nav-link.inactive {{
            background: var(--bg-dark);
            color: var(--text-muted);
            border: 1px solid var(--border-color);
        }}

        .nav-link.inactive:hover {{
            background: var(--border-color);
            color: var(--text-main);
            border-color: var(--border-hover);
        }}

        .nav-badge {{
            background: rgba(16, 185, 129, 0.15);
            color: var(--success);
            border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.78rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        .nav-badge-dot {{
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--success);
            box-shadow: 0 0 8px var(--success);
        }}

        /* Content Container */
        .app-body {{
            max-width: 1280px;
            width: 95%;
            margin-top: 24px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}

        /* Toast Container */
        #toastContainer {{
            position: fixed;
            bottom: 24px;
            right: 24px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            z-index: 9999;
        }}
    </style>
</head>
<body>
    <!-- Persistent Top Navigation Bar -->
    <nav class="app-navbar">
        <a href="/stream" class="nav-brand">
            <div class="nav-brand-logo">⚡</div>
            <span>AI CCTV Surveillance</span>
        </a>

        <div class="nav-items">
            <a href="/stream?camera_id=cam_1" class="nav-link {stream_active}">
                <span>🎥 Live Monitor</span>
            </a>
            <a href="/profiles-ui" class="nav-link {profiles_active}">
                <span>👤 Person Profiles</span>
            </a>
            <a href="/logs-ui" class="nav-link {logs_active}">
                <span>📋 Detection Activity Logs</span>
            </a>
            <a href="/docs" target="_blank" class="nav-link inactive">
                <span>📖 API Docs</span>
            </a>
            <div class="nav-badge">
                <div class="nav-badge-dot"></div>
                <span>SYSTEM ONLINE</span>
            </div>
        </div>
    </nav>

    <!-- Main View Content -->
    <main class="app-body">
        {content}
    </main>

    <div id="toastContainer"></div>
</body>
</html>"""
