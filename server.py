"""
NeuroVision — Sprint 6: Mobile App Server
───────────────────────────────────────────
Flask server that:
  - Streams live camera feed to phone
  - Sends detection results in real time
  - Accepts voice commands from phone
  - Shows danger alerts on phone
  - Displays navigation instructions
Run: python server.py
Then open http://your-mac-ip:5000 on your phone
"""

import cv2
import time
import json
import threading
import base64
import numpy as np
from flask import Flask, render_template_string, Response, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS

app     = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*",
                    async_mode="threading")

# ── Shared state ──────────────────────────────────────────────
state = {
    "frame":       None,
    "detections":  [],
    "danger":      [],
    "depth":       {},
    "room":        "unknown",
    "auto_mode":   False,
    "voice_on":    True,
    "fps":         0.0,
    "last_update": 0,
}
state_lock = threading.Lock()


def update_state(**kwargs):
    with state_lock:
        state.update(kwargs)
        state["last_update"] = time.time()


def get_state():
    with state_lock:
        return state.copy()


# ── Frame streaming ───────────────────────────────────────────

def generate_frames():
    while True:
        with state_lock:
            frame = state["frame"]
        if frame is not None:
            _, buffer = cv2.imencode(
                ".jpg", frame,
                [cv2.IMWRITE_JPEG_QUALITY, 70]
            )
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )
        time.sleep(0.05)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/status")
def api_status():
    s = get_state()
    return jsonify({
        "detections":  s["detections"],
        "danger":      s["danger"],
        "depth":       s["depth"],
        "room":        s["room"],
        "auto_mode":   s["auto_mode"],
        "voice_on":    s["voice_on"],
        "fps":         round(s["fps"], 1),
        "last_update": s["last_update"],
    })


@app.route("/api/command/<cmd>")
def api_command(cmd):
    """Accepts commands from phone."""
    commands = {
        "auto_on":    lambda: update_state(auto_mode=True),
        "auto_off":   lambda: update_state(auto_mode=False),
        "voice_on":   lambda: update_state(voice_on=True),
        "voice_off":  lambda: update_state(voice_on=False),
        "describe":   lambda: socketio.emit("command", {"cmd": "describe"}),
        "read":       lambda: socketio.emit("command", {"cmd": "read"}),
        "who":        lambda: socketio.emit("command", {"cmd": "who"}),
        "help":       lambda: socketio.emit("command", {"cmd": "help"}),
    }
    if cmd in commands:
        commands[cmd]()
        return jsonify({"status": "ok", "command": cmd})
    return jsonify({"status": "error", "message": "Unknown command"})


# ── Main mobile UI ────────────────────────────────────────────

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,
      initial-scale=1.0, maximum-scale=1.0">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="theme-color" content="#060D1F">
<title>NeuroVision</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }

  body {
    background: #060D1F;
    color: #C8D8F0;
    font-family: 'Segoe UI', sans-serif;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  /* Header */
  .header {
    background: #0D1637;
    padding: 12px 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid #1A2540;
  }
  .logo {
    font-size: 1.2rem;
    font-weight: 800;
    color: #00D4E8;
    letter-spacing: 1px;
  }
  .fps-badge {
    background: #0B2040;
    border: 1px solid #00D4E8;
    color: #00D4E8;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
  }

  /* Camera feed */
  .camera-wrap {
    position: relative;
    width: 100%;
    background: #000;
  }
  .camera-wrap img {
    width: 100%;
    display: block;
  }
  .room-badge {
    position: absolute;
    top: 10px;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(13,22,55,0.85);
    border: 1px solid #00D4E8;
    color: #00D4E8;
    padding: 4px 14px;
    border-radius: 14px;
    font-size: 0.75rem;
    font-weight: 600;
    white-space: nowrap;
  }

  /* Danger banner */
  .danger-banner {
    display: none;
    background: #CC0000;
    color: white;
    text-align: center;
    padding: 10px;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 1px;
    animation: pulse 0.5s infinite alternate;
  }
  @keyframes pulse {
    from { opacity: 1; }
    to   { opacity: 0.7; }
  }

  /* Status cards */
  .status-row {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 8px;
    padding: 10px;
  }
  .stat-card {
    background: #0D1637;
    border: 1px solid #1A2540;
    border-radius: 10px;
    padding: 10px 8px;
    text-align: center;
  }
  .stat-card .num {
    font-size: 1.6rem;
    font-weight: 800;
    color: #00D4E8;
  }
  .stat-card .lbl {
    font-size: 0.65rem;
    color: #4A6080;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 2px;
  }

  /* Detection list */
  .section-title {
    padding: 6px 12px;
    font-size: 0.7rem;
    color: #4A6080;
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid #1A2540;
  }
  .det-list {
    padding: 8px 10px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    max-height: 140px;
    overflow-y: auto;
  }
  .det-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #0D1637;
    border-radius: 8px;
    padding: 7px 12px;
    border-left: 3px solid #00D4E8;
  }
  .det-item.danger  { border-left-color: #FF4444; }
  .det-item.warning { border-left-color: #FF8800; }
  .det-label {
    font-size: 0.85rem;
    font-weight: 600;
    color: #E0F0FF;
  }
  .det-meta {
    font-size: 0.72rem;
    color: #4A6080;
  }
  .det-dist {
    font-size: 0.78rem;
    color: #00D4E8;
    font-weight: 600;
  }

  /* Command buttons */
  .cmd-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    padding: 10px;
  }
  .cmd-btn {
    background: #0D1637;
    border: 1px solid #1A2540;
    border-radius: 12px;
    padding: 14px 10px;
    text-align: center;
    cursor: pointer;
    transition: all 0.15s;
    color: #C8D8F0;
    font-size: 0.85rem;
    font-weight: 600;
    -webkit-tap-highlight-color: transparent;
  }
  .cmd-btn:active {
    transform: scale(0.96);
    background: #1A2A50;
    border-color: #00D4E8;
  }
  .cmd-btn .icon { font-size: 1.4rem; display: block; margin-bottom: 4px; }
  .cmd-btn.active { border-color: #00C878; background: #0A2A1A; }
  .cmd-btn.danger-btn { border-color: #FF4444; }

  /* Bottom nav */
  .bottom-nav {
    margin-top: auto;
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    background: #0D1637;
    border-top: 1px solid #1A2540;
  }
  .nav-item {
    padding: 12px 8px;
    text-align: center;
    cursor: pointer;
    font-size: 0.65rem;
    color: #4A6080;
    -webkit-tap-highlight-color: transparent;
  }
  .nav-item.active { color: #00D4E8; }
  .nav-item .nav-icon { font-size: 1.3rem; display: block; }
</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <div class="logo">⬡ NEUROVISION</div>
  <div class="fps-badge" id="fps">-- FPS</div>
</div>

<!-- Camera -->
<div class="camera-wrap">
  <img src="/video_feed" alt="Camera Feed" id="cam">
  <div class="room-badge" id="room-badge">📍 Detecting room...</div>
</div>

<!-- Danger banner -->
<div class="danger-banner" id="danger-banner">
  ⚠️ DANGER DETECTED
</div>

<!-- Status row -->
<div class="status-row">
  <div class="stat-card">
    <div class="num" id="obj-count">0</div>
    <div class="lbl">Objects</div>
  </div>
  <div class="stat-card">
    <div class="num" id="danger-count">0</div>
    <div class="lbl">Alerts</div>
  </div>
  <div class="stat-card">
    <div class="num" id="auto-status">OFF</div>
    <div class="lbl">Auto Mode</div>
  </div>
</div>

<!-- Detections -->
<div class="section-title">LIVE DETECTIONS</div>
<div class="det-list" id="det-list">
  <div style="color:#4A6080;font-size:0.8rem;padding:8px">
    Waiting for detections...
  </div>
</div>

<!-- Commands -->
<div class="section-title">COMMANDS</div>
<div class="cmd-grid">
  <div class="cmd-btn" id="btn-auto" onclick="toggleAuto()">
    <span class="icon">🚗</span>Auto Mode
  </div>
  <div class="cmd-btn" onclick="sendCmd('describe')">
    <span class="icon">👁️</span>Describe Scene
  </div>
  <div class="cmd-btn" onclick="sendCmd('read')">
    <span class="icon">📖</span>Read Text
  </div>
  <div class="cmd-btn" onclick="sendCmd('who')">
    <span class="icon">👤</span>Who Is This
  </div>
  <div class="cmd-btn" onclick="sendCmd('voice_on')">
    <span class="icon">🔊</span>Voice On
  </div>
  <div class="cmd-btn" onclick="sendCmd('voice_off')">
    <span class="icon">🔇</span>Voice Off
  </div>
</div>

<!-- Bottom nav -->
<div class="bottom-nav">
  <div class="nav-item active">
    <span class="nav-icon">📷</span>Live
  </div>
  <div class="nav-item">
    <span class="nav-icon">🗺️</span>Map
  </div>
  <div class="nav-item">
    <span class="nav-icon">👤</span>Faces
  </div>
  <div class="nav-item">
    <span class="nav-icon">⚙️</span>Settings
  </div>
</div>

<script>
  let autoMode = false;

  // Poll status every 500ms
  async function pollStatus() {
    try {
      const res  = await fetch("/api/status");
      const data = await res.json();
      updateUI(data);
    } catch(e) {}
    setTimeout(pollStatus, 500);
  }

  function updateUI(data) {
    // FPS
    document.getElementById("fps").textContent =
      data.fps + " FPS";

    // Room
    document.getElementById("room-badge").textContent =
      "📍 " + (data.room || "unknown").replace("_", " ");

    // Counts
    document.getElementById("obj-count").textContent =
      data.detections.length;
    document.getElementById("danger-count").textContent =
      data.danger.length;

    // Auto mode
    autoMode = data.auto_mode;
    const autoEl = document.getElementById("auto-status");
    const autoBtn = document.getElementById("btn-auto");
    autoEl.textContent = autoMode ? "ON" : "OFF";
    autoEl.style.color = autoMode ? "#00C878" : "#00D4E8";
    autoBtn.classList.toggle("active", autoMode);

    // Danger banner
    const banner = document.getElementById("danger-banner");
    const hasDanger = data.danger.some(
      d => d.tier === "DANGER" || d.tier === "CRITICAL"
    );
    banner.style.display = hasDanger ? "block" : "none";

    // Detection list
    const list = document.getElementById("det-list");
    if (data.detections.length === 0) {
      list.innerHTML =
        '<div style="color:#4A6080;font-size:0.8rem;padding:8px">' +
        'No objects detected</div>';
    } else {
      list.innerHTML = data.detections.map((d, i) => {
        const danger = data.danger.find(
          e => e.zone === d.zone && e.label === d.label
        );
        const tierClass = danger ?
          (danger.tier === "DANGER" || danger.tier === "CRITICAL"
            ? "danger" : "warning") : "";
        const dist = data.depth[i] || "";
        return \`
          <div class="det-item \${tierClass}">
            <div>
              <div class="det-label">\${d.label}</div>
              <div class="det-meta">\${d.zone} · \${(d.confidence*100).toFixed(0)}%</div>
            </div>
            <div class="det-dist">\${dist}</div>
          </div>
        \`;
      }).join("");
    }
  }

  async function sendCmd(cmd) {
    await fetch(\`/api/command/\${cmd}\`);
    // Visual feedback
    event.currentTarget.style.borderColor = "#00D4E8";
    setTimeout(() =>
      event.currentTarget.style.borderColor = "", 300
    );
  }

  async function toggleAuto() {
    const cmd = autoMode ? "auto_off" : "auto_on";
    await fetch(\`/api/command/\${cmd}\`);
  }

  // Start polling
  pollStatus();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


# ── Server startup ────────────────────────────────────────────

def start_server(host="0.0.0.0", port=5000):
    """Start the mobile app server in background thread."""
    import socket
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "localhost"

    print(f"\n📱  NeuroVision Mobile App")
    print(f"   Local  : http://localhost:{port}")
    print(f"   Network: http://{local_ip}:{port}")
    print(f"   Open on your phone browser ↑\n")

    socketio.run(
        app,
        host=host,
        port=port,
        debug=False,
        use_reloader=False,
        log_output=False,
    )


def run_server_background(port=5000):
    """Runs server in background thread."""
    t = threading.Thread(
        target=start_server,
        kwargs={"port": port},
        daemon=True
    )
    t.start()
    return t
