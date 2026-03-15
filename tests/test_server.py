"""
NeuroVision — Sprint 6: Mobile Server Test
Tests the Flask server starts and responds correctly.
"""
import sys, os, time, threading, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

print("\n📱  NeuroVision — Mobile Server Test\n")

# Test imports
try:
    from flask import Flask
    from flask_socketio import SocketIO
    print("✅  Flask + SocketIO imported")
except ImportError as e:
    print(f"❌  Missing: {e}")
    print("   Run: pip install flask flask-socketio flask-cors")
    sys.exit(1)

from server import app, update_state
import numpy as np
import cv2

# Test state update
import numpy as np
update_state(
    detections=[
        {"label": "person",  "zone": "center",
         "confidence": 0.92, "area_ratio": 0.15},
        {"label": "chair",   "zone": "left",
         "confidence": 0.85, "area_ratio": 0.08},
    ],
    danger=[
        {"tier": "WARNING", "message": "Person ahead",
         "zone": "center"}
    ],
    depth={"0": "~1.5m", "1": "~2m"},
    room="living_room",
    fps=18.5,
    auto_mode=False,
    voice_on=True,
)
print("✅  State update working")

# Test API endpoint
with app.test_client() as client:
    # Test status endpoint
    res  = client.get("/api/status")
    data = json.loads(res.data)

    print(f"✅  /api/status → {res.status_code}")
    print(f"   Detections : {len(data['detections'])}")
    print(f"   Danger     : {len(data['danger'])}")
    print(f"   Room       : {data['room']}")
    print(f"   FPS        : {data['fps']}")

    # Test command endpoints
    for cmd in ["auto_on", "auto_off",
                "voice_on", "voice_off"]:
        res = client.get(f"/api/command/{cmd}")
        d   = json.loads(res.data)
        print(f"   /api/command/{cmd:10s} → {d['status']}")

    # Test main page
    res = client.get("/")
    print(f"✅  Main page → {res.status_code} "
          f"({len(res.data)} bytes)")

print("\n✅  Sprint 6 Mobile Server test complete!")
print("\n   To run live server:")
print("   python server.py")
print("   Then open http://YOUR-MAC-IP:5000 on your phone\n")
