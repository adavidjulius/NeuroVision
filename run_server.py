"""
Run NeuroVision mobile server standalone.
Opens phone app without starting the camera.
Good for testing the UI.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server import start_server, update_state
import time, threading, random

# Simulate live data for UI testing
def simulate_data():
    labels = ["person","chair","laptop","bottle","car","door"]
    zones  = ["left","center","right"]
    rooms  = ["living_room","kitchen","corridor","office"]
    i      = 0
    while True:
        update_state(
            detections=[
                {"label":      random.choice(labels),
                 "zone":       random.choice(zones),
                 "confidence": random.uniform(0.6, 0.99),
                 "area_ratio": random.uniform(0.02, 0.35)},
            ],
            danger=[] if random.random() > 0.3 else [
                {"tier":    "WARNING",
                 "message": "Object approaching",
                 "zone":    "center"}
            ],
            depth={"0": f"~{random.uniform(0.5,5):.1f}m"},
            room=random.choice(rooms),
            fps=random.uniform(12, 22),
            auto_mode=False,
            voice_on=True,
        )
        i += 1
        time.sleep(1)

# Start simulation in background
t = threading.Thread(target=simulate_data, daemon=True)
t.start()

# Start server
start_server()
