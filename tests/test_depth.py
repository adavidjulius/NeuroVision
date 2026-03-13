"""
NeuroVision — Phase 4 headless test
Runs detection + depth estimation on sample image.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from modules.detector import Detector
from modules.depth     import DepthEstimator
from modules.voice     import VoiceAlert

IMAGE_PATH   = "tests/samples/bus.jpg"
OUTPUT_PATH  = "tests/samples/result_phase4.jpg"
DEPTH_PATH   = "tests/samples/depth_map.jpg"

print("\n🧠  NeuroVision — Phase 4: Depth Estimation Test\n")

frame      = cv2.imread(IMAGE_PATH)
detector   = Detector()
depth_est  = DepthEstimator()
voice      = VoiceAlert()

# Detect objects
detections = detector.detect(frame)
print(f"\n📦  {len(detections)} objects detected\n")

# Estimate depth
depth_map  = depth_est.estimate(frame)

# Get distance label per detection
depth_labels = {}
print("📏  Distance estimates:")
for i, d in enumerate(detections):
    dist = depth_est.get_distance_label(depth_map, d["box"], d["area_ratio"])
    depth_labels[i] = dist
    print(f"   → {d['label']:15s}  zone={d['zone']:6s}  distance={dist}")

# Speak with distances
print("\n🔊  Voice alerts:")
voice.speak_detections(detections, depth_labels)

# Draw detections on frame
frame = detector.draw(frame, detections)

# Add distance labels to boxes
for i, d in enumerate(detections):
    x1, y1 = d["box"][0], d["box"][1]
    dist   = depth_labels.get(i, "")
    cv2.putText(frame, dist, (x1, y1 - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 2)

# Save depth map visualisation
if depth_map is not None:
    depth_color = depth_est.colorize(depth_map)
    cv2.imwrite(DEPTH_PATH, depth_color)
    print(f"\n🗺️   Depth map saved → {DEPTH_PATH}")

cv2.imwrite(OUTPUT_PATH, frame)
voice.save_log()

print(f"\n✅  Done!")
print(f"   Detections → {OUTPUT_PATH}")
print(f"   Depth map  → {DEPTH_PATH}\n")
