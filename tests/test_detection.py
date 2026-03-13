"""
NeuroVision — Phase 2 headless test
Runs YOLO on a sample image and saves the result.
No camera or display needed.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
from modules.detector import Detector

IMAGE_PATH  = "tests/samples/bus.jpg"
OUTPUT_PATH = "tests/samples/result.jpg"

print("\n🧠  NeuroVision — Detection Test")
print(f"   Input  : {IMAGE_PATH}")
print(f"   Output : {OUTPUT_PATH}\n")

frame    = cv2.imread(IMAGE_PATH)
detector = Detector()

detections = detector.detect(frame)
frame      = detector.draw(frame, detections)

print(f"\n📦  Detected {len(detections)} objects:")
for d in detections:
    print(f"   → {d['label']:15s} {d['confidence']:.0%}  zone={d['zone']:6s}  size={d['area_ratio']:.1%}")

cv2.imwrite(OUTPUT_PATH, frame)
print(f"\n✅  Result saved → {OUTPUT_PATH}")
print("   Open it in the Codespaces file explorer to see the boxes!\n")
