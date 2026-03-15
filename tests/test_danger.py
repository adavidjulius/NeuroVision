"""
NeuroVision — Sprint 2: Danger Alert Test
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
from modules.detector import Detector
from modules.danger   import DangerDetector

IMAGE_PATH  = "tests/samples/bus.jpg"
OUTPUT_PATH = "tests/samples/result_danger.jpg"

print("\n🚨  NeuroVision — Danger Alert Test\n")

frame    = cv2.imread(IMAGE_PATH)
detector = Detector()
danger   = DangerDetector()

detections = detector.detect(frame)
events     = danger.analyze(detections)

print(f"📦  {len(detections)} objects detected")
print(f"🚨  {len(events)} danger events:\n")

for e in events:
    print(f"   [{e['tier']:8s}]  {e['message']}")
    print(f"             speed={e['speed']:.3f}  zone={e['zone']}")

frame = detector.draw(frame, detections)
frame = danger.draw(frame, events)
cv2.imwrite(OUTPUT_PATH, frame)

print(f"\n✅  Result saved → {OUTPUT_PATH}")
print("   Open in file explorer to see danger overlays\n")
