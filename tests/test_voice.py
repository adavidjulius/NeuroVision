"""
NeuroVision — Phase 3 headless test
Runs detection on sample image + generates voice alert log.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
from modules.detector import Detector
from modules.voice     import VoiceAlert
from modules.audio_spatial import SpatialAudio

IMAGE_PATH  = "tests/samples/bus.jpg"
OUTPUT_PATH = "tests/samples/result_phase3.jpg"

print("\n🧠  NeuroVision — Phase 3: Voice Alert Test\n")

frame    = cv2.imread(IMAGE_PATH)
detector = Detector()
voice    = VoiceAlert()
audio    = SpatialAudio()

detections = detector.detect(frame)
frame      = detector.draw(frame, detections)

print(f"\n📦  {len(detections)} objects detected — generating alerts:\n")

voice.speak_detections(detections)

# Also trigger spatial audio per detection
for d in detections:
    urgency = "danger" if d["area_ratio"] > 0.20 else "normal"
    audio.play(d["zone"], urgency)

cv2.imwrite(OUTPUT_PATH, frame)
voice.save_log()

print(f"\n✅  Done!")
print(f"   Image  → {OUTPUT_PATH}")
print(f"   Log    → tests/samples/voice_log.txt\n")
