"""
NeuroVision — Sprint 5: Indoor Navigation Test
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
from modules.detector   import Detector
from modules.indoor_nav import IndoorNavigator

print("\n🗺️   NeuroVision — Indoor Navigation Test\n")
print("=" * 55)

IMAGE_PATH  = "tests/samples/bus.jpg"
OUTPUT_PATH = "tests/samples/result_indoor_nav.jpg"

frame    = cv2.imread(IMAGE_PATH)
detector = Detector()
nav      = IndoorNavigator()

# Simulate detections
detections = detector.detect(frame)
h, w       = frame.shape[:2]

print(f"📦  Detected {len(detections)} objects")

# Simulate depth labels
depth_labels = {i: f"~{(i+1)*1.5:.1f}m"
                for i in range(len(detections))}

# Update navigator
instructions = nav.update(detections, depth_labels, frame.shape)

print(f"\n📍  Current room  : {nav.current_room}")
print(f"🗺️   Known rooms   : {nav.get_known_rooms()}")
print(f"\n🧭  Navigation instructions:")
for inst in instructions:
    print(f"   [{inst['type']:12s}]  {inst['message']}")

print(f"\n📢  Location description:")
print(f"   {nav.describe_current_location()}")

print(f"\n🗺️   Map description:")
print(f"   {nav.describe_map()}")

# Draw on frame
frame = detector.draw(frame, detections)
frame = nav.draw(frame, instructions)

cv2.imwrite(OUTPUT_PATH, frame)
print(f"\n✅  Result saved → {OUTPUT_PATH}\n")

# Test room signatures
print("🏠  Room signature test:")
test_scenes = [
    {"name": "Kitchen scene",
     "objects": ["microwave", "refrigerator", "bottle", "cup"]},
    {"name": "Living room",
     "objects": ["couch", "tv", "potted plant", "chair"]},
    {"name": "Bedroom",
     "objects": ["bed", "lamp", "clock", "laptop"]},
    {"name": "Office",
     "objects": ["laptop", "chair", "book", "keyboard"]},
]

for scene in test_scenes:
    fake_dets = [
        {"label": obj, "zone": "center",
         "area_ratio": 0.08, "cx": 320,
         "cy": 240, "box": (100,100,200,200),
         "confidence": 0.9}
        for obj in scene["objects"]
    ]
    room = nav.detect_room(fake_dets)
    print(f"   {scene['name']:20s} → detected as: {room}")

print("\n" + "=" * 55)
print("✅  Sprint 5 Indoor Navigation test complete!\n")
