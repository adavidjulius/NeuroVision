"""
NeuroVision — Sprint 3: Object Identification Test
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import urllib.request
from modules.detector  import Detector
from modules.ocr       import TextReader
from modules.object_id import ObjectIdentifier

print("\n💊  NeuroVision — Object Identification Test\n")
print("=" * 55)

# Download test images
samples = {
    "bus":    ("https://ultralytics.com/images/bus.jpg",    "tests/samples/bus.jpg"),
    "zidane": ("https://ultralytics.com/images/zidane.jpg", "tests/samples/zidane.jpg"),
}

os.makedirs("tests/samples", exist_ok=True)
for name, (url, path) in samples.items():
    if not os.path.exists(path):
        print(f"📥  Downloading {name}...")
        urllib.request.urlretrieve(url, path)

detector = Detector()
ocr      = TextReader()
obj_id   = ObjectIdentifier(ocr_module=ocr)

for name, (_, path) in samples.items():
    print(f"\n📸  Testing: {name}")
    print("-" * 40)

    frame      = cv2.imread(path)
    detections = detector.detect(frame)
    results    = obj_id.identify(frame, detections)

    print(f"   Detections : {len(detections)}")
    print(f"   ID results : {len(results)}")

    for r in results:
        print(f"\n   {r['icon']}  Category : {r['category']}")
        print(f"      Label    : {r['label']}")
        print(f"      Details  : {r.get('details', '')}")
        if r.get("warning"):
            print(f"      ⚠ WARNING : {r['warning']}")
        if r.get("voice"):
            print(f"      🔊 Voice  : {r['voice']}")

    # Save result image
    frame  = detector.draw(frame, detections)
    frame  = obj_id.draw(frame, results)
    output = f"tests/samples/result_object_id_{name}.jpg"
    cv2.imwrite(output, frame)
    print(f"\n   ✅  Saved → {output}")

print("\n" + "=" * 55)
print("✅  Sprint 3 Object ID test complete!\n")
