"""
NeuroVision — Sprint 4: Multi-language OCR Test
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from modules.multilang_ocr import MultiLangOCR

print("\n🌐  NeuroVision — Multi-language OCR Test\n")
print("=" * 55)

ocr = MultiLangOCR()

# ── Test 1: Create synthetic test images ──────────────────────
def make_text_image(lines: list, filename: str):
    """Creates a white image with text lines for testing."""
    img = np.ones((400, 800, 3), dtype=np.uint8) * 255
    y   = 60
    for line, color in lines:
        cv2.putText(img, line, (40, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        y += 70
    os.makedirs("tests/samples", exist_ok=True)
    cv2.imwrite(filename, img)
    return img

# English test
print("\n📝  TEST 1 — English Text")
print("-" * 40)
en_img = make_text_image([
    ("STOP",        (0, 0, 0)),
    ("EXIT RIGHT",  (0, 0, 200)),
    ("500mg Tablet",(0, 100, 0)),
], "tests/samples/test_english.jpg")

results = ocr.read(en_img, language="english")
print(f"   Found {len(results)} text items:")
for r in results:
    print(f"   {r['flag']} [{r['language']:7s}] \"{r['text']}\" ({r['confidence']:.0%})")

msg = ocr.build_message(results)
print(f"   🔊 Voice: {msg}")

# Save with overlays
en_out = ocr.draw(en_img.copy(), results)
cv2.imwrite("tests/samples/result_english_ocr.jpg", en_out)

# Multi-language test
print("\n🌐  TEST 2 — Multi-language Detection")
print("-" * 40)
multi_img = make_text_image([
    ("Hello World",   (0, 0, 0)),
    ("EXIT",          (200, 0, 0)),
    ("100 rupees",    (0, 150, 0)),
], "tests/samples/test_multi.jpg")

results2 = ocr.read(multi_img, language="multi")
print(f"   Found {len(results2)} text items:")
for r in results2:
    print(f"   {r['flag']} [{r['language']:7s}] \"{r['text']}\" ({r['confidence']:.0%})")

msg2 = ocr.build_message(results2)
print(f"   🔊 Voice: {msg2}")

multi_out = ocr.draw(multi_img.copy(), results2)
cv2.imwrite("tests/samples/result_multi_ocr.jpg", multi_out)

print("\n" + "=" * 55)
print("✅  Sprint 4 Multi-language OCR test complete!")
print("   results saved → tests/samples/\n")
