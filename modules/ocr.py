"""
NeuroVision — OCR Text Reading using EasyOCR
"""
import cv2
import numpy as np
from config import OCR_ENABLED, OCR_LANGUAGES


class TextReader:
    def __init__(self):
        self.enabled = OCR_ENABLED
        self.reader  = None
        self.ready   = False
        self._init()

    def _init(self):
        if not self.enabled:
            print("⏭️   OCR disabled")
            return
        try:
            import easyocr
            print("📖  Loading EasyOCR ...")
            self.reader = easyocr.Reader(OCR_LANGUAGES, gpu=False, verbose=False)
            self.ready  = True
            print("✅  EasyOCR ready")
        except Exception as e:
            print(f"⚠️   EasyOCR unavailable ({e})")

    def read(self, frame: np.ndarray) -> list:
        if not self.ready:
            return []
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.reader.readtext(rgb)
        output  = []
        for (bbox, text, conf) in results:
            if conf < 0.4 or len(text.strip()) < 2:
                continue
            pts = np.array(bbox, dtype=np.int32)
            output.append({"text": text.strip(), "confidence": round(conf, 2), "box": pts})
        return output

    def build_message(self, texts: list) -> str:
        if not texts:
            return "No text detected"
        combined = ", ".join([t["text"] for t in texts[:4]])
        return f"Sign reads: {combined}"

    def draw(self, frame: np.ndarray, texts: list) -> np.ndarray:
        for t in texts:
            pts  = t["box"]
            text = t["text"]
            conf = t["confidence"]
            cv2.polylines(frame, [pts], isClosed=True, color=(255, 200, 0), thickness=2)
            x, y    = pts[0]
            label   = f"{text} ({conf:.0%})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x, y - th - 10), (x + tw + 8, y), (255, 200, 0), -1)
            cv2.putText(frame, label, (x + 4, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        return frame
