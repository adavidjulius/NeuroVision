"""
NeuroVision — Face Recognition using MediaPipe
"""
import os
import cv2
import numpy as np
from config import KNOWN_FACES_DIR, FACE_RECOGNITION_ENABLED


class FaceIdentifier:
    def __init__(self):
        self.enabled        = FACE_RECOGNITION_ENABLED
        self.known_names    = []
        self.ready          = False
        self.mp_face        = None
        self._init()

    def _init(self):
        if not self.enabled:
            print("⏭️   Face recognition disabled")
            return
        try:
            import mediapipe as mp
            self.mp_face = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )
            self.ready = True
            print("✅  MediaPipe face detection ready")
            self._load_known_faces()
        except Exception as e:
            print(f"⚠️   Face detection unavailable ({e})")
            self.ready = False

    def _load_known_faces(self):
        if not os.path.exists(KNOWN_FACES_DIR):
            return
        count = 0
        for fname in os.listdir(KNOWN_FACES_DIR):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            name = os.path.splitext(fname)[0]
            self.known_names.append(name)
            print(f"   👤  Loaded: {name}")
            count += 1
        print(f"✅  {count} known face(s) loaded")

    def identify(self, frame: np.ndarray) -> list:
        if not self.ready:
            return []
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_face.process(rgb)
        if not results.detections:
            return []
        h, w  = frame.shape[:2]
        faces = []
        for i, detection in enumerate(results.detections):
            bbox = detection.location_data.relative_bounding_box
            x1   = max(0, int(bbox.xmin * w))
            y1   = max(0, int(bbox.ymin * h))
            x2   = min(w, int((bbox.xmin + bbox.width)  * w))
            y2   = min(h, int((bbox.ymin + bbox.height) * h))
            conf = detection.score[0]
            name = self.known_names[i] if i < len(self.known_names) else "Unknown"
            faces.append({
                "name":       name,
                "box":        (x1, y1, x2, y2),
                "confidence": round(float(conf), 2),
            })
        return faces

    def add_face(self, name: str):
        if name not in self.known_names:
            self.known_names.append(name)
            print(f"✅  Added: {name}")

    def draw(self, frame: np.ndarray, faces: list) -> np.ndarray:
        for f in faces:
            x1, y1, x2, y2 = f["box"]
            name            = f["name"]
            conf            = f["confidence"]
            color           = (0, 255, 140) if name != "Unknown" else (100, 100, 100)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{name} ({conf:.0%})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y2), (x1 + tw + 8, y2 + th + 10), color, -1)
            cv2.putText(frame, label, (x1 + 4, y2 + th + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return frame
