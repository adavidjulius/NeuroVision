"""
NeuroVision — Phase 2: Object Detection
"""
import cv2
import numpy as np
from ultralytics import YOLO

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    YOLO_MODEL, CONFIDENCE_THRESH, NMS_IOU_THRESH,
    ALERT_CLASSES, LEFT_ZONE_RATIO, RIGHT_ZONE_RATIO,
    BOX_COLOR, DANGER_COLOR, FONT_SCALE, FONT_THICKNESS, SHOW_LABELS,
)


class Detector:
    def __init__(self):
        print(f"🔍  Loading YOLO model: {YOLO_MODEL} ...")
        self.model = YOLO(YOLO_MODEL)
        self.class_names = self.model.names
        print(f"✅  YOLO ready — {len(self.class_names)} classes loaded")

    def get_zone(self, cx: float, frame_width: int) -> str:
        ratio = cx / frame_width
        if ratio < LEFT_ZONE_RATIO:
            return "left"
        elif ratio > RIGHT_ZONE_RATIO:
            return "right"
        return "center"

    def detect(self, frame: np.ndarray) -> list:
        h, w = frame.shape[:2]
        results = self.model(
            frame,
            conf=CONFIDENCE_THRESH,
            iou=NMS_IOU_THRESH,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label  = self.class_names[cls_id]
            if label not in ALERT_CLASSES:
                continue

            conf        = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx          = (x1 + x2) // 2
            cy          = (y1 + y2) // 2
            area_ratio  = ((x2 - x1) * (y2 - y1)) / (w * h)
            zone        = self.get_zone(cx, w)

            detections.append({
                "label":      label,
                "confidence": conf,
                "box":        (x1, y1, x2, y2),
                "cx":         cx,
                "cy":         cy,
                "zone":       zone,
                "area_ratio": area_ratio,
            })

        return detections

    def draw(self, frame: np.ndarray, detections: list) -> np.ndarray:
        for d in detections:
            x1, y1, x2, y2 = d["box"]
            label      = d["label"]
            conf       = d["confidence"]
            zone       = d["zone"]
            area_ratio = d["area_ratio"]

            color = DANGER_COLOR if area_ratio > 0.20 else BOX_COLOR

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            corner, thick = 14, 3
            for px, py, dx, dy in [
                (x1, y1,  1,  1), (x2, y1, -1,  1),
                (x1, y2,  1, -1), (x2, y2, -1, -1),
            ]:
                cv2.line(frame, (px, py), (px + dx * corner, py), color, thick)
                cv2.line(frame, (px, py), (px, py + dy * corner), color, thick)

            if SHOW_LABELS:
                tag = f"{label}  {conf:.0%}  [{zone}]"
                (tw, th), _ = cv2.getTextSize(
                    tag, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS
                )
                cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
                cv2.putText(frame, tag, (x1 + 4, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), FONT_THICKNESS)

        cv2.putText(frame, f"Objects: {len(detections)}",
                    (20, frame.shape[0] - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, BOX_COLOR, 1)
        return frame
