"""
NeuroVision — Sprint 2: Danger Alerts
───────────────────────────────────────
Detects:
  - Fast moving objects toward user
  - Vehicle approach speed
  - Person fall detection
  - Proximity danger zones (4 tiers)
"""

import cv2
import time
import numpy as np
from collections import defaultdict


# ── Danger tiers ──────────────────────────────────────────────
TIER = {
    "CLEAR":    {"label": "Clear",    "color": (0, 255, 100),  "voice": None},
    "CAUTION":  {"label": "Caution",  "color": (0, 229, 255),  "voice": "Caution"},
    "WARNING":  {"label": "Warning",  "color": (0, 165, 255),  "voice": "Warning"},
    "DANGER":   {"label": "Danger!",  "color": (0, 60,  255),  "voice": "Danger"},
    "CRITICAL": {"label": "CRITICAL", "color": (0, 0,   255),  "voice": "Stop! Critical danger"},
}

# Area ratio thresholds per tier
AREA_TIERS = [
    (0.40, "CRITICAL"),
    (0.25, "DANGER"),
    (0.12, "WARNING"),
    (0.05, "CAUTION"),
    (0.00, "CLEAR"),
]

# High-risk object classes
HIGH_RISK   = {"car", "truck", "bus", "motorcycle", "bicycle"}
MEDIUM_RISK = {"person", "dog", "chair", "dining table"}


class DangerDetector:
    def __init__(self):
        # Track object positions over time for motion estimation
        self._history     = defaultdict(list)  # label+zone → [(time, area), ...]
        self._fall_track  = defaultdict(list)  # track_id → [(time, cy), ...]
        self._last_alert  = {}                 # key → last alert time
        self.COOLDOWN     = 2.0                # seconds between same alert
        print("✅  Danger detector ready")

    # ── Proximity tier ────────────────────────────────────────

    def get_tier(self, label: str, area_ratio: float) -> str:
        """Returns danger tier based on object size + class."""
        # High risk objects trigger one tier higher
        boost = 1 if label in HIGH_RISK else 0

        for i, (threshold, tier_name) in enumerate(AREA_TIERS):
            if area_ratio >= threshold:
                # Boost: move up one tier for high-risk objects
                if boost and i > 0:
                    return AREA_TIERS[i - 1][1]
                return tier_name
        return "CLEAR"

    # ── Motion / approach speed ───────────────────────────────

    def estimate_approach_speed(self, label: str, zone: str,
                                 area_ratio: float) -> float:
        """
        Estimates how fast an object is approaching.
        Returns growth rate (area change per second).
        Positive = approaching, negative = moving away.
        """
        key = f"{label}_{zone}"
        now = time.time()
        self._history[key].append((now, area_ratio))

        # Keep only last 1 second of history
        self._history[key] = [
            (t, a) for t, a in self._history[key]
            if now - t <= 1.0
        ]

        if len(self._history[key]) < 2:
            return 0.0

        oldest_t, oldest_a = self._history[key][0]
        newest_t, newest_a = self._history[key][-1]
        dt = newest_t - oldest_t

        if dt < 0.05:
            return 0.0

        return (newest_a - oldest_a) / dt  # area change per second

    # ── Fall detection ────────────────────────────────────────

    def check_fall(self, detections: list) -> list:
        """
        Detects if a person is falling.
        Fall = person bounding box aspect ratio suddenly becomes wide
               (person goes from vertical to horizontal).
        Returns list of fall events.
        """
        falls = []
        now   = time.time()

        for d in detections:
            if d["label"] != "person":
                continue

            x1, y1, x2, y2 = d["box"]
            width           = x2 - x1
            height          = max(1, y2 - y1)
            aspect          = width / height  # >1.2 = horizontal = possible fall

            cx = d["cx"]
            cy = d["cy"]
            track_key = f"person_{d['zone']}"

            self._fall_track[track_key].append((now, cy, aspect))
            # Keep last 1.5 seconds
            self._fall_track[track_key] = [
                (t, c, a) for t, c, a in self._fall_track[track_key]
                if now - t <= 1.5
            ]

            if len(self._fall_track[track_key]) < 3:
                continue

            # Check: aspect ratio changed from <0.8 to >1.2 quickly
            aspects = [a for _, _, a in self._fall_track[track_key]]
            if aspects[0] < 0.85 and aspects[-1] > 1.15:
                falls.append({
                    "zone":  d["zone"],
                    "label": "person",
                    "box":   d["box"],
                })

        return falls

    # ── Main analysis ─────────────────────────────────────────

    def analyze(self, detections: list) -> list:
        """
        Analyzes all detections for danger.
        Returns list of danger events sorted by severity.
        """
        events  = []
        falls   = self.check_fall(detections)

        # Fall events — highest priority
        for fall in falls:
            events.append({
                "type":    "fall",
                "tier":    "CRITICAL",
                "label":   "Person falling",
                "zone":    fall["zone"],
                "box":     fall["box"],
                "message": "Warning! Person falling detected",
                "speed":   0.0,
            })

        for d in detections:
            label      = d["label"]
            zone       = d["zone"]
            area_ratio = d["area_ratio"]
            tier       = self.get_tier(label, area_ratio)
            speed      = self.estimate_approach_speed(label, zone, area_ratio)

            # Upgrade tier if approaching fast
            if speed > 0.15 and tier in ("CAUTION", "WARNING"):
                tier = "DANGER"
            elif speed > 0.30:
                tier = "CRITICAL"

            if tier == "CLEAR":
                continue

            # Build alert message
            direction = (
                "on your left"   if zone == "left"  else
                "on your right"  if zone == "right" else
                "ahead"
            )

            if label in HIGH_RISK:
                msg = f"{label.capitalize()} {direction}"
                if speed > 0.10:
                    msg += ", approaching fast"
            else:
                msg = f"{label.capitalize()} {direction}"

            if tier == "CRITICAL":
                msg = f"Stop! {msg}"
            elif tier == "DANGER":
                msg = f"Danger! {msg}"

            events.append({
                "type":    "proximity",
                "tier":    tier,
                "label":   label,
                "zone":    zone,
                "box":     d["box"],
                "message": msg,
                "speed":   speed,
            })

        # Sort by severity
        tier_order = {"CRITICAL": 0, "DANGER": 1, "WARNING": 2, "CAUTION": 3}
        events.sort(key=lambda e: tier_order.get(e["tier"], 9))

        return events

    # ── Cooldown check ────────────────────────────────────────

    def should_alert(self, key: str, tier: str) -> bool:
        """Rate-limits alerts to avoid spam."""
        now      = time.time()
        cooldown = {
            "CRITICAL": 1.0,
            "DANGER":   2.0,
            "WARNING":  3.0,
            "CAUTION":  5.0,
        }.get(tier, 3.0)

        last = self._last_alert.get(key, 0)
        if now - last >= cooldown:
            self._last_alert[key] = now
            return True
        return False

    # ── Draw ──────────────────────────────────────────────────

    def draw(self, frame: np.ndarray, events: list) -> np.ndarray:
        h, w = frame.shape[:2]

        for e in events:
            box   = e.get("box")
            tier  = e["tier"]
            info  = TIER[tier]
            color = info["color"]

            if box:
                x1, y1, x2, y2 = box

                # Thick danger border
                thickness = 4 if tier in ("CRITICAL", "DANGER") else 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                # Danger label
                tag = f"⚠ {tier}"
                (tw, th), _ = cv2.getTextSize(
                    tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(frame,
                              (x1, y1 - th - 10),
                              (x1 + tw + 8, y1),
                              color, -1)
                cv2.putText(frame, tag, (x1 + 4, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Critical full-frame flash border
        critical = [e for e in events if e["tier"] == "CRITICAL"]
        if critical:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
            cv2.putText(frame, "!! CRITICAL DANGER !!",
                        (w // 2 - 160, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        return frame
