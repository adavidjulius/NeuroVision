"""
NeuroVision — Phase 3+4: Voice Alerts with Distance
"""

import time
import threading
from config import VOICE_ENABLED, VOICE_RATE, VOICE_VOLUME, ALERT_COOLDOWN


class VoiceAlert:
    def __init__(self):
        self.enabled   = VOICE_ENABLED
        self.cooldowns = {}
        self._lock     = threading.Lock()
        self.engine    = None
        self._log      = []
        self._init_engine()

    def _init_engine(self):
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate",   VOICE_RATE)
            self.engine.setProperty("volume", VOICE_VOLUME)
            print("✅  Voice engine ready (pyttsx3)")
        except Exception as e:
            print(f"⚠️   Voice engine unavailable ({e}) — log-only mode")
            self.engine = None

    def _cooldown_ok(self, key: str) -> bool:
        now  = time.time()
        last = self.cooldowns.get(key, 0)
        if now - last >= ALERT_COOLDOWN:
            self.cooldowns[key] = now
            return True
        return False

    def build_message(self, label: str, zone: str,
                      area_ratio: float, distance: str = None) -> str:
        """Builds natural language alert with optional distance."""
        if zone == "left":
            direction = "on your left"
        elif zone == "right":
            direction = "on your right"
        else:
            direction = "ahead"

        dist_str = f", {distance}" if distance else ""

        if label == "person" and area_ratio > 0.25:
            return f"Person approaching {direction}{dist_str}"
        if label in ("car", "truck", "bus", "motorcycle"):
            return f"Vehicle {direction}{dist_str}"
        if label == "stairs":
            return f"Stairs {direction}, caution"

        return f"{label.capitalize()} {direction}{dist_str}"

    def speak(self, text: str, key: str = None):
        if not self.enabled:
            return
        key = key or text
        if not self._cooldown_ok(key):
            return

        timestamp = time.strftime("%H:%M:%S")
        entry     = f"[{timestamp}]  🔊  {text}"
        self._log.append(entry)
        print(entry)

        if self.engine:
            def _run():
                with self._lock:
                    try:
                        self.engine.say(text)
                        self.engine.runAndWait()
                    except Exception as e:
                        print(f"⚠️   TTS error: {e}")
            threading.Thread(target=_run, daemon=True).start()

    def speak_detections(self, detections: list, depth_labels: dict = None):
        """
        detections  : list of detection dicts
        depth_labels: optional dict of index → distance string
        """
        if not detections:
            return

        sorted_d = sorted(detections, key=lambda d: d["area_ratio"], reverse=True)

        spoken = 0
        for i, d in enumerate(sorted_d):
            if spoken >= 2:
                break
            distance = (depth_labels or {}).get(i)
            msg = self.build_message(
                d["label"], d["zone"], d["area_ratio"], distance
            )
            key = f"{d['label']}_{d['zone']}"
            self.speak(msg, key=key)
            spoken += 1

    def save_log(self, path: str = "tests/samples/voice_log.txt"):
        with open(path, "w") as f:
            f.write("\n".join(self._log))
        print(f"📝  Voice log saved → {path}")

    def toggle(self):
        self.enabled = not self.enabled
        print(f"🔊  Voice: {'ON' if self.enabled else 'OFF'}")
        return self.enabled
