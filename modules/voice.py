"""
NeuroVision — Phase 3: Voice Alerts
─────────────────────────────────────
Speaks detection alerts using pyttsx3 (offline TTS).
Falls back to print-only mode if audio hardware unavailable (Codespaces).
"""

import time
import threading
import sys

from config import (
    VOICE_ENABLED, VOICE_RATE, VOICE_VOLUME, ALERT_COOLDOWN
)


class VoiceAlert:
    def __init__(self):
        self.enabled    = VOICE_ENABLED
        self.cooldowns  = {}   # label → last spoken time
        self._lock      = threading.Lock()
        self.engine     = None
        self._log       = []   # stores all alerts (for headless testing)

        self._init_engine()

    def _init_engine(self):
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate",   VOICE_RATE)
            self.engine.setProperty("volume", VOICE_VOLUME)
            print("✅  Voice engine ready (pyttsx3)")
        except Exception as e:
            print(f"⚠️   Voice engine unavailable ({e}) — running in log-only mode")
            self.engine = None

    def _cooldown_ok(self, key: str) -> bool:
        """Returns True if enough time has passed since last alert for this key."""
        now  = time.time()
        last = self.cooldowns.get(key, 0)
        if now - last >= ALERT_COOLDOWN:
            self.cooldowns[key] = now
            return True
        return False

    def build_message(self, label: str, zone: str, area_ratio: float) -> str:
        """Builds a natural language alert message."""
        # Proximity word
        if area_ratio > 0.25:
            proximity = "very close"
        elif area_ratio > 0.10:
            proximity = "nearby"
        else:
            proximity = "detected"

        # Direction
        if zone == "left":
            direction = "on your left"
        elif zone == "right":
            direction = "on your right"
        else:
            direction = "ahead"

        # Special cases
        if label == "person" and area_ratio > 0.25:
            return f"Person approaching, {direction}"
        if label in ("car", "truck", "bus", "motorcycle"):
            return f"Vehicle {direction}, {proximity}"
        if label == "stairs":
            return f"Stairs {direction}, caution"

        return f"{label.capitalize()} {direction}, {proximity}"

    def speak(self, text: str, key: str = None):
        """
        Speak a message with cooldown.
        key = dedup key (usually label+zone). Defaults to text itself.
        """
        if not self.enabled:
            return

        key = key or text
        if not self._cooldown_ok(key):
            return

        # Log it always
        timestamp = time.strftime("%H:%M:%S")
        entry     = f"[{timestamp}]  🔊  {text}"
        self._log.append(entry)
        print(entry)

        # Speak if engine available
        if self.engine:
            def _run():
                with self._lock:
                    try:
                        self.engine.say(text)
                        self.engine.runAndWait()
                    except Exception as e:
                        print(f"⚠️   TTS error: {e}")
            threading.Thread(target=_run, daemon=True).start()

    def speak_detections(self, detections: list):
        """Process a list of detections and speak the most important ones."""
        if not detections:
            return

        # Sort by area_ratio descending — speak closest first
        sorted_d = sorted(detections, key=lambda d: d["area_ratio"], reverse=True)

        # Only speak top 2 per frame to avoid spam
        spoken = 0
        for d in sorted_d:
            if spoken >= 2:
                break
            msg = self.build_message(d["label"], d["zone"], d["area_ratio"])
            key = f"{d['label']}_{d['zone']}"
            self.speak(msg, key=key)
            spoken += 1

    def save_log(self, path: str = "tests/samples/voice_log.txt"):
        """Save all spoken alerts to a file (for headless testing)."""
        with open(path, "w") as f:
            f.write("\n".join(self._log))
        print(f"📝  Voice log saved → {path}")

    def toggle(self):
        self.enabled = not self.enabled
        state = "ON" if self.enabled else "OFF"
        print(f"🔊  Voice alerts: {state}")
        return self.enabled
