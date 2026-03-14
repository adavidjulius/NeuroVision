"""
NeuroVision — Voice Alerts (macOS fixed)
Uses subprocess to call macOS 'say' command directly.
No threading issues, speaks every single time.
"""
import time
import subprocess
import threading
from config import VOICE_ENABLED, ALERT_COOLDOWN


class VoiceAlert:
    def __init__(self):
        self.enabled   = VOICE_ENABLED
        self.cooldowns = {}
        self._lock     = threading.Lock()
        self._log      = []
        self._speaking = False
        print("✅  Voice engine ready (macOS say)")

    def _cooldown_ok(self, key: str) -> bool:
        now  = time.time()
        last = self.cooldowns.get(key, 0)
        if now - last >= ALERT_COOLDOWN:
            self.cooldowns[key] = now
            return True
        return False

    def _say(self, text: str):
        """Speak using macOS native 'say' command — never fails."""
        with self._lock:
            self._speaking = True
            try:
                subprocess.run(
                    ["say", "-r", "200", text],
                    timeout=10,
                    capture_output=True
                )
            except Exception as e:
                print(f"⚠️   say error: {e}")
            finally:
                self._speaking = False

    def speak(self, text: str, key: str = None, force: bool = False):
        """
        Speak text with cooldown.
        force=True bypasses cooldown (for commands).
        """
        if not self.enabled:
            return
        key = key or text
        if not force and not self._cooldown_ok(key):
            return

        timestamp = time.strftime("%H:%M:%S")
        entry     = f"[{timestamp}]  🔊  {text}"
        self._log.append(entry)
        print(entry)

        threading.Thread(target=self._say, args=(text,), daemon=True).start()

    def speak_now(self, text: str):
        """Speak immediately — always, no cooldown, for command responses."""
        if not self.enabled:
            return
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}]  🔊  {text}")
        threading.Thread(target=self._say, args=(text,), daemon=True).start()

    def build_message(self, label: str, zone: str,
                      area_ratio: float, distance: str = None) -> str:
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

    def speak_detections(self, detections: list, depth_labels: dict = None):
        if not detections:
            return
        sorted_d = sorted(detections, key=lambda d: d["area_ratio"], reverse=True)
        spoken   = 0
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
        state = "ON" if self.enabled else "OFF"
        print(f"🔊  Voice: {state}")
        self.speak_now(f"Voice {state}")
        return self.enabled
