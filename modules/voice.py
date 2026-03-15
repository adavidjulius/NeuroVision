"""
NeuroVision — Voice (Thread-safe, spam-free)
"""
import time
import subprocess
import threading
from config import VOICE_ENABLED, ALERT_COOLDOWN


class VoiceAlert:
    def __init__(self):
        self.enabled   = VOICE_ENABLED
        self._lock     = threading.Lock()
        self._cd_lock  = threading.Lock()
        self._cooldowns = {}
        self._speaking  = False
        self._log       = []
        print("✅  Voice engine ready (macOS say)")

    def _cooldown_ok(self, key: str,
                     cooldown: float = None) -> bool:
        with self._cd_lock:
            now  = time.time()
            cd   = cooldown or ALERT_COOLDOWN
            last = self._cooldowns.get(key, 0)
            if now - last >= cd:
                self._cooldowns[key] = now
                return True
            return False

    def reset_cooldown(self, key: str):
        with self._cd_lock:
            self._cooldowns.pop(key, None)

    def _say(self, text: str):
        with self._lock:
            self._speaking = True
            try:
                subprocess.run(
                    ["say", "-r", "185", text],
                    timeout=20,
                    capture_output=True
                )
            except Exception:
                pass
            finally:
                time.sleep(0.5)
                self._speaking = False

    def is_speaking(self) -> bool:
        return self._speaking

    def speak(self, text: str, key: str = None,
              force: bool = False,
              cooldown: float = None):
        if not self.enabled:
            return
        if self._speaking and not force:
            return
        key = key or text
        if not force and not self._cooldown_ok(key, cooldown):
            return
        ts    = time.strftime("%H:%M:%S")
        entry = f"[{ts}]  🔊  {text}"
        self._log.append(entry)
        print(entry)
        threading.Thread(
            target=self._say, args=(text,), daemon=True
        ).start()

    def speak_now(self, text: str):
        """Non-blocking speak for commands."""
        if not self.enabled:
            return
        print(f"🔊  {text}")
        threading.Thread(
            target=self._say, args=(text,), daemon=True
        ).start()
        # Wait for it to finish
        time.sleep(0.3)
        while self._speaking:
            time.sleep(0.1)

    def build_message(self, label: str, zone: str,
                      area_ratio: float,
                      distance: str = None) -> str:
        dir_map = {
            "left":   "on your left",
            "right":  "on your right",
            "center": "ahead",
        }
        direction = dir_map.get(zone, "ahead")
        dist_str  = f", {distance}" if distance else ""

        if label == "person" and area_ratio > 0.25:
            return f"Person approaching {direction}{dist_str}"
        if label in ("car","truck","bus","motorcycle"):
            return f"Vehicle {direction}{dist_str}"
        if label == "stairs":
            return f"Stairs {direction}, caution"
        return f"{label.capitalize()} {direction}{dist_str}"

    def speak_detections(self, detections: list,
                         depth_labels: dict = None):
        if not detections or self._speaking:
            return
        # Only speak the most important object
        top = max(detections, key=lambda d: d["area_ratio"])
        i   = detections.index(top)
        dist = (depth_labels or {}).get(i)
        msg  = self.build_message(
            top["label"], top["zone"],
            top["area_ratio"], dist
        )
        self.speak(msg,
                   key=f"{top['label']}_{top['zone']}",
                   cooldown=5.0)

    def toggle(self):
        self.enabled = not self.enabled
        state = "ON" if self.enabled else "OFF"
        print(f"🔊  Voice: {state}")
        return self.enabled

    def save_log(self, path="tests/samples/voice_log.txt"):
        with open(path, "w") as f:
            f.write("\n".join(self._log))
