"""
NeuroVision — Phase 3: Spatial Audio
──────────────────────────────────────
Plays directional beeps panned left/right using pygame.
Falls back to print-only mode in Codespaces.
"""

import threading

class SpatialAudio:
    def __init__(self):
        self.enabled = False
        self._init_pygame()

    def _init_pygame(self):
        try:
            import pygame
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self.pygame  = pygame
            self.enabled = True
            print("✅  Spatial audio ready (pygame)")
        except Exception as e:
            print(f"⚠️   Spatial audio unavailable ({e}) — log-only mode")
            self.pygame  = None
            self.enabled = False

    def _make_beep(self, frequency: int, duration_ms: int,
                   pan: float, volume: float):
        """
        pan: -1.0 = full left, 0.0 = center, 1.0 = full right
        """
        if not self.pygame:
            return

        import numpy as np
        sr       = 44100
        frames   = int(sr * duration_ms / 1000)
        t        = np.linspace(0, duration_ms / 1000, frames, False)
        wave     = (np.sin(2 * np.pi * frequency * t) * 32767 * volume).astype(np.int16)

        left_vol  = max(0.0, 1.0 - pan)   if pan > 0 else 1.0
        right_vol = max(0.0, 1.0 + pan)   if pan < 0 else 1.0

        stereo   = np.column_stack([
            (wave * left_vol).astype(np.int16),
            (wave * right_vol).astype(np.int16),
        ])
        sound    = self.pygame.sndarray.make_sound(stereo)
        sound.play()

    def play(self, zone: str, urgency: str = "normal"):
        """
        zone    : 'left' | 'center' | 'right'
        urgency : 'normal' | 'danger'
        """
        pan_map  = {"left": -0.85, "center": 0.0, "right": 0.85}
        freq_map = {"normal": 880, "danger": 440}
        dur_map  = {"normal": 120, "danger": 80}

        pan      = pan_map.get(zone, 0.0)
        freq     = freq_map.get(urgency, 880)
        duration = dur_map.get(urgency, 120)
        repeats  = 3 if urgency == "danger" else 1

        label = f"[AUDIO]  zone={zone:<6}  urgency={urgency}"
        print(label)

        if self.enabled:
            def _run():
                for _ in range(repeats):
                    self._make_beep(freq, duration, pan, 0.5)
                    if repeats > 1:
                        import time; time.sleep(0.1)
            threading.Thread(target=_run, daemon=True).start()
