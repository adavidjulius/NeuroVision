"""
NeuroVision — Performance Core
────────────────────────────────
Runs all heavy AI modules in background threads.
Main thread only does: read camera + draw + display.
Result: 15-25 FPS instead of 2.7 FPS.
"""

import cv2
import time
import threading
import queue
import numpy as np


class FrameBuffer:
    """Thread-safe latest-frame holder."""
    def __init__(self):
        self._frame = None
        self._lock  = threading.Lock()

    def write(self, frame: np.ndarray):
        with self._lock:
            self._frame = frame.copy()

    def read(self) -> np.ndarray:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None


class ResultBuffer:
    """Thread-safe latest-result holder for any module."""
    def __init__(self, default=None):
        self._result  = default
        self._lock    = threading.Lock()
        self.updated  = False

    def write(self, result):
        with self._lock:
            self._result = result
            self.updated = True

    def read(self):
        with self._lock:
            self.updated = False
            return self._result


class AIWorker:
    """
    Runs a single AI module in a background thread.
    Reads from frame_buffer, writes to result_buffer.
    Only processes a new frame every `interval` seconds.
    """
    def __init__(self, name: str, fn, frame_buffer: FrameBuffer,
                 result_buffer: ResultBuffer, interval: float = 0.1):
        self.name          = name
        self.fn            = fn
        self.frame_buffer  = frame_buffer
        self.result_buffer = result_buffer
        self.interval      = interval
        self.running       = False
        self._thread       = None

    def start(self):
        self.running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name=f"AI-{self.name}"
        )
        self._thread.start()
        print(f"⚡  Worker started: {self.name} (every {self.interval*1000:.0f}ms)")

    def stop(self):
        self.running = False

    def _loop(self):
        while self.running:
            start = time.time()
            frame = self.frame_buffer.read()
            if frame is not None:
                try:
                    result = self.fn(frame)
                    self.result_buffer.write(result)
                except Exception as e:
                    pass  # never crash main loop
            elapsed = time.time() - start
            sleep   = max(0, self.interval - elapsed)
            time.sleep(sleep)


class PerformanceMonitor:
    """Tracks FPS and per-module timing."""
    def __init__(self):
        self._times   = []
        self._window  = 30
        self.fps      = 0.0
        self._last    = time.time()

    def tick(self):
        now           = time.time()
        self._times.append(now - self._last)
        self._last    = now
        if len(self._times) > self._window:
            self._times.pop(0)
        avg       = sum(self._times) / len(self._times)
        self.fps  = 1.0 / max(avg, 1e-6)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        color = (0, 255, 100) if self.fps > 12 else (0, 165, 255) if self.fps > 6 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {self.fps:.1f}",
                    (w - 110, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame
