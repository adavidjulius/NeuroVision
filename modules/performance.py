"""
NeuroVision — Performance (Crash-safe workers)
Workers auto-restart if they crash.
"""
import time
import threading
import numpy as np


class FrameBuffer:
    def __init__(self):
        self._frame = None
        self._lock  = threading.Lock()

    def write(self, frame):
        with self._lock:
            self._frame = frame.copy()

    def read(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None


class ResultBuffer:
    def __init__(self, default=None):
        self._result  = default
        self._lock    = threading.Lock()

    def write(self, result):
        with self._lock:
            self._result = result

    def read(self):
        with self._lock:
            return self._result


class AIWorker:
    def __init__(self, name, fn, frame_buf,
                 result_buf, interval=0.1):
        self.name        = name
        self.fn          = fn
        self.frame_buf   = frame_buf
        self.result_buf  = result_buf
        self.interval    = interval
        self.running     = False
        self._thread     = None
        self._errors     = 0
        self.MAX_ERRORS  = 10

    def start(self):
        self.running = True
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name=f"AI-{self.name}"
        )
        self._thread.start()
        print(f"⚡  Worker: {self.name} "
              f"({self.interval*1000:.0f}ms)")

    def stop(self):
        self.running = False

    def _loop(self):
        while self.running:
            start = time.time()
            try:
                frame = self.frame_buf.read()
                if frame is not None:
                    result = self.fn(frame)
                    self.result_buf.write(result)
                    self._errors = 0  # reset on success
            except Exception as e:
                self._errors += 1
                if self._errors <= 3:
                    print(f"⚠️   Worker {self.name} "
                          f"error #{self._errors}: {e}")
                if self._errors >= self.MAX_ERRORS:
                    print(f"❌  Worker {self.name} "
                          f"too many errors — restarting")
                    self._errors = 0
                    time.sleep(2)

            elapsed = time.time() - start
            sleep   = max(0, self.interval - elapsed)
            time.sleep(sleep)


class PerformanceMonitor:
    def __init__(self):
        self._times  = []
        self._last   = time.time()
        self.fps     = 0.0

    def tick(self):
        now = time.time()
        self._times.append(now - self._last)
        self._last = now
        if len(self._times) > 30:
            self._times.pop(0)
        avg      = sum(self._times) / len(self._times)
        self.fps = 1.0 / max(avg, 1e-6)
