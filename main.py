"""
NeuroVision — Main Entry Point
──────────────────────────────
Phase 1 : Camera feed        ✅
Phase 2 : Object detection   ✅
Phase 3 : Voice alerts        — coming next
Phase 4 : Depth estimation    — coming next
Phase 5 : Face rec + OCR      — coming next

Controls:
    q → quit
    v → toggle voice (Phase 3)
    d → toggle depth map (Phase 4)
    r → OCR on current frame (Phase 5)
"""

import cv2
import time
import sys
from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FPS_TARGET,
    SHOW_FPS, KEY_QUIT, KEY_TOGGLE_VOICE, KEY_TOGGLE_DEPTH, KEY_OCR,
    BOX_COLOR,
)
from modules.detector import Detector


def draw_hud(frame, fps: float, voice_on: bool, detection_on: bool = True):
    h, w = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 44), (10, 18, 40), -1)
    cv2.putText(frame, "NEUROVISION", (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 229, 255), 2)

    phase_label = "Phase 2 — Detection"
    cv2.putText(frame, phase_label, (w - 290, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 200), 1)

    if SHOW_FPS:
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 110, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 100), 1)

    # Bottom bar
    cv2.rectangle(frame, (0, h - 32), (w, h), (10, 18, 40), -1)
    cv2.putText(frame, "q=quit  v=voice(Ph3)  d=depth(Ph4)  r=OCR(Ph5)",
                (14, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 120, 160), 1)

    # Zone dividers
    third = w // 3
    cv2.line(frame, (third,     44), (third,     h - 32), (30, 50, 80), 1)
    cv2.line(frame, (third * 2, 44), (third * 2, h - 32), (30, 50, 80), 1)
    cv2.putText(frame, "LEFT",   (14,          70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 80, 120), 1)
    cv2.putText(frame, "CENTER", (third + 14,  70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 80, 120), 1)
    cv2.putText(frame, "RIGHT",  (third*2 + 14,70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 80, 120), 1)


def main():
    print("\n🧠  NeuroVision — Phase 2: Object Detection")
    print(f"   Camera : index {CAMERA_INDEX}")
    print("   Press  q  to quit\n")

    # ── Init camera ───────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)

    if not cap.isOpened():
        print(f"❌  Cannot open camera {CAMERA_INDEX}.")
        print("   Run  python tests/test_camera.py  to diagnose.\n")
        sys.exit(1)

    # ── Init detector ─────────────────────────────────────────
    detector = Detector()

    # ── State ─────────────────────────────────────────────────
    voice_enabled = True
    prev_time     = time.time()
    fps           = 0.0

    print("\n✅  Running. Point your camera at objects!\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # FPS
        now        = time.time()
        fps        = 1.0 / max(now - prev_time, 1e-6)
        prev_time  = now

        # ── Phase 2: Detect ───────────────────────────────────
        detections = detector.detect(frame)
        frame      = detector.draw(frame, detections)

        # Print detections to terminal
        for d in detections:
            print(f"  → {d['label']:15s} {d['confidence']:.0%}  zone={d['zone']:6s}  size={d['area_ratio']:.1%}")

        # ── HUD ───────────────────────────────────────────────
        draw_hud(frame, fps, voice_enabled)

        cv2.imshow("NeuroVision", frame)

        # ── Keys ──────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == KEY_QUIT:
            print("\n👋  Shutting down …")
            break
        elif key == KEY_TOGGLE_VOICE:
            voice_enabled = not voice_enabled
            print(f"🔊  Voice: {'ON' if voice_enabled else 'OFF'} (Phase 3 will activate this)")
        elif key == KEY_TOGGLE_DEPTH:
            print("📡  Depth map: coming in Phase 4")
        elif key == KEY_OCR:
            print("📖  OCR: coming in Phase 5")

    cap.release()
    cv2.destroyAllWindows()
    print("✅  Done.\n")


if __name__ == "__main__":
    main()
