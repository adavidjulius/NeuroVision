"""
NeuroVision — Main Entry Point
──────────────────────────────
Runs the full NeuroVision pipeline.

Phase 1 : Camera feed ✅
Phase 2 : Object detection (YOLO) — coming next
Phase 3 : Voice alerts            — coming next
Phase 4 : Depth estimation        — coming next
Phase 5 : Face recognition + OCR  — coming next

Usage:
    python main.py

Controls:
    q → quit
    v → toggle voice
    d → toggle depth map
    r → OCR on current frame
"""

import cv2
import time
import sys
from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FPS_TARGET,
    SHOW_FPS, KEY_QUIT, KEY_TOGGLE_VOICE, KEY_TOGGLE_DEPTH, KEY_OCR,
    BOX_COLOR, TEXT_COLOR, FONT_SCALE, FONT_THICKNESS,
)


def draw_hud(frame, fps: float, voice_on: bool, phase: str = "1 — Camera"):
    """Draws the NeuroVision HUD overlay onto the frame."""
    h, w = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 44), (10, 18, 40), -1)
    cv2.putText(frame, "NEUROVISION", (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 229, 255), 2)
    cv2.putText(frame, f"Phase {phase}", (w - 260, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 200), 1)

    if SHOW_FPS:
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 110, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 100), 1)

    # Bottom bar
    cv2.rectangle(frame, (0, h - 32), (w, h), (10, 18, 40), -1)
    voice_label = "VOICE: ON " if voice_on else "VOICE: OFF"
    voice_color = (0, 229, 100) if voice_on else (100, 100, 100)
    cv2.putText(frame, f"q=quit  v={voice_label}  d=depth  r=OCR", (14, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 120, 160), 1)
    cv2.putText(frame, voice_label, (w - 130, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, voice_color, 1)

    # Zone lines (left / center / right)
    third = w // 3
    cv2.line(frame, (third, 44), (third, h - 32), (30, 50, 80), 1)
    cv2.line(frame, (third * 2, 44), (third * 2, h - 32), (30, 50, 80), 1)
    cv2.putText(frame, "LEFT",   (14,       70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 80, 120), 1)
    cv2.putText(frame, "CENTER", (third+14, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 80, 120), 1)
    cv2.putText(frame, "RIGHT",  (third*2+14, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 80, 120), 1)


def main():
    print("\n🧠  NeuroVision starting …")
    print(f"   Camera index : {CAMERA_INDEX}")
    print("   Press  q  to quit\n")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)

    if not cap.isOpened():
        print(f"❌  Cannot open camera {CAMERA_INDEX}.")
        print("   Run  python tests/test_camera.py  to find the right index.\n")
        sys.exit(1)

    # ── State flags ───────────────────────────────────────────
    voice_enabled = True
    depth_enabled = False

    # ── FPS tracking ──────────────────────────────────────────
    prev_time  = time.time()
    fps        = 0.0
    frame_idx  = 0

    print("✅  Camera opened. NeuroVision running.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️   Frame dropped")
            continue

        frame_idx += 1

        # ── FPS calc ──────────────────────────────────────────
        now      = time.time()
        fps      = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        # ── Phase 2 will add: detector.detect(frame) ──────────
        # ── Phase 3 will add: voice.speak(...)         ────────
        # ── Phase 4 will add: depth.estimate(frame)    ────────
        # ── Phase 5 will add: face_id.identify(frame)  ────────
        #                      ocr.read(frame)           ────────

        # ── HUD overlay ───────────────────────────────────────
        draw_hud(frame, fps, voice_enabled)

        cv2.imshow("NeuroVision", frame)

        # ── Key handling ──────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == KEY_QUIT:
            print("\n👋  Shutting down NeuroVision …")
            break
        elif key == KEY_TOGGLE_VOICE:
            voice_enabled = not voice_enabled
            state = "ON" if voice_enabled else "OFF"
            print(f"🔊  Voice alerts: {state}")
        elif key == KEY_TOGGLE_DEPTH:
            depth_enabled = not depth_enabled
            state = "ON" if depth_enabled else "OFF"
            print(f"📡  Depth map: {state}  (available in Phase 4)")
        elif key == KEY_OCR:
            print("📖  OCR triggered — available in Phase 5")

    cap.release()
    cv2.destroyAllWindows()
    print("✅  Done.\n")


if __name__ == "__main__":
    main()
