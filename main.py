"""
NeuroVision — Main Entry Point (Accuracy Fixed)
"""
import cv2
import time
import sys
import os

from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FPS_TARGET,
    SHOW_FPS, KEY_QUIT, KEY_TOGGLE_VOICE, KEY_TOGGLE_DEPTH, KEY_OCR,
    BOX_COLOR, SPATIAL_AUDIO_ENABLED,
)
from modules.detector      import Detector
from modules.depth         import DepthEstimator
from modules.voice         import VoiceAlert
from modules.audio_spatial import SpatialAudio
from modules.face_id       import FaceIdentifier
from modules.ocr           import TextReader
from modules.assistant     import VoiceAssistant, CONFIG_PATH


def draw_hud(frame, fps, voice_on, assistant_name):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 44), (10, 18, 40), -1)
    cv2.putText(frame, "NEUROVISION", (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 229, 255), 2)
    cv2.putText(frame, f"Hey {assistant_name}", (w - 220, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 100), 1)
    if SHOW_FPS:
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 110, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 100), 1)
    cv2.rectangle(frame, (0, h - 32), (w, h), (10, 18, 40), -1)
    cv2.putText(frame, "q=quit  v=voice  d=depth  r=OCR  s=snapshot",
                (14, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 120, 160), 1)
    third = w // 3
    cv2.line(frame, (third,     44), (third,     h - 32), (30, 50, 80), 1)
    cv2.line(frame, (third * 2, 44), (third * 2, h - 32), (30, 50, 80), 1)
    cv2.putText(frame, "LEFT",    (14,           70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 80, 120), 1)
    cv2.putText(frame, "CENTER",  (third + 14,   70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 80, 120), 1)
    cv2.putText(frame, "RIGHT",   (third*2 + 14, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 80, 120), 1)


def main():
    print("\n🧠  NeuroVision — Full System (Accuracy Mode)\n")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)

    if not cap.isOpened():
        print(f"❌  Cannot open camera {CAMERA_INDEX}")
        sys.exit(1)

    detector  = Detector()
    depth_est = DepthEstimator()
    voice     = VoiceAlert()
    audio     = SpatialAudio()
    face_id   = FaceIdentifier()
    ocr       = TextReader()
    assistant = VoiceAssistant(voice, detector, face_id, ocr, depth_est)

    if not os.path.exists(CONFIG_PATH):
        assistant.first_run_setup()

    current_frame = [None]
    assistant.start_listening(lambda: current_frame[0])

    depth_enabled = False
    prev_time     = time.time()
    fps           = 0.0
    snap_n        = 0

    # Only run face recognition every N frames (expensive)
    FACE_EVERY_N  = 15
    frame_count   = 0

    print(f"✅  Running. Say 'Hey {assistant.name}' for commands.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count   += 1
        current_frame[0] = frame.copy()

        now       = time.time()
        fps       = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        # ── Detection ─────────────────────────────────────────
        detections = detector.detect(frame)

        # ── Depth (every frame) ───────────────────────────────
        depth_map  = depth_est.estimate(frame)

        # ── Distance labels using label + frame ───────────────
        depth_labels = {}
        for i, d in enumerate(detections):
            dist = depth_est.get_distance_label(
                depth_map, d["box"], d["area_ratio"],
                label=d["label"], frame=frame
            )
            depth_labels[i] = dist

        frame = detector.draw(frame, detections)

        # ── Face recognition (every N frames only) ────────────
        faces = []
        if frame_count % FACE_EVERY_N == 0:
            faces = face_id.identify(frame)
        frame = face_id.draw(frame, faces)

        for f in faces:
            if f["name"] != "Unknown":
                voice.speak(f"{f['name']} is here", key=f"face_{f['name']}")

        # ── Voice alerts ──────────────────────────────────────
        voice.speak_detections(detections, depth_labels)
        assistant.auto_announce(detections, depth_labels)

        # ── Spatial audio (only if enabled + danger) ──────────
        if SPATIAL_AUDIO_ENABLED:
            for d in detections:
                if d["area_ratio"] > 0.20:  # only play beep for close objects
                    audio.play(d["zone"], "danger")

        # ── Distance labels on boxes ──────────────────────────
        for i, d in enumerate(detections):
            x1, y1 = d["box"][0], d["box"][1]
            dist   = depth_labels.get(i, "")
            cv2.putText(frame, dist, (x1, y1 - 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 2)

        # ── Depth overlay ─────────────────────────────────────
        if depth_enabled and depth_map is not None:
            depth_color = depth_est.colorize(depth_map)
            if depth_color is not None:
                frame = cv2.addWeighted(frame, 0.7, depth_color, 0.3, 0)

        draw_hud(frame, fps, voice.enabled, assistant.name)
        cv2.imshow("NeuroVision", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == KEY_QUIT:
            print("\n👋  Shutting down …")
            assistant.stop_listening()
            break
        elif key == KEY_TOGGLE_VOICE:
            voice.toggle()
        elif key == KEY_TOGGLE_DEPTH:
            depth_enabled = not depth_enabled
            print(f"📡  Depth: {'ON' if depth_enabled else 'OFF'}")
        elif key == KEY_OCR:
            texts = ocr.read(frame)
            msg   = ocr.build_message(texts)
            print(f"📖  {msg}")
            voice.speak(msg, key="_manual_ocr")
        elif key == ord("s"):
            path = f"assets/snapshot_{snap_n}.jpg"
            os.makedirs("assets", exist_ok=True)
            cv2.imwrite(path, frame)
            snap_n += 1
            print(f"📸  {path}")

    cap.release()
    cv2.destroyAllWindows()
    print("✅  Done.\n")


if __name__ == "__main__":
    main()
