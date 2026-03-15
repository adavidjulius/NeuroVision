"""
NeuroVision — Main (Threaded, High Performance)
─────────────────────────────────────────────────
All AI runs in background threads.
Main thread = camera read + draw + display only.
Target: 15-25 FPS on MacBook.
"""

import cv2
import time
import sys
import os
import threading

from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FPS_TARGET,
    KEY_QUIT, KEY_TOGGLE_VOICE, KEY_TOGGLE_DEPTH, KEY_OCR,
    SPATIAL_AUDIO_ENABLED, BOX_COLOR,
)
from modules.detector      import Detector
from modules.depth         import DepthEstimator
from modules.voice         import VoiceAlert
from modules.audio_spatial import SpatialAudio
from modules.face_id       import FaceIdentifier
from modules.ocr           import TextReader
from modules.assistant     import VoiceAssistant, CONFIG_PATH
from modules.danger import DangerDetector
from modules.performance   import (
    FrameBuffer, ResultBuffer, AIWorker, PerformanceMonitor
)
from modules.model_manager import get_device_profile


def draw_hud(frame, fps, voice_on, assistant_name, auto_mode, profile):
    h, w = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 44), (10, 18, 40), -1)
    cv2.putText(frame, "NEUROVISION", (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 229, 255), 2)

    # Auto mode indicator
    if auto_mode:
        cv2.putText(frame, "AUTO", (200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2)

    cv2.putText(frame, f"Hey {assistant_name}", (w - 230, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 100), 1)

    # FPS with color coding
    fps_color = (0,255,100) if fps>12 else (0,165,255) if fps>6 else (0,0,255)
    cv2.putText(frame, f"FPS:{fps:.0f}", (w - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, fps_color, 2)

    # Bottom bar
    cv2.rectangle(frame, (0, h - 32), (w, h), (10, 18, 40), -1)
    cv2.putText(frame, "q=quit  v=voice  d=depth  r=OCR  s=snapshot  a=auto",
                (14, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 120, 160), 1)

    # Zone lines
    third = w // 3
    cv2.line(frame, (third,     44), (third,     h - 32), (30, 50, 80), 1)
    cv2.line(frame, (third * 2, 44), (third * 2, h - 32), (30, 50, 80), 1)
    cv2.putText(frame, "LEFT",    (14,           68), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40,70,110), 1)
    cv2.putText(frame, "CENTER",  (third + 10,   68), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40,70,110), 1)
    cv2.putText(frame, "RIGHT",   (third*2 + 10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40,70,110), 1)


def main():
    print("\n🧠  NeuroVision — High Performance Mode\n")

    # ── Hardware profile ──────────────────────────────────────
    profile = get_device_profile()
    print(f"   Device : {profile['device']}")
    print(f"   Model  : {profile['yolo_model']}\n")

    # ── Camera ────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # reduce buffer lag

    if not cap.isOpened():
        print(f"❌  Cannot open camera {CAMERA_INDEX}")
        sys.exit(1)

    # ── Modules ───────────────────────────────────────────────
    detector  = Detector()
    depth_est = DepthEstimator()
    voice     = VoiceAlert()
    audio     = SpatialAudio()
    face_id   = FaceIdentifier()
    ocr       = TextReader()
    danger    = DangerDetector()
assistant = VoiceAssistant(voice, detector, face_id, ocr, depth_est)

    if not os.path.exists(CONFIG_PATH):
        assistant.first_run_setup()

    # ── Shared buffers ────────────────────────────────────────
    frame_buf      = FrameBuffer()

    det_results    = ResultBuffer(default=[])
    depth_results  = ResultBuffer(default=None)
    face_results   = ResultBuffer(default=[])
    depth_labels   = {}

    # ── AI Workers (background threads) ───────────────────────
    det_worker = AIWorker(
        "Detection", detector.detect,
        frame_buf, det_results,
        interval=profile["det_interval"]
    )
    depth_worker = AIWorker(
        "Depth", depth_est.estimate,
        frame_buf, depth_results,
        interval=profile["depth_interval"]
    )
    face_worker = AIWorker(
        "FaceID", face_id.identify,
        frame_buf, face_results,
        interval=profile["face_interval"]
    )

    det_worker.start()
    depth_worker.start()
    face_worker.start()

    # ── Voice assistant ───────────────────────────────────────
    assistant.start_listening(lambda: frame_buf.read())

    # ── Performance monitor ───────────────────────────────────
    perf          = PerformanceMonitor()
    depth_enabled = False
    snap_n        = 0
    prev_dets     = []
    prev_faces    = []
    prev_depth    = None

    print(f"\n✅  All workers running. Say 'Hey {assistant.name}' for commands.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Push latest frame to all workers
        frame_buf.write(frame)

        # ── Pull latest AI results (non-blocking) ─────────────
        detections = det_results.read()    or prev_dets
        depth_map  = depth_results.read()
        faces      = face_results.read()   or prev_faces

        if depth_map is not None:
            prev_depth = depth_map
        else:
            depth_map  = prev_depth

        prev_dets  = detections
        prev_faces = faces

        # ── Distance labels ───────────────────────────────────
        for i, d in enumerate(detections):
            depth_labels[i] = depth_est.get_distance_label(
                depth_map, d["box"], d["area_ratio"],
                label=d["label"], frame=frame
            )

        # ── Draw detections ───────────────────────────────────
        frame = detector.draw(frame, detections)
        frame = face_id.draw(frame, faces)

        # Distance labels on boxes
        for i, d in enumerate(detections):
            x1, y1 = d["box"][0], d["box"][1]
            dist   = depth_labels.get(i, "")
            cv2.putText(frame, dist, (x1, max(y1 - 28, 50)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 2)

        # ── Depth overlay ─────────────────────────────────────
        if depth_enabled and depth_map is not None:
            depth_color = depth_est.colorize(depth_map)
            if depth_color is not None:
                frame = cv2.addWeighted(frame, 0.65, depth_color, 0.35, 0)

        # ── Danger analysis
        danger_events = danger.analyze(detections)
        frame = danger.draw(frame, danger_events)
        for e in danger_events:
            key = f"danger_{e['zone']}_{e['tier']}"
            if danger.should_alert(key, e["tier"]):
                voice.speak(e["message"], key=key, force=True)
        # ── Voice alerts ──────────────────────────────────────
        voice.speak_detections(detections, depth_labels)

        # Face alerts
        for f in faces:
            if f["name"] != "Unknown":
                voice.speak(f"{f['name']} is here", key=f"face_{f['name']}")

        # Auto mode
        assistant.auto_announce(detections, depth_labels)

        # Spatial audio for danger
        if SPATIAL_AUDIO_ENABLED:
            for d in detections:
                if d["area_ratio"] > 0.20:
                    audio.play(d["zone"], "danger")

        # ── HUD + FPS ─────────────────────────────────────────
        perf.tick()
        draw_hud(frame, perf.fps, voice.enabled,
                 assistant.name, assistant.auto_mode, profile)

        cv2.imshow("NeuroVision", frame)

        # ── Keys ──────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == KEY_QUIT:
            print("\n👋  Shutting down …")
            det_worker.stop()
            depth_worker.stop()
            face_worker.stop()
            assistant.stop_listening()
            break

        elif key == KEY_TOGGLE_VOICE:
            voice.toggle()

        elif key == KEY_TOGGLE_DEPTH:
            depth_enabled = not depth_enabled
            print(f"📡  Depth: {'ON' if depth_enabled else 'OFF'}")

        elif key == KEY_OCR:
            f = frame_buf.read()
            if f is not None:
                texts = ocr.read(f)
                msg   = ocr.build_message(texts)
                print(f"📖  {msg}")
                voice.speak(msg, key="_manual_ocr")

        elif key == ord("a"):
            assistant.auto_mode = not assistant.auto_mode
            state = "ON" if assistant.auto_mode else "OFF"
            print(f"🚗  Auto mode: {state}")
            voice.speak(f"Auto mode {state}", key="_auto_toggle")

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
