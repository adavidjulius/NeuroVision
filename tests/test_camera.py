"""
NeuroVision — Phase 1: Camera Test
────────────────────────────────────
Run this FIRST to confirm your camera works.

    python tests/test_camera.py

Controls:
  q → quit
  s → save a snapshot to assets/snapshot.jpg
  1/2/3 → switch camera index on the fly
"""

import cv2
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT

os.makedirs("assets", exist_ok=True)


def test_camera(index: int = CAMERA_INDEX):
    print(f"\n🎥  NeuroVision — Camera Test")
    print(f"   Trying camera index {index} ...")
    print(f"   Press  q  to quit  |  s  to snapshot  |  1/2/3  to switch camera\n")

    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"❌  Could not open camera {index}.")
        print("   Try changing CAMERA_INDEX in config.py  (0, 1, 2 …)\n")
        return False

    current_index = index
    frame_count   = 0
    snapshot_n    = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️   Frame dropped — retrying …")
            continue

        frame_count += 1
        h, w = frame.shape[:2]

        # ── Overlay info ──────────────────────────────────────
        cv2.putText(frame, f"NeuroVision Camera Test", (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 229, 255), 2)
        cv2.putText(frame, f"Camera index : {current_index}", (20, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"Resolution   : {w} x {h}", (20, 96),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"Frames       : {frame_count}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, "q=quit  s=snapshot  1/2/3=switch cam", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 100), 1)

        # Cyan crosshair in center
        cx, cy = w // 2, h // 2
        cv2.line(frame, (cx - 30, cy), (cx + 30, cy), (0, 229, 255), 1)
        cv2.line(frame, (cx, cy - 30), (cx, cy + 30), (0, 229, 255), 1)
        cv2.circle(frame, (cx, cy), 40, (0, 229, 255), 1)

        cv2.imshow("NeuroVision — Camera Test", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print(f"\n✅  Camera test passed! {frame_count} frames captured.")
            print(f"   Set CAMERA_INDEX = {current_index} in config.py\n")
            break

        elif key == ord("s"):
            path = f"assets/snapshot_{snapshot_n}.jpg"
            cv2.imwrite(path, frame)
            snapshot_n += 1
            print(f"📸  Snapshot saved → {path}")

        elif key in (ord("1"), ord("2"), ord("3")):
            new_index = int(chr(key)) - 1
            cap.release()
            cap = cv2.VideoCapture(new_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            if cap.isOpened():
                current_index = new_index
                print(f"🔄  Switched to camera {new_index}")
            else:
                print(f"❌  Camera {new_index} not available, staying on {current_index}")
                cap = cv2.VideoCapture(current_index)

    cap.release()
    cv2.destroyAllWindows()
    return True


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else CAMERA_INDEX
    success = test_camera(idx)
    sys.exit(0 if success else 1)
