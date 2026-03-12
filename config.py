# ─────────────────────────────────────────────
#  NeuroVision — Global Configuration
# ─────────────────────────────────────────────
#  Edit this file to tune the system behaviour.
#  No need to touch any other file for basic setup.

# ── Camera ────────────────────────────────────
# 0 = built-in webcam
# 1 = iPhone Continuity Camera / DroidCam (try 1 or 2 if 0 doesn't work)
CAMERA_INDEX = 1

FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720
FPS_TARGET   = 30

# ── Detection ─────────────────────────────────
YOLO_MODEL        = "yolov8n.pt"    # nano = fastest; swap to yolov8s.pt for more accuracy
CONFIDENCE_THRESH = 0.45            # ignore detections below this confidence
NMS_IOU_THRESH    = 0.45

# Objects that trigger alerts (COCO class names)
ALERT_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "chair", "dining table", "couch", "potted plant",
    "stairs", "door", "stop sign", "fire hydrant", "bench",
    "laptop", "tv", "bottle", "cup", "bowl",
]

# ── Depth / Distance ──────────────────────────
DEPTH_MODEL       = "DPT_Large"     # or "MiDaS_small" for speed
# Distance thresholds in "depth units" (relative, not metres yet)
DANGER_DEPTH      = 0.75            # very close — rapid alert
WARNING_DEPTH     = 0.50            # approaching — normal alert

# ── Voice Alerts ──────────────────────────────
VOICE_ENABLED     = True
VOICE_RATE        = 175             # words per minute (macOS default ~200)
VOICE_VOLUME      = 1.0             # 0.0 – 1.0
# Minimum seconds between alerts for the same object (prevents spam)
ALERT_COOLDOWN    = 3.0

# ── Spatial Audio ─────────────────────────────
SPATIAL_AUDIO_ENABLED = True
# Frame zones: left = 0 to LEFT_ZONE, right = RIGHT_ZONE to frame width
LEFT_ZONE_RATIO   = 0.35
RIGHT_ZONE_RATIO  = 0.65

# ── Face Recognition ──────────────────────────
FACE_RECOGNITION_ENABLED = True
KNOWN_FACES_DIR   = "known_faces"   # put Name.jpg files here
FACE_TOLERANCE    = 0.55            # lower = stricter match

# ── OCR ───────────────────────────────────────
OCR_ENABLED       = True
OCR_LANGUAGES     = ["en"]
# Key 'r' to trigger OCR on current frame
OCR_HOTKEY        = ord("r")

# ── Display ───────────────────────────────────
SHOW_FPS          = True
SHOW_DEPTH_MAP    = False           # set True to show depth overlay window
SHOW_LABELS       = True
BOX_COLOR         = (0, 229, 255)   # cyan — BGR
DANGER_COLOR      = (0, 60, 255)    # red  — BGR
TEXT_COLOR        = (255, 255, 255)
FONT_SCALE        = 0.6
FONT_THICKNESS    = 2

# ── Keybindings ───────────────────────────────
KEY_QUIT          = ord("q")
KEY_OCR           = ord("r")        # r = read text in frame
KEY_TOGGLE_DEPTH  = ord("d")        # d = toggle depth map
KEY_TOGGLE_VOICE  = ord("v")        # v = mute/unmute voice
