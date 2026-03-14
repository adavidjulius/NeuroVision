# 🧠 NeuroVision
### Multisensory AI Smart Glasses for the Visually Impaired

> *"It's not a gadget that talks at you — it's a second sense that quietly keeps you safe."*

NeuroVision is an open-source AI assistive wearable that helps visually impaired individuals navigate the world independently. It uses a camera, on-device AI, spatial audio, and haptic vibration to replace missing visual cues with real-time intelligent feedback — no internet connection required.

---

## 🎯 What It Does

Point the camera at any environment and NeuroVision will:

| Capability | Example Output |
|---|---|
| 🚧 Detect obstacles | *"Obstacle ahead, ~1.5 metres"* |
| 👤 Recognise people | *"Professor Kumar approaching on your left"* |
| 📖 Read signs & text | *"Sign reads: EXIT, push door"* |
| 🔊 Spatial audio cues | Sound plays from left/right ear based on where the object is |
| 📳 Haptic guidance | Vibration on left → turn left, rapid pulse → stop immediately |

All processing runs **on-device** — no cloud, no phone dependency, no internet needed.

---

## ⚙️ How It Works

```
📷 Camera captures environment
        ↓
🧠 YOLOv8 detects objects in real time
        ↓
📡 MiDaS estimates distance to each object
        ↓
👤 Face recognition identifies known people
        ↓
📖 OCR reads signs and text
        ↓
🔊 Spatial audio + 📳 haptic feedback guides the user
```

---

## 🛠 Build Phases

| Phase | Feature | Status |
|-------|---------|--------|
| 1 | Camera feed + project setup | ✅ Complete |
| 2 | YOLOv8 object detection | ✅ Complete |
| 3 | Voice alerts + spatial audio | ✅ Complete |
| 4 | MiDaS depth / distance estimation | ✅ Complete |
| 5 | Face recognition + OCR text reading | ✅ Complete |

---

## 🚀 Quick Start (Local — macOS / Linux)

### Prerequisites
- Python 3.11 or 3.12
- A webcam, iPhone (Continuity Camera), or Android phone (DroidCam)
- macOS or Linux (Windows untested)

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/adavidjulius/NeuroVision.git
cd NeuroVision

# 2. Create virtual environment
python3 -m venv venv --system-site-packages
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip setuptools wheel
pip install opencv-python-headless ultralytics torch torchvision timm
pip install pyttsx3 pygame easyocr face-recognition
pip install git+https://github.com/ageitgey/face_recognition_models

# 4. Find your camera index
python3 << 'SCAN'
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} → WORKS")
        cap.release()
SCAN

# 5. Set your camera index in config.py
#    CAMERA_INDEX = 0   ← built-in webcam
#    CAMERA_INDEX = 1   ← iPhone / external camera

# 6. Test the camera
python tests/test_camera.py

# 7. Run the full system
python main.py
```

### Controls while running

| Key | Action |
|-----|--------|
| `q` | Quit |
| `v` | Toggle voice alerts on/off |
| `d` | Toggle depth map overlay |
| `r` | Trigger OCR on current frame |
| `s` | Save snapshot |

---

## 💻 Testing in GitHub Codespaces (No Camera)

All modules can be tested headlessly using sample images — no camera or speakers needed.

```bash
# Fix OpenCV for Codespaces
sudo apt-get install -y libgl1 libglib2.0-0
pip install opencv-python-headless

# Download sample images
mkdir -p tests/samples
curl -L "https://ultralytics.com/images/bus.jpg"    -o tests/samples/bus.jpg
curl -L "https://ultralytics.com/images/zidane.jpg" -o tests/samples/face_sample.jpg

# Run phase tests
python tests/test_camera.py        # Phase 1 — camera check
python tests/test_detection.py     # Phase 2 — object detection
python tests/test_voice.py         # Phase 3 — voice alerts
python tests/test_depth.py         # Phase 4 — depth estimation
python tests/test_phase5.py        # Phase 5 — face rec + OCR
```

Results are saved as images in `tests/samples/` — open them in the Codespaces file explorer to inspect.

---

## 👤 Adding Known Faces

To make NeuroVision recognise specific people:

1. Add a clear, well-lit frontal photo to the `known_faces/` folder
2. Name the file after the person: `ProfessorKumar.jpg`, `Mom.jpg`, etc.
3. Restart the system — it loads faces automatically on startup

```
known_faces/
├── ProfessorKumar.jpg
├── Mom.jpg
└── David.jpg
```

When that person appears on camera, NeuroVision will say: *"Professor Kumar approaching on your left"*

---

## 📱 Phone as Camera

**iPhone (macOS Ventura+)**
Just plug in via USB cable. macOS Continuity Camera exposes it automatically as a camera input. Set `CAMERA_INDEX = 1` in `config.py`.

**Android**
Install [DroidCam](https://www.dev47apps.com/) on your phone and Mac. Set `CAMERA_INDEX = 1` in `config.py`.

---

## 🔧 Configuration

All settings are in `config.py`. Key ones to know:

```python
CAMERA_INDEX      = 1       # which camera to use
YOLO_MODEL        = "yolov8n.pt"   # nano=fast, yolov8s.pt=more accurate
CONFIDENCE_THRESH = 0.45    # detection sensitivity (lower = more detections)
ALERT_COOLDOWN    = 3.0     # seconds between repeat alerts for same object
VOICE_ENABLED     = True    # turn voice on/off
FACE_TOLERANCE    = 0.55    # face match strictness (lower = stricter)
```

---

## 🔬 Tech Stack

| Layer | Technology |
|-------|-----------|
| Object Detection | YOLOv8 nano (Ultralytics) |
| Depth Estimation | MiDaS small (Intel ISL) |
| Face Recognition | face_recognition + dlib |
| OCR | EasyOCR |
| Voice Alerts | pyttsx3 (offline TTS) |
| Spatial Audio | pygame |
| Camera | OpenCV |
| Language | Python 3.12 |

---

## 🛠 Production Hardware (Wearable Build)

| Role | Component | Cost |
|------|-----------|------|
| Camera | Arducam Module | ~$25 |
| Processor | Raspberry Pi 5 (4GB) | ~$80 |
| AI Accelerator | Google Coral USB | ~$60 |
| Depth Sensor | Intel RealSense D435 | ~$180 |
| Haptic Controller | Arduino Nano | ~$10 |
| Audio Output | Bone Conduction Headset | ~$30 |
| Frame + Motors | 3D printed + vibration motors | ~$40 |
| **Total** | | **~$425** |

---

## 🌍 Impact

- **253 million** people worldwide live with visual impairment
- **<5%** have access to assistive technology
- NeuroVision's silent haptic-first design means users navigate without announcing their disability
- Affordable open-source hardware makes it deployable globally

Aligned with **SDG 3** (Good Health), **SDG 10** (Reduced Inequalities), **SDG 11** (Sustainable Cities).

---

## 📄 License

MIT License — free to use, modify, and build upon.

---

## 🙋 Contributing

Pull requests welcome. Please open an issue first to discuss major changes.

Built with ❤️ for the visually impaired community.

---

## 📥 Getting the Project onto Your Computer

There are two ways — pick whichever suits you.

---

### Option A — Using Git (Recommended)

Git downloads the project and lets you pull future updates easily.

**Install Git first if you don't have it:**

| OS | Command |
|----|---------|
| macOS | `brew install git` (get Homebrew from [brew.sh](https://brew.sh) first) |
| Linux | `sudo apt install git` |
| Windows | Download installer from [git-scm.com](https://git-scm.com) |

**Then in Terminal / Command Prompt:**

```bash
# Step 1 — Download the project
git clone https://github.com/adavidjulius/NeuroVision.git

# Step 2 — Enter the folder
cd NeuroVision

# Step 3 — Confirm the files are there
ls
# Should show: main.py  config.py  modules/  tests/  requirements.txt
```

---

### Option B — Download as ZIP (No Git needed)

1. Go to **[github.com/adavidjulius/NeuroVision](https://github.com/adavidjulius/NeuroVision)**
2. Click the green **`< > Code`** button (top right of the file list)
3. Click **Download ZIP**
4. Find the downloaded ZIP in your Downloads folder
5. **Extract / Unzip** it to your Desktop

Then open Terminal and go to the folder:

```bash
# macOS
cd ~/Desktop/NeuroVision-main

# Linux
cd ~/Desktop/NeuroVision-main

# Windows (Command Prompt)
cd C:\Users\YourName\Desktop\NeuroVision-main
```

---

### Verify you're in the right folder

Run this — you should see the project files:

```bash
ls
```

Expected output:
```
README.md   config.py   main.py   modules/   requirements.txt   tests/
```

If you see this, you're ready. Move on to the setup steps below. ✅

---

## 💻 Full Local Setup — Step by Step

### Step 1 — Install Python

NeuroVision requires **Python 3.11 or 3.12**.

Check if you already have it:
```bash
python3 --version
```

If not installed:
- **macOS**: `brew install python@3.11`
- **Linux**: `sudo apt install python3.11 python3.11-venv`
- **Windows**: Download from [python.org](https://www.python.org/downloads/) — tick **"Add to PATH"** during install

---

### Step 2 — Create a Virtual Environment

A virtual environment keeps NeuroVision's packages separate from the rest of your system. Always use one.

```bash
# Create it
python3 -m venv venv --system-site-packages

# Activate it — macOS / Linux
source venv/bin/activate

# Activate it — Windows
venv\Scripts\activate
```

Your terminal prompt will now show `(venv)` at the start. Every time you open a new terminal to work on NeuroVision, re-run the activate command.

---

### Step 3 — Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip setuptools wheel

# Core vision + AI
pip install opencv-python ultralytics torch torchvision timm

# Voice + audio
pip install pyttsx3 pygame

# Face recognition
pip install face-recognition
pip install git+https://github.com/ageitgey/face_recognition_models

# OCR
pip install easyocr
```

> ⚠️ **Note for macOS users**: If you get a `libGL` error, run:
> `brew install ffmpeg` then reinstall opencv.
>
> ⚠️ **Note for Linux / Codespaces**: Use `opencv-python-headless` instead of `opencv-python`:
> `pip install opencv-python-headless`

---

### Step 4 — Connect Your Camera

**iPhone (macOS Ventura or later)**
Just plug your iPhone into your Mac with a USB cable. macOS automatically detects it as a camera. No app needed.

**Android**
Install [DroidCam](https://www.dev47apps.com/) on both your phone and computer. Follow the app's instructions to connect.

**Built-in webcam**
Nothing to do — it works by default.

---

### Step 5 — Find Your Camera Index

Different cameras get different index numbers. Run this to find yours:

```bash
python3 << 'EOF'
import cv2
print("Scanning for cameras...")
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"  Camera {i} → WORKS")
        cap.release()
    else:
        print(f"  Camera {i} → not found")
EOF
```

Note which index number works for your camera. Then open `config.py` and set it:

```python
CAMERA_INDEX = 1   # ← change this to your number
```

---

### Step 6 — Test the Camera

```bash
python tests/test_camera.py
```

A window should open showing your live camera feed with the NeuroVision HUD overlay. Press `q` to quit.

---

### Step 7 — Add Known Faces (Optional)

To make NeuroVision recognise specific people, add their photo to the `known_faces/` folder:

```
known_faces/
├── ProfessorKumar.jpg    ← clear frontal photo
├── Mom.jpg
└── David.jpg
```

The filename becomes the person's name. One good photo per person is enough.

---

### Step 8 — Run NeuroVision

```bash
python main.py
```

Point your camera at the environment and NeuroVision will start detecting, speaking, and guiding in real time.

---

### Everyday Workflow

Every time you come back to work on the project:

```bash
# 1. Go to the project folder
cd ~/Desktop/NeuroVision

# 2. Activate the virtual environment
source venv/bin/activate   # macOS/Linux
# or
venv\Scripts\activate      # Windows

# 3. Run
python main.py
```

---

## ❓ Common Problems & Fixes

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'cv2'` | Run `pip install opencv-python` |
| `Camera not authorized` | macOS: System Settings → Privacy → Camera → enable Terminal |
| `Camera index 0 not found` | Run the camera scan above and update `CAMERA_INDEX` in config.py |
| `No module named 'pkg_resources'` | Run `pip install --upgrade setuptools` |
| `libGL.so not found` (Linux) | Run `sudo apt install libgl1 libglib2.0-0` |
| Voice not speaking | Install eSpeak: `sudo apt install espeak` (Linux) or check System Settings on macOS |
| MiDaS model slow | Normal on CPU first run — it caches after. Use `MiDaS_small` in config.py for speed |
