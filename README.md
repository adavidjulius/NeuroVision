# 🧠 NeuroVision
### Multisensory AI Smart Glasses for the Visually Impaired

A wearable AI system that replaces missing visual cues with intelligent real-time feedback using computer vision, spatial audio, and haptic guidance.

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/adavidjulius/NeuroVision.git
cd NeuroVision

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Test your camera
python tests/test_camera.py

# 5. Run the full system
python main.py
```

---

## 📁 Project Structure

```
neurovision/
├── main.py                 # Entry point — runs full pipeline
├── config.py               # All settings (camera, thresholds, voices)
├── requirements.txt
├── modules/
│   ├── detector.py         # YOLOv8 object detection
│   ├── depth.py            # MiDaS monocular depth estimation
│   ├── voice.py            # pyttsx3 offline voice alerts
│   ├── audio_spatial.py    # Spatial left/right audio panning
│   ├── face_id.py          # Face recognition
│   └── ocr.py              # EasyOCR text reading
├── known_faces/            # Add photos here: person_name.jpg
├── assets/sounds/          # Alert sound files
└── tests/
    └── test_camera.py      # Camera connection test
```

---

## 🔧 Build Phases

| Phase | Feature | Status |
|-------|---------|--------|
| 1 | Camera feed + project setup | ✅ |
| 2 | Object detection (YOLO) | 🔄 |
| 3 | Voice alerts | 🔄 |
| 4 | Depth estimation (MiDaS) | 🔄 |
| 5 | Face recognition + OCR | 🔄 |

---

## 📱 Phone as Camera (macOS)

**iPhone**: Just plug in via USB — macOS Continuity Camera works automatically in Python.

**Android**: Install [DroidCam](https://www.dev47apps.com/) on phone + Mac, then set `CAMERA_INDEX = 1` in `config.py`.

---

## 🛠 Hardware (Production)

| Component | Part |
|-----------|------|
| Camera | Arducam Module |
| Processor | Raspberry Pi 5 |
| AI Accelerator | Google Coral USB |
| Depth Sensor | Intel RealSense D435 |
| Haptic Controller | Arduino Nano |
| Audio | Bone Conduction Headset |
