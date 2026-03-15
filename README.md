# 🧠 NeuroVision
### Multisensory AI Smart Glasses for the Visually Impaired

> *"It's not a gadget that talks at you — it's a second sense that quietly keeps you safe."*

NeuroVision is an open-source AI assistive wearable that helps visually impaired individuals navigate independently using camera, on-device AI, spatial audio and voice guidance — no internet required.

---

## 🎯 What It Does

| Capability | Example |
|---|---|
| 🚧 Obstacle detection | *"Chair on your left, ~1.5m"* |
| 👤 Face recognition | *"David is here"* |
| 📖 Text reading (OCR) | *"Sign reads: EXIT RIGHT"* |
| 🚨 Danger alerts | *"Danger! Vehicle ahead"* |
| 🗺️ Room detection | *"You are in the living room"* |
| 🎙️ Voice assistant | *"Hey Julius, what do you see?"* |

---

## ⚙️ System Architecture
```
📷 Camera → 🧠 YOLOv8 Detection → 📡 MiDaS Depth
                    ↓
        Environment Interpretation
                    ↓
    🔊 Voice Alerts + 🚨 Danger Detection
                    ↓
        👤 Face ID + 📖 OCR + 🗺️ Navigation
                    ↓
         🎙️ Hey Julius Voice Assistant
```

---

## 🛠 Build Phases

| Phase | Feature | Status |
|-------|---------|--------|
| 1 | Camera feed + setup | ✅ |
| 2 | YOLOv8 object detection | ✅ |
| 3 | Voice alerts + spatial audio | ✅ |
| 4 | MiDaS depth estimation | ✅ |
| 5 | Face recognition + OCR | ✅ |
| S1 | Performance — threaded workers | ✅ |
| S2 | Danger alerts — proximity tiers | ✅ |
| S3 | Object identification | ✅ |
| S4 | Multi-language OCR (Tamil+Hindi+EN) | ✅ |
| S5 | Indoor navigation + room mapping | ✅ |
| S6 | Mobile app — Flask server | ✅ |

---

## 🚀 Quick Start
```bash
git clone https://github.com/adavidjulius/NeuroVision.git
cd NeuroVision
bash install.sh
source venv/bin/activate
python main.py
```

---

## 🎙️ Voice Commands

Say **"Hey Julius"** then:

| Command | What happens |
|---------|-------------|
| *"What do you see?"* | Describes full scene with distances |
| *"Auto mode on"* | Continuously announces objects |
| *"Read that"* | Reads any text or sign |
| *"Who is this?"* | Identifies person in view |
| *"This is my friend David"* | Saves face as David |
| *"Be quiet"* | Mutes for 30 seconds |
| *"Help"* | Lists all commands |

---

## 📱 Phone as Camera

**iPhone** — plug via USB or use Continuity Camera (WiFi, same Apple ID).
Set `CAMERA_INDEX = 1` in `config.py`.

**Android** — install [DroidCam](https://www.dev47apps.com/).

---

## 🔧 Key Settings (config.py)
```python
CAMERA_INDEX      = 1        # 0=built-in, 1=iPhone
YOLO_MODEL        = "yolov8s.pt"
CONFIDENCE_THRESH = 0.65     # higher = fewer false detections
ALERT_COOLDOWN    = 5.0      # seconds between same alerts
VOICE_ENABLED     = True
SPATIAL_AUDIO_ENABLED = False
```

---

## 💻 Tech Stack

| Layer | Technology |
|-------|-----------|
| Object Detection | YOLOv8s (Ultralytics) |
| Depth Estimation | MiDaS small |
| Face Detection | MediaPipe Tasks API |
| OCR | EasyOCR (EN + Tamil + Hindi) |
| Speech Recognition | Vosk (offline) |
| Voice Output | macOS say command |
| Spatial Audio | pygame |
| Mobile App | Flask + SocketIO |
| Language | Python 3.13 |

---

## 🛠 Production Hardware (~$425)

| Component | Part | Cost |
|-----------|------|------|
| Camera | Arducam Module | ~$25 |
| Processor | Raspberry Pi 5 | ~$80 |
| AI Accelerator | Google Coral USB | ~$60 |
| Depth Sensor | Intel RealSense D435 | ~$180 |
| Haptic Controller | Arduino Nano | ~$10 |
| Audio | Bone Conduction Headset | ~$30 |
| Frame + Motors | 3D printed | ~$40 |

---

## ❓ Troubleshooting

| Problem | Fix |
|---------|-----|
| `No module named 'cv2'` | `pip install opencv-python` |
| `mediapipe has no attribute solutions` | `pip install mediapipe==0.10.30` |
| Camera not found | Run camera scan in README |
| Voice not speaking | macOS: check Terminal microphone permission |
| `libGL not found` (Linux) | `sudo apt install libgl1 libglib2.0-0` |
| dlib build fails | Don't install dlib — we use mediapipe |

---

## 🌍 Impact

- 253M people live with visual impairment globally
- <5% have access to assistive technology
- SDG 3 · SDG 10 · SDG 11

---

## 📄 License
MIT — free to use, modify, build upon.

Built with ❤️ for the visually impaired community.
