#!/bin/bash
# NeuroVision — Full Setup Script
# Run: bash install.sh

set -e  # stop on any error

echo ""
echo "🧠  NeuroVision — Setup"
echo "========================"

# ── Check Python ──────────────────────────────────────────────
echo ""
echo "Checking Python..."
if command -v python3.11 &>/dev/null; then
    PYTHON=python3.11
    echo "✅  Python 3.11 found"
elif command -v python3.12 &>/dev/null; then
    PYTHON=python3.12
    echo "✅  Python 3.12 found"
elif command -v python3 &>/dev/null; then
    PYTHON=python3
    echo "✅  Python 3 found"
else
    echo "❌  Python not found"
    echo "   Install from https://python.org"
    exit 1
fi

# ── Check Homebrew (macOS) ────────────────────────────────────
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo ""
    echo "Checking Homebrew..."
    if ! command -v brew &>/dev/null; then
        echo "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        eval "$(/opt/homebrew/bin/brew shellenv)"
    else
        echo "✅  Homebrew found"
    fi

    echo "Installing system dependencies..."
    brew install cmake portaudio 2>/dev/null || true
    echo "✅  System deps ready"
fi

# ── Virtual environment ───────────────────────────────────────
echo ""
echo "Setting up virtual environment..."
if [ ! -d "venv" ]; then
    $PYTHON -m venv venv --system-site-packages
    echo "✅  Virtual environment created"
else
    echo "✅  Virtual environment exists"
fi

source venv/bin/activate
echo "✅  Virtual environment activated"

# ── Upgrade pip ───────────────────────────────────────────────
echo ""
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel -q
echo "✅  pip upgraded"

# ── Install packages ──────────────────────────────────────────
echo ""
echo "Installing packages (this takes 3-5 minutes)..."
echo ""

echo "  [1/8] Core packages..."
pip install opencv-python numpy Pillow -q
echo "  ✅  Core done"

echo "  [2/8] AI models (largest download)..."
pip install ultralytics torch torchvision timm -q
echo "  ✅  AI models done"

echo "  [3/8] Voice & audio..."
pip install pygame SpeechRecognition pyaudio -q
echo "  ✅  Voice done"

echo "  [4/8] Vosk offline speech..."
pip install vosk -q
echo "  ✅  Vosk done"

echo "  [5/8] Face detection..."
pip install "mediapipe==0.10.14" -q
echo "  ✅  Face detection done"

echo "  [6/8] OCR..."
pip install easyocr -q
echo "  ✅  OCR done"

echo "  [7/8] Mobile server..."
pip install flask flask-socketio flask-cors -q
echo "  ✅  Server done"

echo "  [8/8] Utilities..."
pip install pyyaml psutil requests python-dotenv -q
echo "  ✅  Utilities done"

# ── Download Vosk model ───────────────────────────────────────
echo ""
echo "Downloading Vosk speech model..."
if [ ! -d "models/vosk-en" ]; then
    mkdir -p models
    curl -L "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip" \
         -o models/vosk-model.zip --progress-bar
    cd models
    unzip -q vosk-model.zip
    mv vosk-model-small-en-us-0.15 vosk-en
    rm vosk-model.zip
    cd ..
    echo "✅  Vosk model ready"
else
    echo "✅  Vosk model already downloaded"
fi

# ── Create folders ────────────────────────────────────────────
echo ""
echo "Creating project folders..."
mkdir -p known_faces assets/sounds tests/samples
echo "✅  Folders ready"

# ── Find camera ───────────────────────────────────────────────
echo ""
echo "Scanning for cameras..."
python3 << 'PYEOF'
import cv2, time
time.sleep(1)
found = []
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        found.append(i)
        cap.release()
if found:
    print(f"✅  Cameras found at index: {found}")
    print(f"   Built-in camera is usually index 0")
    print(f"   iPhone/external is usually index 1")
    print(f"   Edit CAMERA_INDEX in config.py to set your camera")
else:
    print("⚠️   No cameras found right now")
    print("   Plug in your camera and update CAMERA_INDEX in config.py")
PYEOF

# ── Done ──────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "✅  NeuroVision setup complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo ""
echo "  1. Activate venv (every new terminal):"
echo "     source venv/bin/activate"
echo ""
echo "  2. Set your camera index in config.py:"
echo "     CAMERA_INDEX = 0  (built-in)"
echo "     CAMERA_INDEX = 1  (iPhone/external)"
echo ""
echo "  3. First time setup (sets name + voice):"
echo "     python setup_assistant.py"
echo ""
echo "  4. Run NeuroVision:"
echo "     python main.py"
echo ""
echo "  5. Mobile app (open on phone browser):"
echo "     python run_server.py"
echo "     Then: http://YOUR-MAC-IP:5000"
echo ""
echo "  Say 'Hey Julius' to activate voice commands"
echo ""
