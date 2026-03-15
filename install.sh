#!/bin/bash
set -e

echo ""
echo "🧠  NeuroVision — Setup"
echo "========================"

# Python check
if command -v python3.11 &>/dev/null; then PYTHON=python3.11
elif command -v python3.12 &>/dev/null; then PYTHON=python3.12
elif command -v python3 &>/dev/null; then PYTHON=python3
else echo "❌ Python not found"; exit 1; fi
echo "✅  Python: $($PYTHON --version)"

# Homebrew (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    if ! command -v brew &>/dev/null; then
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
    brew install cmake portaudio 2>/dev/null || true
    echo "✅  System deps ready"
fi

# Virtual environment
if [ ! -d "venv" ]; then
    $PYTHON -m venv venv --system-site-packages
fi
source venv/bin/activate
pip install --upgrade pip setuptools wheel -q
echo "✅  Virtual environment ready"

# Install packages
echo ""
echo "Installing packages..."

echo "  [1/7] Core..."
pip install opencv-python numpy Pillow -q
echo "  ✅  Core"

echo "  [2/7] AI models..."
pip install ultralytics torch torchvision timm -q
echo "  ✅  AI models"

echo "  [3/7] Voice..."
pip install pygame SpeechRecognition pyaudio vosk -q
echo "  ✅  Voice"

echo "  [4/7] Face detection..."
pip install "mediapipe==0.10.30" -q
echo "  ✅  Face detection"

echo "  [5/7] OCR..."
pip install easyocr -q
echo "  ✅  OCR"

echo "  [6/7] Mobile server..."
pip install flask flask-socketio flask-cors -q
echo "  ✅  Server"

echo "  [7/7] Utilities..."
pip install pyyaml psutil requests python-dotenv -q
echo "  ✅  Utilities"

# Vosk model
echo ""
echo "Downloading speech model..."
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
    echo "✅  Vosk model exists"
fi

# Folders
mkdir -p known_faces assets/sounds tests/samples
echo "✅  Folders ready"

# Camera scan
echo ""
echo "Scanning cameras..."
$PYTHON -c "
import cv2, time; time.sleep(1)
found = []
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        found.append(i)
        cap.release()
if found:
    print(f'✅  Cameras found: {found}')
    print(f'   Set CAMERA_INDEX in config.py')
else:
    print('⚠️   No cameras found')
"

echo ""
echo "========================================"
echo "✅  NeuroVision setup complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  source venv/bin/activate"
echo "  python main.py"
echo ""
echo "  Say 'Hey Julius' for voice commands"
echo ""
