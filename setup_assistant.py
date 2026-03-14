"""
NeuroVision — Assistant Setup
──────────────────────────────
Run this once to:
  1. Set your assistant's name
  2. Tell it your name
  3. Enroll your voice

After this, only your voice will trigger commands.
Run: python setup_assistant.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.voice     import VoiceAlert
from modules.detector  import Detector
from modules.depth     import DepthEstimator
from modules.face_id   import FaceIdentifier
from modules.ocr       import TextReader
from modules.assistant import VoiceAssistant

print("\n🧠  NeuroVision Assistant Setup\n")

voice    = VoiceAlert()
detector = Detector()
depth    = DepthEstimator()
face_id  = FaceIdentifier()
ocr      = TextReader()

assistant = VoiceAssistant(voice, detector, face_id, ocr, depth)
assistant.first_run_setup()

print("\n✅  Setup complete!")
print(f"   Assistant name : {assistant.name}")
print(f"   Owner          : {assistant.owner_name}")
print(f"   Config saved   : assistant_config.json")
print("\n   Run  python main.py  to start NeuroVision\n")
