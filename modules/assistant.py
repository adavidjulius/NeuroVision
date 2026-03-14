"""
NeuroVision — Voice Assistant
──────────────────────────────
Features:
  - Custom assistant name (set on first run, saved)
  - Wake word detection ("Hey [name]")
  - Voice owner fingerprinting (only responds to registered voice)
  - Voice commands: add face, describe scene, read text, help
"""

import os
import json
import time
import threading
import numpy as np

CONFIG_PATH = "assistant_config.json"

DEFAULT_CONFIG = {
    "assistant_name":    "NeuroVision",
    "wake_words":        ["hey neurovision"],
    "owner_enrolled":    False,
    "owner_name":        "User",
    "response_style":    "concise",   # concise | detailed
    "confirm_commands":  True,
}


class VoiceAssistant:
    def __init__(self, voice_alert, detector, face_id, ocr, depth):
        self.voice      = voice_alert
        self.detector   = detector
        self.face_id    = face_id
        self.ocr        = ocr
        self.depth      = depth

        self.config     = self._load_config()
        self.listening  = False
        self.awake      = False
        self._lock      = threading.Lock()

        self.sr         = None
        self.mic        = None
        self.embedder   = None
        self.owner_emb  = None

        self._init_sr()
        self._init_voice_fingerprint()

        print(f"🎙️   Voice Assistant ready")
        print(f"   Name     : {self.config['assistant_name']}")
        print(f"   Wake word: Hey {self.config['assistant_name']}")
        print(f"   Owner    : {self.config['owner_name']}")

    # ── Config ────────────────────────────────────────────────

    def _load_config(self) -> dict:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)
            # merge with defaults for any missing keys
            for k, v in DEFAULT_CONFIG.items():
                cfg.setdefault(k, v)
            return cfg
        return DEFAULT_CONFIG.copy()

    def _save_config(self):
        with open(CONFIG_PATH, "w") as f:
            json.dump(self.config, f, indent=2)

    # ── Speech recognition ────────────────────────────────────

    def _init_sr(self):
        try:
            import speech_recognition as sr
            self.sr  = sr
            self.mic = sr.Microphone()
            print("✅  Speech recognition ready")
        except Exception as e:
            print(f"⚠️   Speech recognition unavailable ({e})")
            self.sr  = None
            self.mic = None

    # ── Voice fingerprint ─────────────────────────────────────

    def _init_voice_fingerprint(self):
        try:
            from resemblyzer import VoiceEncoder
            self.embedder  = VoiceEncoder()
            emb_path       = "assistant_owner_voice.npy"
            if os.path.exists(emb_path) and self.config["owner_enrolled"]:
                self.owner_emb = np.load(emb_path)
                print(f"✅  Owner voice loaded ({self.config['owner_name']})")
            else:
                print("⚠️   No owner voice enrolled yet — run setup_assistant.py")
        except Exception as e:
            print(f"⚠️   Voice fingerprint unavailable ({e}) — responding to all voices")
            self.embedder  = None
            self.owner_emb = None

    def _is_owner(self, audio_data: np.ndarray) -> bool:
        """Returns True if the speaker matches the enrolled owner."""
        if self.embedder is None or self.owner_emb is None:
            return True  # no fingerprint → respond to anyone

        try:
            emb        = self.embedder.embed_utterance(audio_data)
            similarity = np.dot(emb, self.owner_emb) / (
                np.linalg.norm(emb) * np.linalg.norm(self.owner_emb)
            )
            return float(similarity) > 0.75
        except Exception:
            return True

    # ── First-run setup ───────────────────────────────────────

    def first_run_setup(self):
        """
        Called on first launch.
        Asks user to set assistant name and enroll their voice.
        """
        print("\n" + "=" * 50)
        print("  NeuroVision — First Time Setup")
        print("=" * 50)

        self.voice.speak(
            "Welcome to NeuroVision. Let's set things up. "
            "First, what would you like to call me?"
        )

        name = self._listen_for_text(timeout=8)
        if name and len(name.strip()) > 1:
            clean = name.strip().title()
            self.config["assistant_name"] = clean
            self.config["wake_words"]     = [f"hey {clean.lower()}"]
            self.voice.speak(
                f"Great, you can call me {clean}. "
                f"Just say Hey {clean} to wake me up anytime."
            )
            print(f"✅  Assistant name set to: {clean}")
        else:
            self.voice.speak(
                "I'll go by NeuroVision. You can change this later."
            )

        # Ask owner name
        self.voice.speak("What's your name? I'll remember you.")
        owner = self._listen_for_text(timeout=6)
        if owner and len(owner.strip()) > 1:
            self.config["owner_name"] = owner.strip().title()
            self.voice.speak(f"Nice to meet you, {self.config['owner_name']}.")

        # Enroll voice
        self.voice.speak(
            "Now I'll learn your voice so I only respond to you. "
            "Please say something after the beep — anything works."
        )
        self._enroll_owner_voice()

        self.config["owner_enrolled"] = True
        self._save_config()

        self.voice.speak(
            f"All set! Say Hey {self.config['assistant_name']} whenever you need me."
        )
        print("=" * 50 + "\n")

    def _enroll_owner_voice(self):
        """Records owner voice sample and saves embedding."""
        if self.sr is None or self.embedder is None:
            print("⚠️   Cannot enroll voice — SR or embedder unavailable")
            return

        try:
            import speech_recognition as sr_lib
            recognizer = sr_lib.Recognizer()
            with self.mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=1)
                print("🎙️   Recording voice sample ... speak now")
                audio = recognizer.listen(source, timeout=8, phrase_time_limit=6)

            raw   = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
            flt   = raw.astype(np.float32) / 32768.0
            emb   = self.embedder.embed_utterance(flt)
            np.save("assistant_owner_voice.npy", emb)
            self.owner_emb = emb
            print("✅  Voice sample enrolled and saved")
        except Exception as e:
            print(f"⚠️   Voice enrollment failed ({e})")

    # ── Main listen loop ──────────────────────────────────────

    def start_listening(self, frame_provider):
        """
        Starts background thread that listens for wake word.
        frame_provider: callable that returns current camera frame.
        """
        if self.sr is None:
            print("⚠️   No microphone — voice commands disabled")
            return

        self.listening     = True
        self._frame_fn     = frame_provider
        self._thread       = threading.Thread(
            target=self._listen_loop, daemon=True
        )
        self._thread.start()
        print(f"🎙️   Listening for: Hey {self.config['assistant_name']}")

    def stop_listening(self):
        self.listening = False

    def _listen_loop(self):
        recognizer = self.sr.Recognizer()
        recognizer.energy_threshold        = 300
        recognizer.dynamic_energy_threshold = True

        while self.listening:
            try:
                with self.mic as source:
                    recognizer.adjust_for_ambient_noise(source, duration=0.3)
                    audio = recognizer.listen(
                        source, timeout=3, phrase_time_limit=6
                    )

                raw  = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
                flt  = raw.astype(np.float32) / 32768.0

                # Check owner voice first
                if not self._is_owner(flt):
                    continue

                text = recognizer.recognize_google(audio).lower().strip()
                print(f"🎤  Heard: \"{text}\"")

                # Check for wake word
                name_lower = self.config["assistant_name"].lower()
                wake_words = self.config["wake_words"] + [name_lower]

                if any(w in text for w in wake_words):
                    self._handle_wake(text, recognizer)

            except self.sr.WaitTimeoutError:
                pass
            except self.sr.UnknownValueError:
                pass
            except Exception as e:
                time.sleep(0.5)

    def _handle_wake(self, initial_text: str, recognizer):
        """Called when wake word is detected. Listens for command."""
        name = self.config["assistant_name"]
        self.voice.speak(f"Yes?", key="_wake_ack")

        # Strip wake word from initial text to check for inline command
        for w in self.config["wake_words"] + [name.lower()]:
            initial_text = initial_text.replace(w, "").strip()

        command = initial_text if len(initial_text) > 2 else self._listen_for_text(
            timeout=5, recognizer=recognizer
        )

        if command:
            self._handle_command(command)

    # ── Command dispatcher ────────────────────────────────────

    def _handle_command(self, text: str):
        text = text.lower().strip()
        print(f"📋  Command: \"{text}\"")

        # ── Add face ──────────────────────────────────────────
        if any(p in text for p in [
            "this is my friend", "this is", "remember this person",
            "add this person", "who is this", "save this face"
        ]):
            self._cmd_add_face(text)

        # ── Describe scene ────────────────────────────────────
        elif any(p in text for p in [
            "what do you see", "describe", "what's around",
            "what is in front", "surroundings", "where am i"
        ]):
            self._cmd_describe_scene()

        # ── Read text ─────────────────────────────────────────
        elif any(p in text for p in [
            "read", "what does it say", "read the sign",
            "read that", "what's written"
        ]):
            self._cmd_read_text()

        # ── Change name ───────────────────────────────────────
        elif any(p in text for p in [
            "change your name", "call you", "your name is",
            "rename yourself", "i want to call you"
        ]):
            self._cmd_change_name(text)

        # ── Re-enroll voice ───────────────────────────────────
        elif any(p in text for p in [
            "learn my voice", "remember my voice",
            "enroll my voice", "register my voice"
        ]):
            self.voice.speak("Sure, please say something after the beep.")
            self._enroll_owner_voice()
            self.voice.speak("Got it. I've updated your voice profile.")

        # ── Help ──────────────────────────────────────────────
        elif any(p in text for p in ["help", "what can you do", "commands"]):
            self._cmd_help()

        # ── Stop / mute ───────────────────────────────────────
        elif any(p in text for p in ["stop", "quiet", "mute", "be quiet"]):
            self.voice.speak("I'll stay quiet. Say my name when you need me.")
            self.voice.enabled = False
            threading.Timer(30.0, lambda: setattr(self.voice, "enabled", True)).start()

        else:
            self.voice.speak(
                f"Sorry, I didn't catch that. Say help to hear what I can do."
            )

    # ── Commands ──────────────────────────────────────────────

    def _cmd_add_face(self, text: str):
        """Captures current frame, extracts face, saves to known_faces."""
        import cv2

        # Try to extract a name from the command
        name = None
        for prefix in ["this is my friend ", "this is ", "call them ", "her name is ", "his name is "]:
            if prefix in text:
                name = text.split(prefix)[-1].strip().title()
                break

        if not name:
            self.voice.speak("What's their name?")
            spoken_name = self._listen_for_text(timeout=5)
            name = spoken_name.strip().title() if spoken_name else "Friend"

        frame = self._frame_fn()
        if frame is None:
            self.voice.speak("I can't see anything right now.")
            return

        os.makedirs("known_faces", exist_ok=True)
        path = f"known_faces/{name}.jpg"
        cv2.imwrite(path, frame)

        # Reload known faces
        if hasattr(self.face_id, "_load_known_faces"):
            self.face_id.known_names      = []
            self.face_id.known_encodings  = []
            self.face_id._load_known_faces()

        self.voice.speak(
            f"Got it! I'll remember {name}. "
            f"I'll let you know when I see them again."
        )
        print(f"✅  Face saved: known_faces/{name}.jpg")

    def _cmd_describe_scene(self):
        frame = self._frame_fn()
        if frame is None:
            self.voice.speak("I can't see anything right now.")
            return

        detections = self.detector.detect(frame)
        depth_map  = self.depth.estimate(frame)

        if not detections:
            self.voice.speak("The path looks clear. I don't see any obstacles.")
            return

        parts = []
        for i, d in enumerate(detections[:4]):
            dist = self.depth.get_distance_label(depth_map, d["box"], d["area_ratio"])
            zone = d["zone"]
            lbl  = d["label"]
            parts.append(f"{lbl} {zone}, {dist}")

        desc = ". ".join(parts)
        self.voice.speak(f"I can see: {desc}.", key="_scene_desc")

    def _cmd_read_text(self):
        frame = self._frame_fn()
        if frame is None:
            self.voice.speak("I can't see anything right now.")
            return

        texts = self.ocr.read(frame)
        msg   = self.ocr.build_message(texts)
        self.voice.speak(msg, key="_ocr_read")

    def _cmd_change_name(self, text: str):
        self.voice.speak("What would you like to call me?")
        new_name = self._listen_for_text(timeout=6)
        if new_name and len(new_name.strip()) > 1:
            clean = new_name.strip().title()
            self.config["assistant_name"] = clean
            self.config["wake_words"]     = [f"hey {clean.lower()}"]
            self._save_config()
            self.voice.speak(
                f"Done! You can now call me {clean}. "
                f"Just say Hey {clean} to get my attention."
            )
            print(f"✅  Name updated to: {clean}")
        else:
            self.voice.speak("Sorry, I didn't catch that. Name unchanged.")

    def _cmd_help(self):
        name = self.config["assistant_name"]
        self.voice.speak(
            f"Here's what I can do. "
            f"Say Hey {name} then: "
            f"describe surroundings. "
            f"Read the sign. "
            f"This is my friend, and their name. "
            f"Change your name. "
            f"Learn my voice. "
            f"Or just ask what you need."
        )

    # ── Utility ───────────────────────────────────────────────

    def _listen_for_text(self, timeout: int = 5, recognizer=None) -> str:
        """Listens for a single spoken phrase and returns transcribed text."""
        if self.sr is None:
            return ""
        try:
            rec = recognizer or self.sr.Recognizer()
            with self.mic as source:
                rec.adjust_for_ambient_noise(source, duration=0.5)
                audio = rec.listen(source, timeout=timeout, phrase_time_limit=6)
            return rec.recognize_google(audio).lower().strip()
        except Exception:
            return ""

    @property
    def name(self) -> str:
        return self.config["assistant_name"]

    @property
    def owner_name(self) -> str:
        return self.config["owner_name"]
