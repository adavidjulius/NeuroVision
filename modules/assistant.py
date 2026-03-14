"""
NeuroVision — Voice Assistant (Echo-free)
Mic mutes while Julius is speaking — no self-triggering.
"""

import os
import json
import time
import threading
import subprocess
import numpy as np

CONFIG_PATH = "assistant_config.json"
VOSK_MODEL  = "models/vosk-en"

DEFAULT_CONFIG = {
    "assistant_name":   "Julius",
    "wake_words":       ["hey julius", "julius"],
    "owner_enrolled":   False,
    "owner_name":       "David Julius",
    "response_style":   "concise",
    "confirm_commands": True,
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
        self.auto_mode  = False
        self._frame_fn  = None
        self._thread    = None
        self._muted     = False   # ← mic mute flag

        self._init_vosk()

        print(f"🎙️   Assistant : {self.config['assistant_name']}")
        print(f"   Wake word : Hey {self.config['assistant_name']}")
        print(f"   Owner     : {self.config['owner_name']}")

    def _load_config(self) -> dict:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)
            for k, v in DEFAULT_CONFIG.items():
                cfg.setdefault(k, v)
            return cfg
        return DEFAULT_CONFIG.copy()

    def _save_config(self):
        with open(CONFIG_PATH, "w") as f:
            json.dump(self.config, f, indent=2)

    def _init_vosk(self):
        try:
            import vosk, pyaudio
            if not os.path.exists(VOSK_MODEL):
                print(f"⚠️   Vosk model not found at {VOSK_MODEL}")
                self.vosk_ready = False
                return
            self.vosk_model = vosk.Model(VOSK_MODEL)
            self.pyaudio    = pyaudio
            self.vosk       = vosk
            self.vosk_ready = True
            print("✅  Vosk offline speech recognition ready")
        except Exception as e:
            print(f"⚠️   Vosk unavailable ({e})")
            self.vosk_ready = False

    # ── Speak (mutes mic while speaking) ──────────────────────

    def _speak_and_mute(self, text: str):
        """Speak text while mic is muted. Unmute after done."""
        self._muted = True
        print(f"🔊  {text}")
        try:
            subprocess.run(
                ["say", "-r", "200", text],
                timeout=30,
                capture_output=True
            )
        except Exception as e:
            print(f"⚠️   say error: {e}")
        finally:
            # Wait a little after speaking before listening again
            time.sleep(0.6)
            self._muted = False

    def say(self, text: str):
        """Speak in background thread — mic muted until done."""
        threading.Thread(
            target=self._speak_and_mute,
            args=(text,),
            daemon=True
        ).start()

    def say_now(self, text: str):
        """Speak and BLOCK until done — use for command responses."""
        self._speak_and_mute(text)

    # ── Listen loop ───────────────────────────────────────────

    def start_listening(self, frame_provider):
        if not self.vosk_ready:
            print("⚠️   Voice commands disabled")
            return
        self.listening = True
        self._frame_fn = frame_provider
        self._thread   = threading.Thread(
            target=self._listen_loop, daemon=True
        )
        self._thread.start()
        print(f"🎙️   Listening for: Hey {self.config['assistant_name']}\n")

    def stop_listening(self):
        self.listening = False

    def _listen_loop(self):
        p      = self.pyaudio.PyAudio()
        stream = p.open(
            format=self.pyaudio.paInt16,
            channels=1, rate=16000,
            input=True, frames_per_buffer=8192
        )
        rec = self.vosk.KaldiRecognizer(self.vosk_model, 16000)
        stream.start_stream()

        while self.listening:
            try:
                data = stream.read(4096, exception_on_overflow=False)

                # Skip processing while Julius is speaking
                if self._muted:
                    continue

                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text   = result.get("text", "").strip().lower()
                    if not text:
                        continue

                    print(f"🎤  Heard: '{text}'")

                    wake_words = self.config["wake_words"]
                    is_wake    = any(w in text for w in wake_words)

                    if is_wake:
                        command = text
                        for w in wake_words:
                            command = command.replace(w, "").strip()

                        if len(command) > 2:
                            # Inline command — handle directly
                            threading.Thread(
                                target=self._handle_command,
                                args=(command,),
                                daemon=True
                            ).start()
                        else:
                            # Wake only — say yes then listen
                            self.say_now("Yes?")
                            command = self._listen_once(
                                rec, stream, timeout=6
                            )
                            if command:
                                self._handle_command(command)
                            else:
                                self.say_now(
                                    "I didn't catch that."
                                )

            except Exception:
                time.sleep(0.2)

        stream.stop_stream()
        stream.close()
        p.terminate()

    def _listen_once(self, rec, stream, timeout=6) -> str:
        """
        Listen for one phrase after wake word.
        Mic is NOT muted here — we want to hear the command.
        """
        start = time.time()
        # Drain stale audio buffer first
        for _ in range(8):
            try:
                stream.read(4096, exception_on_overflow=False)
            except Exception:
                pass

        while time.time() - start < timeout:
            try:
                data = stream.read(4096, exception_on_overflow=False)
                if self._muted:
                    continue
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text   = result.get("text", "").strip().lower()
                    if text:
                        # Ignore if it's just the wake word again
                        wake_words = self.config["wake_words"]
                        clean = text
                        for w in wake_words:
                            clean = clean.replace(w, "").strip()
                        if len(clean) > 2:
                            print(f"🎤  Command: '{clean}'")
                            return clean
            except Exception:
                pass
        return ""

    # ── Command dispatcher ────────────────────────────────────

    def _handle_command(self, text: str):
        text = text.strip().lower()
        print(f"📋  Processing: '{text}'")

        if any(p in text for p in [
            "auto mode on", "start auto", "auto on",
            "navigation mode", "guide me", "start guiding"
        ]):
            self.auto_mode = True
            self.say_now("Auto mode on. I will announce everything I see.")

        elif any(p in text for p in [
            "auto mode off", "stop auto", "auto off",
            "stop guiding", "stop navigation", "manual mode"
        ]):
            self.auto_mode = False
            self.say_now("Auto mode off.")

        elif any(p in text for p in [
            "what do you see", "describe", "surroundings",
            "what around", "where am i", "what is in front",
            "what's there", "look around"
        ]):
            self._cmd_describe_scene()

        elif any(p in text for p in [
            "read", "what does it say", "read the sign",
            "read that", "what written", "read this",
            "read the book", "read the page", "read text",
            "read it", "what's written"
        ]):
            self._cmd_read_text()

        elif any(p in text for p in [
            "who is this", "who is that",
            "who is in front", "who do you see", "who's there"
        ]):
            self._cmd_who_is_this()

        elif any(p in text for p in [
            "this is my friend", "this is",
            "remember this", "add this person",
            "save this face", "remember her", "remember him"
        ]):
            self._cmd_add_face(text)

        elif any(p in text for p in [
            "change your name", "call you",
            "your name is", "rename"
        ]):
            self._cmd_change_name()

        elif any(p in text for p in [
            "stop", "quiet", "mute", "be quiet", "shut up"
        ]):
            self.auto_mode = False
            self.say_now("Going quiet for 30 seconds.")
            self.voice.enabled = False
            threading.Timer(
                30.0, lambda: setattr(self.voice, "enabled", True)
            ).start()

        elif any(p in text for p in [
            "help", "what can you do", "commands", "what do you do"
        ]):
            self._cmd_help()

        else:
            self.say_now(
                "Sorry, I didn't understand. Say help for commands."
            )

    # ── Auto announce (called from main loop) ─────────────────

    def auto_announce(self, detections: list, depth_labels: dict):
        if not self.auto_mode:
            return
        if not detections:
            self.voice.speak("Path clear.", key="_clear_path")
            return
        for i, d in enumerate(detections):
            dist = depth_labels.get(i, "")
            msg  = self.voice.build_message(
                d["label"], d["zone"], d["area_ratio"], dist
            )
            key  = f"auto_{d['label']}_{d['zone']}"
            self.voice.speak(msg, key=key)

    # ── Individual commands ───────────────────────────────────

    def _cmd_describe_scene(self):
        frame = self._frame_fn()
        if frame is None:
            self.say_now("I can't see anything.")
            return
        detections = self.detector.detect(frame)
        depth_map  = self.depth.estimate(frame)
        if not detections:
            self.say_now("The path looks clear. Nothing detected.")
            return
        parts = []
        for i, d in enumerate(detections[:5]):
            dist = self.depth.get_distance_label(
                depth_map, d["box"], d["area_ratio"],
                label=d["label"], frame=frame
            )
            parts.append(f"{d['label']} {d['zone']}, {dist}")
        self.say_now(". ".join(parts))

    def _cmd_read_text(self):
        frame = self._frame_fn()
        if frame is None:
            self.say_now("I can't see anything.")
            return
        self.say_now("Reading now.")
        time.sleep(0.3)
        texts = self.ocr.read(frame)
        if not texts:
            self.say_now("I don't see any text.")
            return
        full_text = " ".join([t["text"] for t in texts])
        print(f"📖  OCR: {full_text}")
        self.say_now(f"It says: {full_text}")

    def _cmd_who_is_this(self):
        frame = self._frame_fn()
        if frame is None:
            self.say_now("I can't see anyone.")
            return
        faces = self.face_id.identify(frame)
        if not faces:
            self.say_now("I don't recognise anyone in view.")
            return
        for f in faces:
            name = f["name"]
            if name != "Unknown":
                self.say_now(f"That's {name}.")
            else:
                self.say_now(
                    "I don't recognise this person. "
                    "Say hey Julius this is my friend, then their name."
                )

    def _cmd_add_face(self, text: str):
        import cv2
        name = None
        for prefix in [
            "this is my friend ", "this is ",
            "her name is ", "his name is ",
            "remember ", "call them "
        ]:
            if prefix in text:
                candidate = text.split(prefix)[-1].strip().title()
                if len(candidate) > 1:
                    name = candidate
                    break

        if not name:
            self.say_now("What is their name?")
            time.sleep(0.5)
            # Listen for name
            p      = self.pyaudio.PyAudio()
            stream = p.open(
                format=self.pyaudio.paInt16,
                channels=1, rate=16000,
                input=True, frames_per_buffer=8192
            )
            rec    = self.vosk.KaldiRecognizer(self.vosk_model, 16000)
            stream.start_stream()
            name   = self._listen_once(rec, stream, timeout=6).title() or "Friend"
            stream.stop_stream()
            stream.close()
            p.terminate()

        frame = self._frame_fn()
        if frame is None:
            self.say_now("I can't see anyone right now.")
            return
        os.makedirs("known_faces", exist_ok=True)
        path = f"known_faces/{name}.jpg"
        cv2.imwrite(path, frame)
        self.face_id.add_face(name)
        self.say_now(f"Got it! I saved {name}. I will tell you when I see them again.")
        print(f"✅  Face saved → {path}")

    def _cmd_change_name(self):
        self.say_now("What would you like to call me?")
        time.sleep(0.5)
        p      = self.pyaudio.PyAudio()
        stream = p.open(
            format=self.pyaudio.paInt16,
            channels=1, rate=16000,
            input=True, frames_per_buffer=8192
        )
        rec    = self.vosk.KaldiRecognizer(self.vosk_model, 16000)
        stream.start_stream()
        new_name = self._listen_once(rec, stream, timeout=6).title()
        stream.stop_stream()
        stream.close()
        p.terminate()

        if new_name:
            self.config["assistant_name"] = new_name
            self.config["wake_words"]     = [
                f"hey {new_name.lower()}", new_name.lower()
            ]
            self._save_config()
            self.say_now(f"Done! Call me {new_name} from now on.")
            print(f"✅  Name → {new_name}")

    def _cmd_help(self):
        name = self.config["assistant_name"]
        self.say_now(
            f"Here is what I can do. "
            f"Say Hey {name} auto mode on to start navigation. "
            f"Say Hey {name} what do you see to describe surroundings. "
            f"Say Hey {name} read that to read any text or book. "
            f"Say Hey {name} who is this to identify a person. "
            f"Say Hey {name} this is my friend name to save a face. "
            f"Say Hey {name} be quiet to mute me."
        )

    @property
    def name(self) -> str:
        return self.config["assistant_name"]

    @property
    def owner_name(self) -> str:
        return self.config["owner_name"]
