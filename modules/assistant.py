"""
NeuroVision — Voice Assistant (Non-blocking)
Commands run in separate thread — never block listener.
"""
import os, json, time, threading, subprocess
import numpy as np

CONFIG_PATH = "assistant_config.json"
VOSK_MODEL  = "models/vosk-en"

DEFAULT_CONFIG = {
    "assistant_name":   "Julius",
    "wake_words":       ["hey julius", "julius"],
    "owner_name":       "David Julius",
    "confirm_commands": True,
}


class VoiceAssistant:
    def __init__(self, voice, detector, face_id, ocr, depth):
        self.voice     = voice
        self.detector  = detector
        self.face_id   = face_id
        self.ocr       = ocr
        self.depth     = depth
        self.config    = self._load_config()
        self.listening = False
        self.auto_mode = False
        self._frame_fn = None
        self._muted    = False
        self._init_vosk()

        print(f"🎙️   Assistant : {self.name}")
        print(f"   Wake word : Hey {self.name}")

    def _load_config(self):
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)
            for k,v in DEFAULT_CONFIG.items():
                cfg.setdefault(k,v)
            return cfg
        return DEFAULT_CONFIG.copy()

    def _save_config(self):
        with open(CONFIG_PATH,"w") as f:
            json.dump(self.config,f,indent=2)

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
            print("✅  Vosk offline STT ready")
        except Exception as e:
            print(f"⚠️   Vosk unavailable ({e})")
            self.vosk_ready = False

    def _say(self, text: str):
        """Speak and mute mic while speaking."""
        self._muted = True
        try:
            subprocess.run(
                ["say", "-r", "185", text],
                timeout=20, capture_output=True
            )
        except Exception:
            pass
        finally:
            time.sleep(0.7)
            self._muted = False

    def say(self, text: str):
        """Non-blocking speak."""
        print(f"🔊  {text}")
        threading.Thread(
            target=self._say, args=(text,), daemon=True
        ).start()

    def say_wait(self, text: str):
        """Blocking speak — waits until done."""
        print(f"🔊  {text}")
        self._say(text)

    def start_listening(self, frame_provider):
        if not self.vosk_ready:
            print("⚠️   Voice commands disabled")
            return
        self.listening = True
        self._frame_fn = frame_provider
        threading.Thread(
            target=self._listen_loop,
            daemon=True
        ).start()
        print(f"🎙️   Listening for: Hey {self.name}\n")

    def stop_listening(self):
        self.listening = False

    def _listen_loop(self):
        import json as _json
        p      = self.pyaudio.PyAudio()
        stream = p.open(
            format=self.pyaudio.paInt16,
            channels=1, rate=16000,
            input=True, frames_per_buffer=8192
        )
        rec = self.vosk.KaldiRecognizer(
            self.vosk_model, 16000
        )
        stream.start_stream()

        while self.listening:
            try:
                data = stream.read(
                    4096, exception_on_overflow=False
                )
                if self._muted:
                    continue
                if rec.AcceptWaveform(data):
                    result = _json.loads(rec.Result())
                    text   = result.get("text","").strip().lower()
                    if not text:
                        continue

                    print(f"🎤  Heard: '{text}'")
                    wake = self.config["wake_words"]

                    if any(w in text for w in wake):
                        cmd = text
                        for w in wake:
                            cmd = cmd.replace(w,"").strip()

                        if len(cmd) > 2:
                            # Inline command
                            threading.Thread(
                                target=self._handle_command,
                                args=(cmd,), daemon=True
                            ).start()
                        else:
                            self.say_wait("Yes?")
                            cmd = self._listen_once(
                                rec, stream, timeout=6
                            )
                            if cmd:
                                threading.Thread(
                                    target=self._handle_command,
                                    args=(cmd,), daemon=True
                                ).start()
                            else:
                                self.say("I didn't catch that.")

            except Exception:
                time.sleep(0.2)

        stream.stop_stream()
        stream.close()
        p.terminate()

    def _listen_once(self, rec, stream,
                     timeout=6) -> str:
        import json as _json
        # Drain buffer
        for _ in range(10):
            try:
                stream.read(4096, exception_on_overflow=False)
            except Exception:
                pass

        start = time.time()
        while time.time() - start < timeout:
            try:
                data = stream.read(
                    4096, exception_on_overflow=False
                )
                if self._muted:
                    continue
                if rec.AcceptWaveform(data):
                    result = _json.loads(rec.Result())
                    text   = result.get("text","").strip().lower()
                    if len(text) > 2:
                        # Strip wake word if repeated
                        for w in self.config["wake_words"]:
                            text = text.replace(w,"").strip()
                        if len(text) > 2:
                            print(f"🎤  Command: '{text}'")
                            return text
            except Exception:
                pass
        return ""

    def _handle_command(self, text: str):
        text = text.strip().lower()
        print(f"📋  Command: '{text}'")

        if any(p in text for p in [
            "auto mode on","start auto","auto on",
            "guide me","navigation mode"
        ]):
            self.auto_mode = True
            self.say_wait("Auto mode on. Announcing everything.")

        elif any(p in text for p in [
            "auto mode off","stop auto","auto off",
            "stop guiding","manual"
        ]):
            self.auto_mode = False
            self.say_wait("Auto mode off.")

        elif any(p in text for p in [
            "what do you see","describe","surroundings",
            "what around","where am i","look around",
            "what is in front","what's there"
        ]):
            self._cmd_describe()

        elif any(p in text for p in [
            "read","what does it say","read the sign",
            "read that","read this","what's written",
            "read the book","read text","read it"
        ]):
            self._cmd_read()

        elif any(p in text for p in [
            "who is this","who is that",
            "who do you see","who's there"
        ]):
            self._cmd_who()

        elif any(p in text for p in [
            "this is my friend","this is",
            "remember","save this face"
        ]):
            self._cmd_add_face(text)

        elif any(p in text for p in [
            "help","what can you do","commands",
            "what do you do"
        ]):
            self._cmd_help()

        elif any(p in text for p in [
            "stop","quiet","mute","be quiet","shut up"
        ]):
            self.auto_mode = False
            self.say_wait("Going quiet for 30 seconds.")
            self.voice.enabled = False
            threading.Timer(
                30.0,
                lambda: setattr(self.voice,"enabled",True)
            ).start()

        else:
            self.say_wait(
                "I didn't understand. Say help for commands."
            )

    def auto_announce(self, detections, depth_labels):
        if not self.auto_mode or self._muted:
            return
        if not detections:
            self.voice.speak("Path clear.", key="_clear", cooldown=8.0)
            return
        top  = max(detections, key=lambda d: d["area_ratio"])
        i    = detections.index(top)
        dist = depth_labels.get(i,"")
        msg  = self.voice.build_message(
            top["label"], top["zone"],
            top["area_ratio"], dist
        )
        self.voice.speak(
            msg, key=f"auto_{top['label']}_{top['zone']}",
            cooldown=4.0
        )

    def _cmd_describe(self):
        frame = self._frame_fn()
        if frame is None:
            self.say_wait("I can't see anything.")
            return
        dets  = self.detector.detect(frame)
        dm    = self.depth.estimate(frame)
        if not dets:
            self.say_wait("Path looks clear. Nothing detected.")
            return
        parts = []
        for i,d in enumerate(dets[:4]):
            dist = self.depth.get_distance_label(
                dm, d["box"], d["area_ratio"],
                label=d["label"], frame=frame
            )
            parts.append(f"{d['label']} {d['zone']} {dist}")
        self.say_wait(". ".join(parts))

    def _cmd_read(self):
        frame = self._frame_fn()
        if frame is None:
            self.say_wait("I can't see anything.")
            return
        self.say_wait("Reading now.")
        time.sleep(0.3)
        texts = self.ocr.read(frame)
        if not texts:
            self.say_wait("No text detected.")
            return
        full = " ".join([t["text"] for t in texts])
        print(f"📖  OCR: {full}")
        self.say_wait(f"It says: {full}")

    def _cmd_who(self):
        frame = self._frame_fn()
        if frame is None:
            self.say_wait("I can't see anyone.")
            return
        faces = self.face_id.identify(frame)
        if not faces:
            self.say_wait("I don't recognise anyone.")
            return
        for f in faces:
            name = f["name"]
            if name != "Unknown":
                self.say_wait(f"That's {name}.")
            else:
                self.say_wait(
                    "I don't recognise this person. "
                    "Say hey julius this is my friend "
                    "and their name to save them."
                )

    def _cmd_add_face(self, text: str):
        import cv2
        name = None
        for prefix in [
            "this is my friend ","this is ",
            "her name is ","his name is ","remember "
        ]:
            if prefix in text:
                candidate = text.split(prefix)[-1].strip().title()
                if len(candidate) > 1:
                    name = candidate
                    break

        if not name:
            self.say_wait("What is their name?")
            # Listen for name
            p      = self.pyaudio.PyAudio()
            stream = p.open(
                format=self.pyaudio.paInt16,
                channels=1, rate=16000,
                input=True, frames_per_buffer=8192
            )
            rec  = self.vosk.KaldiRecognizer(
                self.vosk_model, 16000)
            stream.start_stream()
            name = self._listen_once(
                rec, stream, timeout=6
            ).title() or "Friend"
            stream.stop_stream()
            stream.close()
            p.terminate()

        frame = self._frame_fn()
        if frame is None:
            self.say_wait("I can't see anyone.")
            return

        os.makedirs("known_faces", exist_ok=True)
        path = f"known_faces/{name}.jpg"
        cv2.imwrite(path, frame)
        self.face_id.add_face(name)
        self.say_wait(
            f"Got it! I saved {name}. "
            f"I will tell you when I see them."
        )
        print(f"✅  Face saved → {path}")

    def _cmd_help(self):
        n = self.name
        self.say_wait(
            f"Here is what I can do. "
            f"Say Hey {n} auto mode on for navigation. "
            f"Say Hey {n} what do you see to describe. "
            f"Say Hey {n} read that to read any text. "
            f"Say Hey {n} who is this to identify someone. "
            f"Say Hey {n} this is my friend name to save. "
            f"Say Hey {n} be quiet to mute."
        )

    @property
    def name(self):
        return self.config["assistant_name"]

    @property
    def owner_name(self):
        return self.config["owner_name"]
