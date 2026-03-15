"""
NeuroVision — Sprint 4: Multi-language OCR
───────────────────────────────────────────
Reads text in:
  - English
  - Tamil (தமிழ்)
  - Hindi (हिन्दी)
Auto-detects language and reads aloud in correct language.
"""

import cv2
import numpy as np
import re


# Language configs
LANG_CONFIGS = {
    "english": {
        "code":    ["en"],
        "name":    "English",
        "voice":   "en",
        "flag":    "🇬🇧",
    },
    "tamil": {
        "code":    ["ta"],
        "name":    "Tamil",
        "voice":   "ta-IN",
        "flag":    "🇮🇳",
    },
    "hindi": {
        "code":    ["hi"],
        "name":    "Hindi",
        "voice":   "hi-IN",
        "flag":    "🇮🇳",
    },
    "multi": {
        "code":    ["en", "ta", "hi"],
        "name":    "Multi-language",
        "voice":   "en",
        "flag":    "🌐",
    },
}

# Unicode ranges for language detection
TAMIL_RANGE  = (0x0B80, 0x0BFF)
HINDI_RANGE  = (0x0900, 0x097F)


class MultiLangOCR:
    def __init__(self):
        self.ready        = False
        self.reader_en    = None
        self.reader_ta    = None
        self.reader_hi    = None
        self.reader_multi = None
        self.current_lang = "multi"
        self._init()

    def _init(self):
        try:
            import easyocr
            print("📖  Loading Multi-language OCR models...")
            print("   Loading English...")
            self.reader_en    = easyocr.Reader(
                ["en"], gpu=False, verbose=False
            )
            print("   Loading Tamil...")
            self.reader_ta    = easyocr.Reader(
                ["en", "ta"], gpu=False, verbose=False
            )
            print("   Loading Hindi...")
            self.reader_hi    = easyocr.Reader(
                ["en", "hi"], gpu=False, verbose=False
            )
            self.reader_multi = self.reader_ta  # Tamil+English covers most
            self.ready        = True
            print("✅  Multi-language OCR ready (EN + TA + HI)")
        except Exception as e:
            print(f"⚠️   Multi-language OCR unavailable ({e})")
            self.ready = False

    def detect_language(self, text: str) -> str:
        """Auto-detects language from unicode character ranges."""
        tamil_count = sum(
            1 for c in text
            if TAMIL_RANGE[0] <= ord(c) <= TAMIL_RANGE[1]
        )
        hindi_count = sum(
            1 for c in text
            if HINDI_RANGE[0] <= ord(c) <= HINDI_RANGE[1]
        )

        if tamil_count > 2:
            return "tamil"
        elif hindi_count > 2:
            return "hindi"
        return "english"

    def read(self, frame: np.ndarray,
             language: str = "auto") -> list:
        """
        Read text from frame in specified language.
        language: 'auto' | 'english' | 'tamil' | 'hindi' | 'multi'
        Returns list of { text, confidence, box, language }
        """
        if not self.ready:
            return []

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Pick reader
        if language == "tamil":
            reader = self.reader_ta
        elif language == "hindi":
            reader = self.reader_hi
        elif language == "english":
            reader = self.reader_en
        else:
            # auto or multi — use multi reader
            reader = self.reader_multi

        try:
            raw_results = reader.readtext(rgb)
        except Exception as e:
            print(f"⚠️   OCR error: {e}")
            return []

        output = []
        for (bbox, text, conf) in raw_results:
            if conf < 0.35 or len(text.strip()) < 1:
                continue

            pts      = np.array(bbox, dtype=np.int32)
            detected = self.detect_language(text)

            output.append({
                "text":       text.strip(),
                "confidence": round(conf, 2),
                "box":        pts,
                "language":   detected,
                "flag":       LANG_CONFIGS[detected]["flag"],
            })

        return output

    def read_all_languages(self, frame: np.ndarray) -> list:
        """
        Runs all three language readers and merges results.
        Use when you don't know what language is in the frame.
        """
        if not self.ready:
            return []

        all_results = []
        seen_texts  = set()

        for lang, reader in [
            ("english", self.reader_en),
            ("tamil",   self.reader_ta),
            ("hindi",   self.reader_hi),
        ]:
            try:
                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = reader.readtext(rgb)
                for (bbox, text, conf) in results:
                    clean = text.strip()
                    if conf < 0.35 or len(clean) < 1:
                        continue
                    if clean.lower() in seen_texts:
                        continue
                    seen_texts.add(clean.lower())
                    pts      = np.array(bbox, dtype=np.int32)
                    detected = self.detect_language(clean)
                    all_results.append({
                        "text":       clean,
                        "confidence": round(conf, 2),
                        "box":        pts,
                        "language":   detected,
                        "flag":       LANG_CONFIGS[detected]["flag"],
                    })
            except Exception:
                continue

        return all_results

    def build_message(self, texts: list) -> str:
        """Builds voice message grouping by language."""
        if not texts:
            return "No text detected"

        # Group by language
        by_lang = {}
        for t in texts:
            lang = t["language"]
            by_lang.setdefault(lang, []).append(t["text"])

        parts = []
        for lang, words in by_lang.items():
            flag = LANG_CONFIGS[lang]["flag"]
            combined = " ".join(words[:6])
            if lang == "english":
                parts.append(f"English text: {combined}")
            elif lang == "tamil":
                parts.append(f"Tamil text detected: {combined}")
            elif lang == "hindi":
                parts.append(f"Hindi text detected: {combined}")

        return ". ".join(parts)

    def draw(self, frame: np.ndarray,
             texts: list) -> np.ndarray:
        """Draws OCR results with language color coding."""
        lang_colors = {
            "english": (255, 200, 0),    # yellow
            "tamil":   (0,   255, 150),  # green
            "hindi":   (255, 100, 100),  # red
        }

        for t in texts:
            pts   = t["box"]
            text  = t["text"]
            conf  = t["confidence"]
            lang  = t["language"]
            flag  = t["flag"]
            color = lang_colors.get(lang, (200, 200, 200))

            cv2.polylines(
                frame, [pts],
                isClosed=True, color=color, thickness=2
            )

            x, y   = pts[0]
            label  = f"{flag} {text} ({conf:.0%})"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                frame,
                (x, y - th - 8), (x + tw + 6, y),
                color, -1
            )
            cv2.putText(
                frame, label, (x + 3, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1
            )

        return frame

    def set_language(self, lang: str):
        """Switch active language."""
        if lang in LANG_CONFIGS:
            self.current_lang = lang
            print(f"🌐  OCR language set to: {lang}")
