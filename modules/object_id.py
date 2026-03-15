"""
NeuroVision — Sprint 3: Object Identification
──────────────────────────────────────────────
Identifies specific objects:
  - Medicine bottles
  - Currency notes
  - Food labels
  - General objects
Uses YOLO + OCR combined for rich identification.
"""

import cv2
import numpy as np
import re


# ── Object categories ─────────────────────────────────────────
MEDICINE_KEYWORDS = [
    "mg", "ml", "tablet", "capsule", "syrup", "dose",
    "medicine", "pharmacy", "rx", "prescription",
    "paracetamol", "ibuprofen", "amoxicillin", "aspirin",
    "take", "daily", "twice", "thrice", "warning",
]

CURRENCY_PATTERNS = {
    "INR": [
        (r"(?i)rupee|₹|inr",           "Indian Rupee"),
        (r"\b(10|20|50|100|200|500|2000)\b", "denomination"),
    ],
    "USD": [
        (r"(?i)dollar|\$|usd",          "US Dollar"),
        (r"\b(1|5|10|20|50|100)\b",     "denomination"),
    ],
    "EUR": [
        (r"(?i)euro|€|eur",             "Euro"),
    ],
}

FOOD_KEYWORDS = [
    "ingredients", "nutrition", "calories", "protein",
    "carbohydrate", "fat", "sugar", "sodium", "allergen",
    "contains", "wheat", "milk", "nuts", "gluten",
    "serving", "per 100g", "energy",
]

# YOLO classes that map to specific categories
MEDICINE_CLASSES  = {"bottle", "cup"}
CURRENCY_CLASSES  = {"book"}  # bills detected as flat objects
FOOD_CLASSES      = {"bottle", "cup", "bowl", "sandwich", "apple",
                     "orange", "banana", "broccoli", "carrot", "pizza"}


class ObjectIdentifier:
    def __init__(self, ocr_module=None):
        self.ocr   = ocr_module
        self.ready = True
        print("✅  Object identifier ready")

    # ── Category detection ────────────────────────────────────

    def identify_from_text(self, texts: list) -> dict:
        """
        Analyses OCR text to identify object category.
        Returns { category, label, details, warning }
        """
        if not texts:
            return None

        full_text = " ".join([t["text"].lower() for t in texts])
        words     = full_text.split()

        # ── Medicine check ────────────────────────────────────
        med_score = sum(1 for kw in MEDICINE_KEYWORDS if kw in full_text)
        if med_score >= 2:
            return self._identify_medicine(full_text, texts)

        # ── Currency check ────────────────────────────────────
        for currency, patterns in CURRENCY_PATTERNS.items():
            for pattern, label in patterns:
                if re.search(pattern, full_text):
                    return self._identify_currency(full_text, currency)

        # ── Food check ────────────────────────────────────────
        food_score = sum(1 for kw in FOOD_KEYWORDS if kw in full_text)
        if food_score >= 2:
            return self._identify_food(full_text, texts)

        # ── General object ────────────────────────────────────
        if texts:
            main_text = texts[0]["text"]
            return {
                "category": "general",
                "label":    f"Object with text: {main_text}",
                "details":  full_text[:100],
                "warning":  None,
                "icon":     "📦",
            }

        return None

    def _identify_medicine(self, text: str, texts: list) -> dict:
        """Extracts medicine name, dosage, warnings."""
        # Find dosage pattern (e.g. 500mg, 10ml)
        dosage_match = re.search(r"(\d+\s*(?:mg|ml|mcg|g))", text, re.I)
        dosage       = dosage_match.group(1) if dosage_match else ""

        # Find medicine name (usually first or second text item)
        name = texts[0]["text"] if texts else "Medicine"

        # Check for warnings
        warning = None
        warn_kw = ["warning", "caution", "danger", "poison",
                   "keep out", "do not", "avoid"]
        if any(kw in text for kw in warn_kw):
            warning = "Warning label detected on medicine"

        return {
            "category": "medicine",
            "label":    f"Medicine: {name}",
            "details":  f"Dosage: {dosage}" if dosage else "Read label carefully",
            "warning":  warning,
            "icon":     "💊",
            "voice":    f"Medicine detected. {name}. {dosage}. {warning or ''}".strip()
        }

    def _identify_currency(self, text: str, currency: str) -> dict:
        """Identifies currency denomination."""
        # Find denomination
        if currency == "INR":
            match = re.search(r"\b(10|20|50|100|200|500|2000)\b", text)
            denom = f"₹{match.group(1)}" if match else "Indian Rupee"
            voice = f"Currency detected. {denom} note."
        elif currency == "USD":
            match = re.search(r"\b(1|5|10|20|50|100)\b", text)
            denom = f"${match.group(1)}" if match else "US Dollar"
            voice = f"Currency detected. {denom} bill."
        else:
            denom = currency
            voice = f"Currency detected. {currency}."

        return {
            "category": "currency",
            "label":    f"Currency: {denom}",
            "details":  f"{currency} note",
            "warning":  None,
            "icon":     "💵",
            "voice":    voice,
        }

    def _identify_food(self, text: str, texts: list) -> dict:
        """Identifies food item and checks for allergens."""
        allergens = []
        allergen_list = ["wheat", "milk", "nuts", "peanut", "soy",
                         "egg", "fish", "shellfish", "gluten"]
        for allergen in allergen_list:
            if allergen in text:
                allergens.append(allergen)

        # Find calories
        cal_match = re.search(r"(\d+)\s*(?:cal|kcal|calories)", text, re.I)
        calories  = f"{cal_match.group(1)} calories" if cal_match else ""

        name    = texts[0]["text"] if texts else "Food item"
        warning = f"Contains allergens: {', '.join(allergens)}" if allergens else None

        return {
            "category": "food",
            "label":    f"Food: {name}",
            "details":  calories or "Food item detected",
            "warning":  warning,
            "icon":     "🍎",
            "voice":    f"Food item. {name}. {warning or calories or ''}".strip()
        }

    # ── Main identify function ────────────────────────────────

    def identify(self, frame: np.ndarray,
                 detections: list = None) -> list:
        """
        Identifies objects in frame using OCR + YOLO labels.
        Returns list of identification results.
        """
        results = []

        if self.ocr is None or not self.ocr.ready:
            return results

        # Run OCR on full frame
        texts = self.ocr.read(frame)
        if texts:
            result = self.identify_from_text(texts)
            if result:
                result["box"] = None  # full frame result
                results.append(result)

        # Also check YOLO detections for category hints
        if detections:
            for d in detections:
                label = d["label"]
                if label in MEDICINE_CLASSES and not any(
                    r["category"] == "medicine" for r in results
                ):
                    results.append({
                        "category": "possible_medicine",
                        "label":    "Possible medicine bottle",
                        "details":  "Point camera closer to read label",
                        "warning":  None,
                        "icon":     "💊",
                        "box":      d["box"],
                        "voice":    "Medicine bottle detected. Move closer to read label.",
                    })

                elif label in FOOD_CLASSES and not any(
                    r["category"] == "food" for r in results
                ):
                    results.append({
                        "category": "food",
                        "label":    f"Food item: {label}",
                        "details":  label,
                        "warning":  None,
                        "icon":     "🍎",
                        "box":      d["box"],
                        "voice":    f"Food item detected. {label}.",
                    })

        return results

    # ── Draw ──────────────────────────────────────────────────

    def draw(self, frame: np.ndarray, results: list) -> np.ndarray:
        h, w = frame.shape[:2]

        for i, r in enumerate(results):
            icon     = r.get("icon", "📦")
            label    = r["label"]
            details  = r.get("details", "")
            warning  = r.get("warning")
            box      = r.get("box")

            # Color by category
            color_map = {
                "medicine":          (0, 140, 255),
                "possible_medicine": (0, 200, 255),
                "currency":          (0, 220, 100),
                "food":              (255, 180, 0),
                "general":           (200, 200, 200),
            }
            color = color_map.get(r["category"], (200, 200, 200))

            if box:
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                # Full frame — show at top
                y_pos = 55 + i * 55
                cv2.rectangle(frame, (10, y_pos - 20),
                              (w - 10, y_pos + 30), color, 2)
                cv2.putText(frame, f"{label}",
                            (16, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
                if details:
                    cv2.putText(frame, details,
                                (16, y_pos + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (200, 200, 200), 1)

            # Warning banner
            if warning:
                cv2.rectangle(frame, (0, h - 70), (w, h - 35),
                              (0, 0, 200), -1)
                cv2.putText(frame, f"⚠ {warning}",
                            (10, h - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)

        return frame

    def build_voice_message(self, results: list) -> str:
        """Builds a combined voice message for all identified objects."""
        if not results:
            return ""
        messages = []
        for r in results:
            msg = r.get("voice") or r["label"]
            messages.append(msg)
            if r.get("warning"):
                messages.append(r["warning"])
        return ". ".join(messages)
