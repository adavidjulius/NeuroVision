"""
NeuroVision — Sprint 5: Indoor Navigation
───────────────────────────────────────────
Features:
  - Room detection (kitchen, corridor, bedroom etc.)
  - Layout memory (remembers rooms between sessions)
  - Landmark tracking (door, stairs, furniture positions)
  - Navigation instructions (turn left, go straight)
  - Room mapping from object detections
"""

import cv2
import json
import time
import os
import numpy as np
from collections import defaultdict, Counter


MAP_FILE = "indoor_map.json"

# Room classification based on detected objects
ROOM_SIGNATURES = {
    "kitchen": {
        "objects":  ["microwave", "oven", "refrigerator",
                     "sink", "bottle", "cup", "bowl",
                     "knife", "fork", "spoon"],
        "min_score": 2,
    },
    "bedroom": {
        "objects":  ["bed", "pillow", "lamp", "clock",
                     "laptop", "tv", "chair"],
        "min_score": 2,
    },
    "living_room": {
        "objects":  ["couch", "tv", "remote", "chair",
                     "potted plant", "book", "laptop"],
        "min_score": 2,
    },
    "bathroom": {
        "objects":  ["sink", "toilet", "toothbrush",
                     "bottle", "cup"],
        "min_score": 2,
    },
    "corridor": {
        "objects":  ["door", "person"],
        "min_score": 1,
        "aspect_hint": "narrow",
    },
    "office": {
        "objects":  ["laptop", "chair", "book",
                     "keyboard", "mouse", "monitor", "tv"],
        "min_score": 2,
    },
    "dining_room": {
        "objects":  ["dining table", "chair", "cup",
                     "bowl", "fork", "knife", "spoon"],
        "min_score": 2,
    },
}

# Navigation landmarks
LANDMARKS = {
    "door":          "door",
    "stairs":        "stairs",
    "chair":         "seating",
    "couch":         "seating",
    "dining table":  "table",
    "bed":           "bed",
    "refrigerator":  "kitchen appliance",
    "tv":            "screen",
    "potted plant":  "plant",
    "sink":          "sink",
}


class IndoorNavigator:
    def __init__(self):
        self.map_data       = self._load_map()
        self.current_room   = "unknown"
        self.room_history   = []
        self.landmark_memory = {}
        self.object_history = defaultdict(list)
        self.last_nav_time  = 0
        self.frame_count    = 0
        print("✅  Indoor navigator ready")
        if self.map_data["rooms"]:
            print(f"   📍 Loaded map with "
                  f"{len(self.map_data['rooms'])} known rooms")

    # ── Map persistence ───────────────────────────────────────

    def _load_map(self) -> dict:
        default = {
            "rooms":     {},
            "landmarks": {},
            "last_room": "unknown",
            "visits":    {},
        }
        if os.path.exists(MAP_FILE):
            try:
                with open(MAP_FILE) as f:
                    data = json.load(f)
                print(f"📍  Indoor map loaded from {MAP_FILE}")
                return data
            except Exception:
                pass
        return default

    def _save_map(self):
        try:
            with open(MAP_FILE, "w") as f:
                json.dump(self.map_data, f, indent=2)
        except Exception as e:
            print(f"⚠️   Map save failed: {e}")

    # ── Room detection ────────────────────────────────────────

    def detect_room(self, detections: list) -> str:
        """
        Classifies current room based on detected objects.
        Returns room type string.
        """
        if not detections:
            return self.current_room

        detected_labels = [d["label"] for d in detections]
        scores          = {}

        for room, sig in ROOM_SIGNATURES.items():
            score = sum(
                1 for obj in sig["objects"]
                if obj in detected_labels
            )
            if score >= sig["min_score"]:
                scores[room] = score

        if scores:
            best_room = max(scores, key=scores.get)
            return best_room

        return self.current_room or "unknown"

    # ── Landmark tracking ─────────────────────────────────────

    def update_landmarks(self, detections: list,
                         frame_shape: tuple):
        """Tracks landmark positions in current room."""
        h, w = frame_shape[:2]
        room = self.current_room

        if room not in self.map_data["landmarks"]:
            self.map_data["landmarks"][room] = {}

        for d in detections:
            label = d["label"]
            if label not in LANDMARKS:
                continue

            # Normalize position to 0-1
            cx_norm = d["cx"] / w
            cy_norm = d["cy"] / h

            landmark_key = f"{label}_{d['zone']}"
            self.map_data["landmarks"][room][landmark_key] = {
                "type":     LANDMARKS[label],
                "label":    label,
                "zone":     d["zone"],
                "position": [cx_norm, cy_norm],
                "seen_at":  time.time(),
            }

    # ── Navigation guidance ───────────────────────────────────

    def get_navigation_guidance(self,
                                 detections: list,
                                 depth_labels: dict) -> list:
        """
        Returns navigation instructions based on scene.
        """
        instructions = []
        now          = time.time()

        if not detections:
            instructions.append({
                "type":    "clear",
                "message": "Path ahead is clear",
                "zone":    "center",
            })
            return instructions

        # Check for navigable path
        zones_blocked = {d["zone"] for d in detections
                         if d["area_ratio"] > 0.05}

        if "center" not in zones_blocked:
            instructions.append({
                "type":    "go_straight",
                "message": "Path clear, go straight",
                "zone":    "center",
            })
        elif "left" not in zones_blocked:
            instructions.append({
                "type":    "turn_left",
                "message": "Turn left, path is clear",
                "zone":    "left",
            })
        elif "right" not in zones_blocked:
            instructions.append({
                "type":    "turn_right",
                "message": "Turn right, path is clear",
                "zone":    "right",
            })
        else:
            instructions.append({
                "type":    "blocked",
                "message": "Path blocked on all sides, stop",
                "zone":    "center",
            })

        # Add door detection
        doors = [d for d in detections if d["label"] == "door"]
        for door in doors:
            dist = depth_labels.get(
                detections.index(door), "unknown distance"
            )
            instructions.append({
                "type":    "landmark",
                "message": f"Door {door['zone']}, {dist}",
                "zone":    door["zone"],
            })

        # Add stairs warning
        stairs = [d for d in detections if d["label"] == "stairs"]
        for stair in stairs:
            instructions.append({
                "type":    "warning",
                "message": f"Stairs detected {stair['zone']}, caution",
                "zone":    stair["zone"],
            })

        return instructions

    # ── Room memory ───────────────────────────────────────────

    def update(self, detections: list,
               depth_labels: dict,
               frame_shape: tuple):
        """
        Main update — call every frame.
        Updates room, landmarks, and navigation.
        """
        self.frame_count += 1

        # Detect room every 30 frames
        if self.frame_count % 30 == 0:
            new_room = self.detect_room(detections)
            if new_room != self.current_room:
                self._on_room_change(new_room)
            self.current_room = new_room

            # Update visit count
            visits = self.map_data["visits"]
            visits[new_room] = visits.get(new_room, 0) + 1

        # Update landmarks every 10 frames
        if self.frame_count % 10 == 0:
            self.update_landmarks(detections, frame_shape)

        # Save map every 5 minutes
        if self.frame_count % 9000 == 0:
            self._save_map()

        return self.get_navigation_guidance(
            detections, depth_labels
        )

    def _on_room_change(self, new_room: str):
        """Called when user moves to a new room."""
        old_room = self.current_room
        self.room_history.append({
            "from": old_room,
            "to":   new_room,
            "time": time.time(),
        })

        # Remember room in map
        if new_room not in self.map_data["rooms"]:
            self.map_data["rooms"][new_room] = {
                "first_visit": time.strftime("%Y-%m-%d %H:%M"),
                "visit_count": 0,
            }
        self.map_data["rooms"][new_room]["visit_count"] += 1
        self.map_data["last_room"] = new_room
        self._save_map()

        print(f"📍  Room changed: {old_room} → {new_room}")

    # ── Memory queries ────────────────────────────────────────

    def get_known_rooms(self) -> list:
        return list(self.map_data["rooms"].keys())

    def get_room_landmarks(self, room: str = None) -> dict:
        room = room or self.current_room
        return self.map_data["landmarks"].get(room, {})

    def describe_current_location(self) -> str:
        """Returns a voice-friendly description of current location."""
        room      = self.current_room
        landmarks = self.get_room_landmarks()
        visits    = self.map_data["visits"].get(room, 0)

        if room == "unknown":
            return "I cannot determine your current location."

        room_name = room.replace("_", " ").title()
        desc      = f"You are in the {room_name}."

        if visits > 1:
            desc += f" You have been here {visits} times."

        if landmarks:
            items = list(landmarks.keys())[:3]
            names = [v["label"] for k, v in
                     list(landmarks.items())[:3]]
            desc += f" I can see {', '.join(names)} nearby."

        return desc

    def describe_map(self) -> str:
        """Describes the full known map."""
        rooms = self.get_known_rooms()
        if not rooms:
            return "No rooms mapped yet. Walk around to build the map."

        room_names = [r.replace("_", " ") for r in rooms]
        return (f"I have mapped {len(rooms)} areas: "
                f"{', '.join(room_names)}.")

    # ── Draw ──────────────────────────────────────────────────

    def draw(self, frame: np.ndarray,
             instructions: list) -> np.ndarray:
        h, w = frame.shape[:2]

        # Room label top center
        room_name = self.current_room.replace("_", " ").title()
        label     = f"📍 {room_name}"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        cx = (w - tw) // 2
        cv2.rectangle(frame,
                      (cx - 8, 50), (cx + tw + 8, 82),
                      (20, 40, 80), -1)
        cv2.putText(frame, label, (cx, 74),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 229, 255), 2)

        # Navigation arrow
        if instructions:
            inst = instructions[0]
            nav_type = inst["type"]
            msg      = inst["message"]

            color = {
                "go_straight": (0, 255, 100),
                "turn_left":   (0, 229, 255),
                "turn_right":  (0, 229, 255),
                "blocked":     (0, 0, 255),
                "landmark":    (255, 200, 0),
                "warning":     (0, 100, 255),
                "clear":       (0, 255, 100),
            }.get(nav_type, (200, 200, 200))

            # Arrow direction
            mid_x, mid_y = w // 2, h // 2
            if nav_type == "go_straight" or nav_type == "clear":
                # Up arrow
                pts = np.array([
                    [mid_x,      mid_y - 60],
                    [mid_x - 25, mid_y - 20],
                    [mid_x + 25, mid_y - 20],
                ], np.int32)
                cv2.fillPoly(frame, [pts], color)

            elif nav_type == "turn_left":
                pts = np.array([
                    [mid_x - 60, mid_y],
                    [mid_x - 20, mid_y - 25],
                    [mid_x - 20, mid_y + 25],
                ], np.int32)
                cv2.fillPoly(frame, [pts], color)

            elif nav_type == "turn_right":
                pts = np.array([
                    [mid_x + 60, mid_y],
                    [mid_x + 20, mid_y - 25],
                    [mid_x + 20, mid_y + 25],
                ], np.int32)
                cv2.fillPoly(frame, [pts], color)

            elif nav_type == "blocked":
                cv2.line(frame,
                         (mid_x - 30, mid_y - 30),
                         (mid_x + 30, mid_y + 30),
                         color, 4)
                cv2.line(frame,
                         (mid_x + 30, mid_y - 30),
                         (mid_x - 30, mid_y + 30),
                         color, 4)

            # Instruction text
            cv2.putText(frame, msg,
                        (14, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)

        return frame
