"""
NeuroVision — Depth Estimator (Stable distances)
Rolling average prevents distance from jumping.
"""
import cv2
import numpy as np
import torch
from collections import defaultdict

KNOWN_SIZES = {
    "person": 1.70, "car": 1.50, "truck": 2.50,
    "bus": 3.00, "bicycle": 1.00, "motorcycle": 1.10,
    "chair": 0.90, "dining table": 0.75, "laptop": 0.30,
    "tv": 0.60, "bottle": 0.25, "cup": 0.12,
    "dog": 0.50, "cat": 0.30, "default": 0.50,
}
FOCAL_LENGTH = 700
SMOOTH_N     = 8   # average over 8 readings


class DepthEstimator:
    def __init__(self):
        print("📡  Loading MiDaS depth model ...")
        self.device = torch.device(
            "mps"  if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"   Device: {self.device}")

        # Rolling average per object key
        self._dist_history = defaultdict(list)

        try:
            self.model = torch.hub.load(
                "intel-isl/MiDaS", "MiDaS_small",
                trust_repo=True
            )
            self.model.to(self.device)
            self.model.eval()
            transforms     = torch.hub.load(
                "intel-isl/MiDaS","transforms",trust_repo=True)
            self.transform = transforms.small_transform
            self.ready     = True
            print("✅  MiDaS ready")
        except Exception as e:
            print(f"⚠️   MiDaS unavailable ({e})")
            self.model     = None
            self.transform = None
            self.ready     = False

    def estimate(self, frame: np.ndarray):
        if not self.ready:
            return None
        img     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp     = self.transform(img).to(self.device)
        with torch.no_grad():
            pred = self.model(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        d       = pred.cpu().numpy()
        d_min,d_max = d.min(), d.max()
        if d_max > d_min:
            d = (d - d_min) / (d_max - d_min)
        return d

    def _smooth_distance(self, key: str,
                          raw: float) -> float:
        """Rolling average — stable distances."""
        self._dist_history[key].append(raw)
        if len(self._dist_history[key]) > SMOOTH_N:
            self._dist_history[key].pop(0)
        return sum(self._dist_history[key]) / len(
            self._dist_history[key])

    def get_distance_metres(self, frame, depth_map,
                             box, label="default") -> float:
        x1,y1,x2,y2 = box
        box_h = max(1, y2 - y1)
        known = KNOWN_SIZES.get(label, KNOWN_SIZES["default"])

        # Size triangulation
        size_dist = (known * FOCAL_LENGTH) / box_h

        # MiDaS
        midas_dist = None
        if depth_map is not None:
            region = depth_map[y1:y2, x1:x2]
            if region.size > 0:
                dv = float(np.mean(region))
                if dv > 0.01:
                    midas_dist = 0.5 / dv * 0.45

        # Blend
        if midas_dist is not None:
            raw = 0.6 * size_dist + 0.4 * midas_dist
        else:
            raw = size_dist

        raw = max(0.3, min(raw, 15.0))

        # Smooth
        key = f"{label}_{box[0]//100}"  # group by rough position
        smoothed = self._smooth_distance(key, raw)
        return round(smoothed, 1)

    def get_distance_label(self, depth_map, box,
                            area_ratio, label="default",
                            frame=None) -> str:
        if frame is not None:
            m = self.get_distance_metres(
                frame, depth_map, box, label)
            return f"~{m}m"
        # Fallback
        if area_ratio > 0.30: return "~0.5m"
        if area_ratio > 0.18: return "~1m"
        if area_ratio > 0.10: return "~1.5m"
        if area_ratio > 0.05: return "~2.5m"
        if area_ratio > 0.02: return "~4m"
        return ">5m"

    def colorize(self, depth_map):
        if depth_map is None: return None
        d8 = (depth_map * 255).astype(np.uint8)
        return cv2.applyColorMap(d8, cv2.COLORMAP_MAGMA)
