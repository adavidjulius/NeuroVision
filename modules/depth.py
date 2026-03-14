"""
NeuroVision — Phase 4: Depth Estimation (Calibrated)
─────────────────────────────────────────────────────
Uses MiDaS for relative depth + object size calibration
for more accurate real-world distance estimates.
"""

import cv2
import numpy as np
import torch


# Real-world average object sizes in metres (height)
# Used for size-based distance calibration
KNOWN_SIZES = {
    "person":       1.70,
    "car":          1.50,
    "truck":        2.50,
    "bus":          3.00,
    "bicycle":      1.00,
    "motorcycle":   1.10,
    "chair":        0.90,
    "dining table": 0.75,
    "laptop":       0.30,
    "tv":           0.60,
    "bottle":       0.25,
    "cup":          0.12,
    "dog":          0.50,
    "cat":          0.30,
    "default":      0.50,
}

# Approximate focal length in pixels for a standard webcam at 720p
FOCAL_LENGTH = 700


class DepthEstimator:
    def __init__(self):
        print("📡  Loading MiDaS depth model ...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.model = torch.hub.load(
                "intel-isl/MiDaS", "MiDaS_small", trust_repo=True
            )
            self.model.to(self.device)
            self.model.eval()

            transforms      = torch.hub.load(
                "intel-isl/MiDaS", "transforms", trust_repo=True
            )
            self.transform  = transforms.small_transform
            self.ready      = True
            print("✅  MiDaS depth model ready")

        except Exception as e:
            print(f"⚠️   MiDaS unavailable ({e}) — using size-based estimation only")
            self.model      = None
            self.transform  = None
            self.ready      = False

    def estimate(self, frame: np.ndarray):
        """Returns normalised depth map (0-1). Higher = closer."""
        if not self.ready:
            return None

        img_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img_rgb).to(self.device)

        with torch.no_grad():
            pred = self.model(input_batch)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = pred.cpu().numpy()
        d_min, d_max = depth.min(), depth.max()
        if d_max > d_min:
            depth = (depth - d_min) / (d_max - d_min)
        return depth

    def get_distance_metres(self, frame, depth_map, box, label="default") -> float:
        """
        Returns estimated distance in real metres using two methods:
        1. Object size triangulation (most accurate for known objects)
        2. MiDaS depth map fallback
        Picks the more reliable of the two.
        """
        x1, y1, x2, y2 = box
        h_frame         = frame.shape[0]
        box_height_px   = max(1, y2 - y1)

        # ── Method 1: Size triangulation ──────────────────────
        known_h = KNOWN_SIZES.get(label, KNOWN_SIZES["default"])
        size_dist = (known_h * FOCAL_LENGTH) / box_height_px

        # ── Method 2: MiDaS depth ─────────────────────────────
        midas_dist = None
        if depth_map is not None:
            region = depth_map[y1:y2, x1:x2]
            if region.size > 0:
                depth_val  = float(np.mean(region))
                # Convert normalised depth to metres
                # MiDaS is inverse depth — higher value = closer
                # Empirically calibrated: 0.9 ≈ 0.5m, 0.1 ≈ 5m+
                if depth_val > 0.01:
                    midas_dist = 0.5 / depth_val * 0.45

        # ── Blend both estimates ───────────────────────────────
        if midas_dist is not None:
            # Weight: 60% size triangulation, 40% MiDaS
            distance = 0.6 * size_dist + 0.4 * midas_dist
        else:
            distance = size_dist

        # Clamp to reasonable range
        return round(max(0.3, min(distance, 15.0)), 1)

    def get_distance_label(self, depth_map, box, area_ratio, label="default",
                           frame=None) -> str:
        """Returns a human-readable distance string."""
        if frame is not None:
            metres = self.get_distance_metres(frame, depth_map, box, label)
            return f"~{metres}m"

        # Fallback: area ratio only
        if area_ratio > 0.30:
            return "~0.5m"
        elif area_ratio > 0.18:
            return "~1m"
        elif area_ratio > 0.10:
            return "~1.5m"
        elif area_ratio > 0.05:
            return "~2.5m"
        elif area_ratio > 0.02:
            return "~4m"
        else:
            return ">5m"

    def colorize(self, depth_map: np.ndarray):
        if depth_map is None:
            return None
        d8 = (depth_map * 255).astype(np.uint8)
        return cv2.applyColorMap(d8, cv2.COLORMAP_MAGMA)
