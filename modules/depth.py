"""
NeuroVision — Phase 4: Depth Estimation
─────────────────────────────────────────
Uses MiDaS monocular depth model to estimate
object distances from a single camera frame.
No depth camera hardware needed.
"""

import cv2
import numpy as np
import torch

class DepthEstimator:
    def __init__(self):
        print("📡  Loading MiDaS depth model ...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   Device: {self.device}")

        try:
            self.model = torch.hub.load(
                "intel-isl/MiDaS", "MiDaS_small",
                trust_repo=True
            )
            self.model.to(self.device)
            self.model.eval()

            midas_transforms = torch.hub.load(
                "intel-isl/MiDaS", "transforms",
                trust_repo=True
            )
            self.transform = midas_transforms.small_transform
            self.ready = True
            print("✅  MiDaS depth model ready")

        except Exception as e:
            print(f"⚠️   MiDaS unavailable ({e}) — using fallback size estimation")
            self.model     = None
            self.transform = None
            self.ready     = False

    def estimate(self, frame: np.ndarray):
        """
        Returns a depth map (HxW float array).
        Higher value = closer to camera.
        Returns None if model unavailable.
        """
        if not self.ready:
            return None

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        # Normalize to 0-1
        d_min = depth_map.min()
        d_max = depth_map.max()
        if d_max > d_min:
            depth_map = (depth_map - d_min) / (d_max - d_min)

        return depth_map

    def get_distance_label(self, depth_map, box: tuple,
                           area_ratio: float) -> str:
        """
        Returns a human-readable distance string.
        Uses depth map if available, falls back to area_ratio.
        """
        if depth_map is not None:
            x1, y1, x2, y2 = box
            region = depth_map[y1:y2, x1:x2]
            if region.size > 0:
                depth_val = float(np.mean(region))
                return self._depth_to_label(depth_val)

        # Fallback: estimate from bounding box size
        return self._area_to_label(area_ratio)

    def _depth_to_label(self, depth_val: float) -> str:
        """Converts normalised depth (0-1) to distance label."""
        if depth_val > 0.75:
            return "~0.5m"
        elif depth_val > 0.60:
            return "~1m"
        elif depth_val > 0.45:
            return "~1.5m"
        elif depth_val > 0.30:
            return "~2m"
        elif depth_val > 0.15:
            return "~3m"
        else:
            return ">4m"

    def _area_to_label(self, area_ratio: float) -> str:
        """Fallback: estimate distance from bounding box area."""
        if area_ratio > 0.30:
            return "~0.5m"
        elif area_ratio > 0.20:
            return "~1m"
        elif area_ratio > 0.10:
            return "~1.5m"
        elif area_ratio > 0.05:
            return "~2m"
        elif area_ratio > 0.02:
            return "~3m"
        else:
            return ">4m"

    def colorize(self, depth_map: np.ndarray) -> np.ndarray:
        """Returns a colourised depth map for visualisation."""
        if depth_map is None:
            return None
        d8 = (depth_map * 255).astype(np.uint8)
        return cv2.applyColorMap(d8, cv2.COLORMAP_MAGMA)
