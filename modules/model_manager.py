"""
NeuroVision — Smart Model Manager
───────────────────────────────────
Automatically picks the best model based on hardware.
MacBook Air M1/M2 → YOLOv8n (fastest, still accurate)
MacBook Pro / GPU → YOLOv8s (more accurate)
Raspberry Pi      → YOLOv8n (only option)
"""

import platform
import subprocess
import torch


def get_device_profile() -> dict:
    """Detects hardware and returns optimal settings."""
    profile = {
        "device":      "cpu",
        "yolo_model":  "yolov8n.pt",
        "depth_model": "MiDaS_small",
        "det_interval": 0.08,   # seconds between detections
        "depth_interval": 0.3,  # seconds between depth estimates
        "face_interval":  0.5,  # seconds between face scans
        "ocr_interval":   1.0,  # seconds between OCR scans
        "conf_threshold": 0.55,
    }

    # Check for Apple Silicon MPS
    if torch.backends.mps.is_available():
        profile["device"]       = "mps"
        profile["yolo_model"]   = "yolov8n.pt"
        profile["det_interval"] = 0.05
        print("⚡  Apple Silicon MPS detected — using optimised settings")

    # Check for CUDA GPU
    elif torch.cuda.is_available():
        profile["device"]        = "cuda"
        profile["yolo_model"]    = "yolov8s.pt"
        profile["depth_model"]   = "DPT_Large"
        profile["det_interval"]  = 0.03
        profile["depth_interval"] = 0.15
        print("⚡  CUDA GPU detected — using high accuracy settings")

    # CPU only
    else:
        print("⚡  CPU mode — using lightweight settings for speed")

    return profile


def get_cpu_count() -> int:
    try:
        import os
        return os.cpu_count() or 4
    except Exception:
        return 4
