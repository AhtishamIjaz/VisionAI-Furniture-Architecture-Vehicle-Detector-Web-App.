import os
from pathlib import Path
from ultralytics import YOLO

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

models = ["yolov8m.pt", "yolov8s-worldv2.pt"]

print("[*] Downloading / Verifying VisionAI model weights...")
for m in models:
    dest = MODELS_DIR / m
    if not dest.exists():
        print(f"Downloading {m}...")
        y = YOLO(m)
        if Path(m).exists() and not dest.exists():
            Path(m).rename(dest)
        print(f"[OK] {m} ready in {dest}")
    else:
        print(f"[OK] {m} already present in {dest}")

print("[*] All models verified successfully!")
