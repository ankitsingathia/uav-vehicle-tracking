"""Fine-tune YOLOv8n (COCO-pretrained) on VisDrone-DET, single class "vehicle".

The RTX 3050 laptop GPU has 4 GB, which sets batch 8 at 960 px. 960 rather
than the default 640, because most VisDrone vehicles are under 30 px wide
and shrink below what the network can see at 640.

Usage: python scripts/train.py   (writes runs/vehicle_v8n and copies best.pt to models/)
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EPOCHS, IMGSZ, BATCH, SEED = 30, 960, 8, 0


def main():
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    model.train(data=str(ROOT / "data/yolo/vehicle.yaml"), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=0,
                workers=4, seed=SEED, deterministic=False, project=str(ROOT / "runs"), name="vehicle_v8n",
                exist_ok=True, plots=True, verbose=False)
    (ROOT / "models").mkdir(exist_ok=True)
    shutil.copy(ROOT / "runs/vehicle_v8n/weights/best.pt", ROOT / "models/vehicle_v8n.pt")


if __name__ == "__main__":
    main()
