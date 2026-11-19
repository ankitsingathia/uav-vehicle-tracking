"""Run a detector once over every frame of a MOT split and save the boxes.

Every tracker is then scored on the same detections, so differences between
trackers are about association, not about the detector.

detectors:
  coco      YOLOv8n pretrained on COCO, keeping car, bus and truck as "vehicle"
  visdrone  YOLOv8n fine-tuned on VisDrone-DET (scripts/train.py)

Boxes are kept down to confidence 0.05 so the trackers can choose their own
thresholds (ByteSORT uses the low-confidence ones).

Usage: python scripts/cache_detections.py coco val
       python scripts/cache_detections.py visdrone testdev
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uavtrack.visdrone import mot_split  # noqa: E402

IMGSZ = 960
DEVICE = os.environ.get("DEVICE", "0")  # "cpu" lets detection run while the GPU is training
COCO_VEHICLES = [2, 5, 7]  # car, bus, truck
WEIGHTS = {"coco": "yolov8n.pt", "visdrone": str(ROOT / "models/vehicle_v8n.pt")}


def split_root(split: str) -> Path:
    return next((ROOT / "data/raw/mot" / split).rglob("sequences")).parent


def main(detector: str, split: str):
    from ultralytics import YOLO

    model = YOLO(WEIGHTS[detector])
    classes = COCO_VEHICLES if detector == "coco" else None
    out_dir = ROOT / "data/detections" / detector / split
    out_dir.mkdir(parents=True, exist_ok=True)
    seqs = mot_split(split_root(split))
    t_total, n_total = 0.0, 0
    for s in seqs:
        rows = []
        t0 = time.perf_counter()
        for k, f in enumerate(s.frames, start=1):
            # max_det raised from YOLO's default 300: the densest VisDrone images hold 349 vehicles
            r = model.predict(str(f), imgsz=IMGSZ, conf=0.05, iou=0.6, classes=classes, agnostic_nms=True,
                              max_det=1000, device=DEVICE, verbose=False, half=DEVICE != "cpu")[0]
            b = r.boxes
            if len(b):
                xyxy = b.xyxy.cpu().numpy()
                conf = b.conf.cpu().numpy()
                rows.append(np.column_stack([np.full(len(conf), k), xyxy, conf]))
        t_total += time.perf_counter() - t0
        n_total += len(s.frames)
        dets = np.vstack(rows) if rows else np.zeros((0, 6))
        np.save(out_dir / f"{s.name}.npy", dets.astype(np.float32))
        print(f"{s.name}: {len(s.frames)} frames, {len(dets)} boxes")
    print(f"{detector}/{split}: {n_total} frames, {n_total / t_total:.1f} frames/s end to end (device {DEVICE}, {IMGSZ} px)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
