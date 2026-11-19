"""VisDrone2019: MOT sequences and DET images, read for one class, "vehicle".

MOT annotation rows (one file per sequence):
    frame, target_id, left, top, width, height, score, category, truncation, occlusion
DET annotation rows (one file per image):
    left, top, width, height, score, category, truncation, occlusion

Categories: 0 ignored region, 1 pedestrian, 2 people, 3 bicycle, 4 car, 5 van,
6 truck, 7 tricycle, 8 awning-tricycle, 9 bus, 10 motor, 11 others.

This project counts and tracks vehicles: car, van, truck and bus are merged
into one class. Category 0 marks regions the annotators skipped (too dense or
too far to label); following the VisDrone protocol, detections that fall
mostly inside them are removed before scoring rather than counted as false.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

VEHICLE = (4, 5, 6, 9)
IGNORED = 0


@dataclass
class Sequence:
    name: str
    frames: list[Path]  # sorted image paths, frame k is frames[k-1]
    gt: np.ndarray  # (n, 6): frame, id, x1, y1, x2, y2 for vehicles
    ignore: dict[int, np.ndarray]  # frame -> (m, 4) x1, y1, x2, y2 ignored regions


def read_mot(seq_dir: Path, ann_file: Path) -> Sequence:
    a = np.loadtxt(ann_file, delimiter=",", ndmin=2)
    frame, tid, l, t, w, h, score, cat = (a[:, i] for i in range(8))
    boxes = np.column_stack([l, t, l + w, t + h])
    veh = np.isin(cat, VEHICLE) & (score > 0)
    gt = np.column_stack([frame[veh], tid[veh], boxes[veh]])
    ign = {}
    for f in np.unique(frame[cat == IGNORED]):
        ign[int(f)] = boxes[(cat == IGNORED) & (frame == f)]
    frames = sorted(seq_dir.glob("*.jpg"))
    return Sequence(seq_dir.name, frames, gt, ign)


def mot_split(root: Path) -> list[Sequence]:
    """root contains sequences/ and annotations/ (an unpacked VisDrone2019-MOT split)."""
    seqs = []
    for d in sorted((root / "sequences").iterdir()):
        if d.is_dir():
            seqs.append(read_mot(d, root / "annotations" / f"{d.name}.txt"))
    return seqs


def det_to_yolo(ann_file: Path, img_w: int, img_h: int) -> list[str]:
    """One VisDrone DET annotation file -> YOLO label lines, single class 0."""
    if ann_file.stat().st_size == 0:
        return []
    a = np.loadtxt(ann_file, delimiter=",", ndmin=2, usecols=range(6))
    out = []
    for l, t, w, h, score, cat in a:
        if int(cat) in VEHICLE and score > 0 and w > 0 and h > 0:
            cx, cy = (l + w / 2) / img_w, (t + h / 2) / img_h
            out.append(f"0 {cx:.6f} {cy:.6f} {w / img_w:.6f} {h / img_h:.6f}")
    return out


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a (n, 4), b (m, 4) as x1, y1, x2, y2 -> (n, m) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def inside_ignored(boxes: np.ndarray, regions: np.ndarray, frac: float = 0.5) -> np.ndarray:
    """True for boxes with more than `frac` of their area inside any ignored region."""
    if len(boxes) == 0 or regions is None or len(regions) == 0:
        return np.zeros(len(boxes), bool)
    x1 = np.maximum(boxes[:, None, 0], regions[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], regions[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], regions[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], regions[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) + 1e-9
    return (inter.max(axis=1) / area) > frac
