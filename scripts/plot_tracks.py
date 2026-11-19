"""Side-by-side picture: IoU tracker vs ByteSORT + camera-motion compensation.

The sequence is the validation sequence with the largest camera motion (95th
percentile shift), and the frame is the one where the camera moved most
within it, so the picture shows the case the compensation is for. Trails
cover the previous 20 frames, each point where the vehicle was in its own
frame, so during a pan the trails sweep with the camera.

Usage: python scripts/plot_tracks.py [detector]   (writes docs/tracks.jpg)
"""

from __future__ import annotations

import colorsys
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from uavtrack.visdrone import mot_split  # noqa: E402
from cache_detections import split_root  # noqa: E402
from run_tracking import TRACKERS, track  # noqa: E402

TRAIL = 20


def colour(tid: int):
    h = (tid * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def draw(img, hyp, frame, label):
    out = img.copy()
    recent = hyp[(hyp[:, 0] > frame - TRAIL) & (hyp[:, 0] <= frame)]
    for tid in np.unique(recent[:, 1]):
        t = recent[recent[:, 1] == tid]
        pts = np.column_stack([(t[:, 2] + t[:, 4]) / 2, t[:, 5]]).astype(np.int32)
        cv2.polylines(out, [pts], False, colour(int(tid)), 2, cv2.LINE_AA)
    now = hyp[hyp[:, 0] == frame]
    for f, tid, x1, y1, x2, y2 in now:
        c = colour(int(tid))
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), c, 2, cv2.LINE_AA)
        cv2.putText(out, str(int(tid)), (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (out.shape[1], 56), (20, 20, 20), -1)
    cv2.putText(out, label, (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main(detector: str = "visdrone"):
    seqs = {s.name: s for s in mot_split(split_root("val"))}
    shifts = {}
    for name in seqs:
        w = np.load(ROOT / "data/warps/val" / f"{name}.npz")["warps"]
        shifts[name] = np.hypot(w[:, 0, 2], w[:, 1, 2])
    name = max(shifts, key=lambda n: np.percentile(shifts[n][1:], 95))
    s = seqs[name]
    frame = int(np.argmax(shifts[name])) + 1
    frame = min(max(frame, TRAIL + 1), len(s.frames))
    dets = np.load(ROOT / "data/detections" / detector / "val" / f"{name}.npy")
    warps = np.load(ROOT / "data/warps/val" / f"{name}.npz")["warps"]
    a, _ = track(s, dets, None, TRACKERS["iou"])
    b, _ = track(s, dets, warps, TRACKERS["bytesort+cmc"])
    img = cv2.imread(str(s.frames[frame - 1]))
    left = draw(img, a, frame, f"IoU tracker: {int((a[:, 0] == frame).sum())} vehicles tracked in this frame")
    right = draw(img, b, frame, f"ByteSORT + camera motion: {int((b[:, 0] == frame).sum())} tracked")
    gt_n = len(np.unique(s.gt[:, 1]))
    both = np.hstack([left, np.full((img.shape[0], 12, 3), 255, np.uint8), right])
    scale = 1800 / both.shape[1]
    both = cv2.resize(both, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    (ROOT / "docs").mkdir(exist_ok=True)
    cv2.imwrite(str(ROOT / "docs/tracks.jpg"), both, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"{name}, frame {frame}, camera shift {shifts[name][frame - 1]:.0f} px; ground truth {gt_n} vehicles")


if __name__ == "__main__":
    main(*sys.argv[1:])
