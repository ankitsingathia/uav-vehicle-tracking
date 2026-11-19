"""Score every tracker on cached detections.

Tracker settings are fixed in TRACKERS below, chosen before scoring and
identical for every detector and split. Validation is used to compare
designs; test-dev is reported once, with the same settings.

Usage: python scripts/run_tracking.py val        (writes results/tracking_val.json)
       python scripts/run_tracking.py testdev
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from uavtrack import evaluate, trackers  # noqa: E402
from uavtrack.visdrone import mot_split  # noqa: E402
from cache_detections import split_root  # noqa: E402

TRACKERS = {
    "iou": lambda: trackers.IoUTracker(high=0.5, iou=0.3, min_hits=3),
    "sort": lambda: trackers.SORT(high=0.5, iou=0.3, max_age=10, min_hits=3),
    "sort+cmc": lambda: trackers.SORT(high=0.5, iou=0.3, max_age=10, min_hits=3, use_warp=True),
    "bytesort": lambda: trackers.ByteSORT(high=0.5, low=0.1, iou=0.3, iou_low=0.5, max_age=10, min_hits=3),
    "bytesort+cmc": lambda: trackers.ByteSORT(high=0.5, low=0.1, iou=0.3, iou_low=0.5, max_age=10, min_hits=3,
                                              use_warp=True),
}


def track(seq, dets, warps, make):
    t = make()
    out = []
    t0 = time.perf_counter()
    for k in range(1, len(seq.frames) + 1):
        d = dets[dets[:, 0] == k][:, 1:6] if len(dets) else np.zeros((0, 5))
        res = t.update(d, warps[k - 1] if warps is not None else None)
        out += [[k, r[4], *r[:4]] for r in res]
    return np.array(out).reshape(-1, 6), time.perf_counter() - t0


def main(split: str):
    seqs = mot_split(split_root(split))
    n_frames = sum(len(s.frames) for s in seqs)
    res = {"split": split, "sequences": len(seqs), "frames": n_frames,
           "gt_vehicle_boxes": int(sum(len(s.gt) for s in seqs)), "results": {}}
    for detector in ("coco", "visdrone"):
        ddir = ROOT / "data/detections" / detector / split
        if not ddir.exists():
            continue
        for name, make in TRACKERS.items():
            hyps, secs = {}, 0.0
            for s in seqs:
                dets = np.load(ddir / f"{s.name}.npy")
                warps = np.load(ROOT / "data/warps" / split / f"{s.name}.npz")["warps"] if "cmc" in name else None
                hyps[s.name], dt = track(s, dets, warps, make)
                secs += dt
            r = evaluate.summarise(seqs, hyps)
            r["overall"]["tracker_fps"] = n_frames / secs
            res["results"][f"{detector}/{name}"] = r
            o = r["overall"]
            print(f"{detector:8s} {name:13s} MOTA {o['mota']:6.3f}  IDF1 {o['idf1']:.3f}  IDsw {int(o['num_switches']):5d}  "
                  f"FP {int(o['num_false_positives']):6d}  FN {int(o['num_misses']):6d}  "
                  f"count error {o['count_mape_pct']:5.1f}% (bias {o['count_bias_pct']:+.1f}%)  {o['tracker_fps']:.0f} fps")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / f"results/tracking_{split}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
