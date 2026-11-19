"""Estimate the camera's motion between consecutive frames for a MOT split.

Usage: python scripts/cache_warps.py val   (writes data/warps/<split>/<seq>.npz)
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uavtrack import camera_motion  # noqa: E402
from uavtrack.visdrone import mot_split  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from cache_detections import split_root  # noqa: E402


def one(args):
    name, frames, out = args
    t0 = time.perf_counter()
    warps, inl = camera_motion.sequence_warps(frames)
    np.savez(out, warps=warps, inliers=inl)
    shift = np.hypot(warps[1:, 0, 2], warps[1:, 1, 2])
    return name, len(frames), time.perf_counter() - t0, float(np.median(shift)), float(np.percentile(shift, 95)), int((inl[1:] == 0).sum())


def main(split: str):
    out_dir = ROOT / "data/warps" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    seqs = mot_split(split_root(split))
    jobs = [(s.name, s.frames, out_dir / f"{s.name}.npz") for s in seqs]
    with ProcessPoolExecutor() as ex:
        for name, n, secs, med, p95, fails in ex.map(one, jobs):
            print(f"{name}: {n} frames, {n / secs:.0f} fps, camera shift median {med:.1f} px, p95 {p95:.1f} px, fallbacks {fails}")


if __name__ == "__main__":
    main(sys.argv[1])
