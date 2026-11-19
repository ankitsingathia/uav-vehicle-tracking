"""CPU latency of the fp32 and int8 ONNX detectors, timed fairly.

Timings taken inside edge.py ran minutes apart while other work was running,
and the same fp32 model measured 219 ms and 1,167 ms per frame on one thread.
Here the two models are timed back to back and alternated, three rounds each
on the same 100 validation frames, and the median over all rounds is kept.

Usage: python scripts/time_cpu.py   (updates the latency fields in results/edge.json)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from uavtrack.visdrone import mot_split  # noqa: E402
from cache_detections import split_root  # noqa: E402
from edge import letterbox  # noqa: E402

ROUNDS, FRAMES = 3, 100


def session(path, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def main():
    frames = mot_split(split_root("val"))[3].frames[:FRAMES]
    xs = [letterbox(cv2.imread(str(f))) for f in frames]
    res = json.loads((ROOT / "results/edge.json").read_text())
    for threads, key in ((1, "latency_ms_1_thread"), (os.cpu_count() or 4, "latency_ms_all_threads")):
        sess = {m: session(ROOT / f"models/vehicle_v8n_{m}.onnx", threads) for m in ("fp32", "int8")}
        times = {m: [] for m in sess}
        for _ in range(ROUNDS):
            for m, s in sess.items():
                for x in xs[:5]:
                    s.run(None, {"images": x})
                for x in xs:
                    t0 = time.perf_counter()
                    s.run(None, {"images": x})
                    times[m].append(time.perf_counter() - t0)
        for m in sess:
            res["models"][m][key] = round(float(np.median(times[m]) * 1000), 1)
        print(f"{threads} thread(s): fp32 {res['models']['fp32'][key]} ms, int8 {res['models']['int8'][key]} ms")
    res["timing"] = f"median of {ROUNDS} alternating rounds x {FRAMES} frames, {res['imgsz']} px input, ONNX Runtime CPU"
    (ROOT / "results/edge.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
