"""Edge step: the fine-tuned detector as ONNX fp32 and int8, on a laptop CPU.

A drone's companion computer is closer to a CPU than to a desktop GPU, so
both models are timed on the CPU with ONNX Runtime. Accuracy is judged by
what matters downstream: the same tracker run on each model's detections
over the validation split.

int8: ONNX Runtime static quantisation (QDQ, per-channel weights), calibrated
on 64 training images, never on the validation sequences.

Usage: python scripts/edge.py   (needs models/vehicle_v8n.pt; writes results/edge.json)
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from uavtrack import evaluate  # noqa: E402
from uavtrack.visdrone import mot_split  # noqa: E402
from cache_detections import IMGSZ, split_root  # noqa: E402
from run_tracking import TRACKERS, track  # noqa: E402

CALIB, TIMING_FRAMES, TRACKER = 64, 100, "sort+cmc"  # the main system chosen on val


def export_fp32() -> Path:
    from ultralytics import YOLO

    out = YOLO(str(ROOT / "models/vehicle_v8n.pt")).export(format="onnx", imgsz=IMGSZ, dynamic=False, simplify=True,
                                                           opset=17, device="cpu")
    dst = ROOT / "models/vehicle_v8n_fp32.onnx"
    Path(out).replace(dst)
    return dst


def letterbox(img, size=IMGSZ):
    import cv2

    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    canvas = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(x)


def quantise(fp32: Path) -> Path:
    import cv2
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

    imgs = sorted((ROOT / "data/yolo/train/images").glob("*.jpg"))
    rng = np.random.default_rng(0)
    pick = [imgs[i] for i in rng.choice(len(imgs), CALIB, replace=False)]

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(pick)

        def get_next(self):
            p = next(self.it, None)
            return None if p is None else {"images": letterbox(cv2.imread(str(p)))}

    dst = ROOT / "models/vehicle_v8n_int8.onnx"
    quantize_static(str(fp32), str(dst), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8)
    return dst


def detect_split(model_path: Path, split: str) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(model_path), task="detect")
    out = {}
    for s in mot_split(split_root(split)):
        rows = []
        for k, f in enumerate(s.frames, start=1):
            b = model.predict(str(f), imgsz=IMGSZ, conf=0.05, iou=0.6, max_det=1000, device="cpu", verbose=False)[0].boxes
            if len(b):
                rows.append(np.column_stack([np.full(len(b), k), b.xyxy.numpy(), b.conf.numpy()]))
        out[s.name] = np.vstack(rows) if rows else np.zeros((0, 6))
    return out


def time_cpu(model_path: Path, threads: int) -> float:
    import cv2
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
    frames = mot_split(split_root("val"))[0].frames[:TIMING_FRAMES]
    xs = [letterbox(cv2.imread(str(f))) for f in frames]
    for x in xs[:5]:
        sess.run(None, {"images": x})
    t = []
    for x in xs:
        t0 = time.perf_counter()
        sess.run(None, {"images": x})
        t.append(time.perf_counter() - t0)
    return float(np.median(t) * 1000)


def main():
    fp32 = export_fp32()
    int8 = quantise(fp32)
    seqs = mot_split(split_root("val"))
    res = {"cpu": platform.processor(), "imgsz": IMGSZ, "tracker": TRACKER, "models": {}}
    for name, path in (("fp32", fp32), ("int8", int8)):
        dets = detect_split(path, "val")
        hyps = {}
        for s in seqs:
            warps = np.load(ROOT / "data/warps/val" / f"{s.name}.npz")["warps"]
            hyps[s.name], _ = track(s, dets[s.name], warps, TRACKERS[TRACKER])
        o = evaluate.summarise(seqs, hyps)["overall"]
        res["models"][name] = {
            "file_mb": round(path.stat().st_size / 2**20, 2),
            "latency_ms_1_thread": round(time_cpu(path, 1), 1),
            "latency_ms_all_threads": round(time_cpu(path, os.cpu_count() or 4), 1),
            "mota": o["mota"], "idf1": o["idf1"],
            "count_mape_pct_30f": evaluate.count_error(seqs, hyps, 30)["count_mape_pct"],
        }
        print(name, res["models"][name])
    (ROOT / "results/edge.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
