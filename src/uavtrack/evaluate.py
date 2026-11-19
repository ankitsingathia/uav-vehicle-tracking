"""Tracking metrics with py-motmetrics, plus a vehicle count.

Per frame, ground-truth vehicles and tracker output are matched at IoU >= 0.5
(the MOTChallenge and VisDrone threshold). Tracker boxes lying mostly inside
an annotated "ignored" region are dropped first, as VisDrone's own evaluation
does.

MOTA = 1 - (misses + false positives + identity switches) / ground-truth boxes
IDF1 = how often a vehicle keeps the same identity over its whole life

The count is what a traffic survey would report: distinct vehicles seen in the
sequence. The ground truth's count is its number of distinct vehicle IDs. The
tracker's count is its number of distinct track IDs, so every broken track
counts a vehicle twice and every merged one hides a vehicle.
"""

from __future__ import annotations

import motmetrics as mm
import numpy as np

from .visdrone import Sequence, inside_ignored, iou_matrix

METRICS = ["mota", "idf1", "num_switches", "num_false_positives", "num_misses", "precision", "recall",
           "mostly_tracked", "mostly_lost", "num_unique_objects"]


def accumulate(seq: Sequence, hyp: np.ndarray) -> mm.MOTAccumulator:
    """hyp: (n, 6) rows of frame, id, x1, y1, x2, y2."""
    acc = mm.MOTAccumulator(auto_id=False)
    n_frames = len(seq.frames)
    for f in range(1, n_frames + 1):
        g = seq.gt[seq.gt[:, 0] == f]
        h = hyp[hyp[:, 0] == f] if len(hyp) else np.zeros((0, 6))
        if len(h):
            h = h[~inside_ignored(h[:, 2:6], seq.ignore.get(f))]
        d = 1.0 - iou_matrix(g[:, 2:6], h[:, 2:6])
        d[d > 0.5] = np.nan
        acc.update(g[:, 1].astype(int).tolist(), h[:, 1].astype(int).tolist(), d, frameid=f)
    return acc


def drop_ignored(seq: Sequence, hyp: np.ndarray) -> np.ndarray:
    """Remove tracker rows lying mostly inside the frame's ignored regions."""
    if len(hyp) == 0:
        return hyp
    keep = np.ones(len(hyp), bool)
    for f, regions in seq.ignore.items():
        m = hyp[:, 0] == f
        if m.any():
            keep[m] = ~inside_ignored(hyp[m, 2:6], regions)
    return hyp[keep]


def count_error(seqs: list[Sequence], hyps: dict[str, np.ndarray], min_len: int) -> dict:
    """Count only tracks seen in at least `min_len` frames, as a counting
    system would: a vehicle is real once it has been followed long enough.
    Short tracks are mostly false boxes that flickered for a few frames."""
    errs = []
    for s in seqs:
        h = drop_ignored(s, hyps[s.name])
        ids, n = np.unique(h[:, 1], return_counts=True) if len(h) else (np.array([]), np.array([]))
        pred = int((n >= min_len).sum())
        gt = len(np.unique(s.gt[:, 1]))
        errs.append(100.0 * (pred - gt) / max(gt, 1))
    errs = np.array(errs)
    return {"min_len": min_len, "count_mape_pct": float(np.mean(np.abs(errs))), "count_bias_pct": float(np.mean(errs))}


def summarise(seqs: list[Sequence], hyps: dict[str, np.ndarray]) -> dict:
    hyps = {s.name: drop_ignored(s, hyps[s.name]) for s in seqs}
    accs = [accumulate(s, hyps[s.name]) for s in seqs]
    mh = mm.metrics.create()
    df = mh.compute_many(accs, metrics=METRICS, names=[s.name for s in seqs], generate_overall=True)
    overall = {k: float(df.loc["OVERALL", k]) for k in METRICS}
    counts = {}
    for s in seqs:
        gt_n = len(np.unique(s.gt[:, 1]))
        h = hyps[s.name]
        pred_n = len(np.unique(h[:, 1])) if len(h) else 0
        counts[s.name] = {"gt": int(gt_n), "tracker": int(pred_n), "error_pct": 100.0 * (pred_n - gt_n) / max(gt_n, 1)}
    errs = np.array([c["error_pct"] for c in counts.values()])
    overall["count_mape_pct"] = float(np.mean(np.abs(errs)))
    overall["count_bias_pct"] = float(np.mean(errs))
    per_seq = {s.name: {k: float(df.loc[s.name, k]) for k in METRICS} for s in seqs}
    return {"overall": overall, "per_sequence": per_seq, "counts": counts}
