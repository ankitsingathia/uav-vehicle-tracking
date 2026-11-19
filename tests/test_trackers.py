"""Trackers and metrics on synthetic scenes with known answers."""

from pathlib import Path

import numpy as np
import pytest

from uavtrack import evaluate, trackers
from uavtrack.visdrone import Sequence, inside_ignored, iou_matrix


def box(cx, cy, w=20, h=10, score=0.9):
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, score])


def run(tracker, frames, warps=None):
    out = []
    for k, dets in enumerate(frames):
        w = None if warps is None else warps[k]
        res = tracker.update(np.array(dets).reshape(-1, 5), w)
        out += [[k + 1, *r[4:5], *r[:4]] for r in res]
    return np.array(out).reshape(-1, 6)


def test_iou_of_identical_and_disjoint_boxes():
    a = np.array([[0, 0, 10, 10]], float)
    assert iou_matrix(a, a)[0, 0] == pytest.approx(1.0)
    assert iou_matrix(a, a + 50)[0, 0] == 0.0


def test_kalman_learns_constant_velocity():
    kf = trackers.KalmanBox(box(0, 0)[:4])
    for k in range(1, 30):
        kf.predict()
        kf.update(box(5 * k, 0)[:4])
    nxt = kf.predict()
    assert (nxt[0] + nxt[2]) / 2 == pytest.approx(150, abs=1.0)  # next centre at 5 * 30


def test_warp_moves_a_track_with_the_camera():
    kf = trackers.KalmanBox(box(100, 100)[:4])
    shift = np.array([[1.0, 0, 30.0], [0, 1.0, -10.0]])
    kf.apply_warp(shift)
    b = kf.box
    assert ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) == pytest.approx((130.0, 90.0))


def test_sort_keeps_identity_through_two_missed_frames():
    frames = [[box(10 * k, 50)] for k in range(10)]
    frames[5] = []
    frames[6] = []
    hyp = run(trackers.SORT(min_hits=1), frames)
    assert len(np.unique(hyp[:, 1])) == 1


def test_iou_tracker_breaks_on_a_missed_frame():
    frames = [[box(10 * k, 50)] for k in range(10)]
    frames[5] = []
    hyp = run(trackers.IoUTracker(min_hits=1), frames)
    assert len(np.unique(hyp[:, 1])) == 2


def test_bytesort_uses_a_low_confidence_detection_to_keep_a_track():
    frames = [[box(10 * k, 50)] for k in range(10)]
    frames[5] = [box(50, 50, score=0.2)]  # partly hidden: weak detection
    frames[6] = [box(60, 50, score=0.2)]
    sort_ids = run(trackers.SORT(min_hits=1, max_age=1), frames)
    byte_ids = run(trackers.ByteSORT(min_hits=1, max_age=1), frames)
    assert len(np.unique(byte_ids[:, 1])) == 1  # the weak detections keep the track alive
    assert len(np.unique(sort_ids[:, 1])) == 2  # SORT ignores them, loses the track, and counts the car twice
    assert (byte_ids[:, 0] == 6).any() and not (sort_ids[:, 0] == 6).any()


def test_camera_motion_compensation_rescues_a_fast_pan():
    """A parked car while the camera pans 30 px per frame: without the warp the
    filter has to relearn the motion; with it, the box never moves."""
    frames = [[box(200 - 30 * k, 80)] for k in range(8)]
    pan = np.array([[1.0, 0, -30.0], [0, 1.0, 0]])
    warps = [None] + [pan] * 7
    with_cmc = run(trackers.SORT(min_hits=1, iou=0.5, use_warp=True), frames, warps)
    without = run(trackers.SORT(min_hits=1, iou=0.5), frames)
    assert len(np.unique(with_cmc[:, 1])) == 1
    assert len(np.unique(without[:, 1])) > 1


def test_ignored_region_filter():
    regions = np.array([[0, 0, 100, 100]], float)
    boxes = np.array([[10, 10, 20, 20], [90, 90, 130, 130]], float)
    assert inside_ignored(boxes, regions).tolist() == [True, False]


def test_perfect_tracker_scores_mota_1_and_zero_count_error():
    gt = np.array([[f, i, 10 * f + 40 * i, 50, 10 * f + 40 * i + 20, 60] for f in range(1, 6) for i in (1, 2)], float)
    seq = Sequence("s", [Path(f"{k}.jpg") for k in range(5)], gt, {})
    r = evaluate.summarise([seq], {"s": gt.copy()})
    assert r["overall"]["mota"] == pytest.approx(1.0)
    assert r["overall"]["idf1"] == pytest.approx(1.0)
    assert r["counts"]["s"] == {"gt": 2, "tracker": 2, "error_pct": 0.0}
