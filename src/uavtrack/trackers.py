"""Three online trackers, written from scratch, sharing one interface.

    tracker.update(dets, warp=None) -> (k, 5) array of x1, y1, x2, y2, track_id

`dets` is (n, 5): x1, y1, x2, y2, score for one frame. `warp` is an optional
2x3 affine matrix that maps the previous frame's pixel coordinates to this
frame's: the camera's own motion, estimated by camera_motion.py.

* IoUTracker: the baseline. Each track is extended by the detection that
  overlaps its last box most (Bochinski et al., 2017). No motion model.
* SORT: a constant-velocity Kalman filter per track, Hungarian assignment on
  IoU between predicted and detected boxes (Bewley et al., 2016).
* ByteSORT: SORT plus ByteTrack's second pass (Zhang et al., 2022): tracks
  left unmatched by confident detections get a second chance against
  low-confidence ones, which is where a partly hidden vehicle usually is.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .visdrone import iou_matrix


# ---------------------------------------------------------------- Kalman box


def box_to_z(b: np.ndarray) -> np.ndarray:
    """x1, y1, x2, y2 -> centre x, centre y, area, aspect ratio."""
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w * h, w / max(h, 1e-6)])


def x_to_box(x: np.ndarray) -> np.ndarray:
    s, r = max(x[2], 1e-6), max(x[3], 1e-6)
    w = np.sqrt(s * r)
    h = s / max(w, 1e-6)
    return np.array([x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2])


class KalmanBox:
    """State: cx, cy, area, aspect, d(cx), d(cy), d(area); aspect is constant.

    Noise settings follow SORT. The area velocity is clamped so the predicted
    area cannot go negative, a known failure of the original filter on
    shrinking boxes.
    """

    F = np.eye(7)
    F[0, 4] = F[1, 5] = F[2, 6] = 1.0
    H = np.eye(4, 7)
    R = np.diag([1.0, 1.0, 10.0, 10.0])
    Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 1e-4])

    def __init__(self, box: np.ndarray):
        self.x = np.zeros(7)
        self.x[:4] = box_to_z(box)
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 1e4, 1e4, 1e4])

    def predict(self) -> np.ndarray:
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return x_to_box(self.x)

    def update(self, box: np.ndarray) -> None:
        z = box_to_z(box)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    def apply_warp(self, A: np.ndarray) -> None:
        """Move the state with the camera: centre and velocity through the
        affine map, area scaled by its determinant."""
        M, t = A[:, :2], A[:, 2]
        self.x[0:2] = M @ self.x[0:2] + t
        self.x[4:6] = M @ self.x[4:6]
        scale = abs(np.linalg.det(M))
        self.x[2] *= scale
        self.x[6] *= scale
        J = np.eye(7)
        J[0:2, 0:2] = M
        J[4:6, 4:6] = M
        self.P = J @ self.P @ J.T

    @property
    def box(self) -> np.ndarray:
        return x_to_box(self.x)


class _Track:
    def __init__(self, box, tid, score):
        self.kf = KalmanBox(box)
        self.id = tid
        self.hits = 1
        self.age_since_update = 0
        self.score = score


def _assign(cost: np.ndarray, max_cost: float):
    """Hungarian assignment; pairs costing more than max_cost are rejected."""
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    r, c = linear_sum_assignment(cost)
    keep = cost[r, c] <= max_cost
    matches = list(zip(r[keep], c[keep]))
    mr, mc = set(r[keep]), set(c[keep])
    return matches, [i for i in range(cost.shape[0]) if i not in mr], [j for j in range(cost.shape[1]) if j not in mc]


# ---------------------------------------------------------------- SORT family


class SORT:
    name = "sort"

    def __init__(self, high=0.5, iou=0.3, max_age=10, min_hits=3, use_warp=False):
        self.high, self.iou, self.max_age, self.min_hits = high, iou, max_age, min_hits
        self.use_warp = use_warp
        self.tracks: list[_Track] = []
        self.next_id = 1
        self.frame = 0

    def _predict(self, warp):
        boxes = []
        for t in self.tracks:
            if self.use_warp and warp is not None:
                t.kf.apply_warp(warp)
            boxes.append(t.kf.predict())
            t.age_since_update += 1
        return np.array(boxes).reshape(-1, 4)

    def _match(self, track_idx, pred, dets, iou_thr):
        if not track_idx or len(dets) == 0:
            return [], track_idx, list(range(len(dets)))
        cost = 1.0 - iou_matrix(pred[track_idx], dets[:, :4])
        m, ut, ud = _assign(cost, 1.0 - iou_thr)
        return [(track_idx[i], j) for i, j in m], [track_idx[i] for i in ut], ud

    def _commit(self, matches, dets):
        for ti, dj in matches:
            t = self.tracks[ti]
            t.kf.update(dets[dj, :4])
            t.hits += 1
            t.age_since_update = 0
            t.score = dets[dj, 4]

    def _spawn(self, dets):
        for d in dets:
            self.tracks.append(_Track(d[:4], self.next_id, d[4]))
            self.next_id += 1

    def _output(self):
        self.tracks = [t for t in self.tracks if t.age_since_update <= self.max_age]
        out = [np.r_[t.kf.box, t.id] for t in self.tracks
               if t.age_since_update == 0 and (t.hits >= self.min_hits or self.frame <= self.min_hits)]
        return np.array(out).reshape(-1, 5)

    def update(self, dets: np.ndarray, warp: np.ndarray | None = None) -> np.ndarray:
        self.frame += 1
        pred = self._predict(warp)
        high = dets[dets[:, 4] >= self.high]
        matches, _, unmatched = self._match(list(range(len(self.tracks))), pred, high, self.iou)
        self._commit(matches, high)
        self._spawn(high[unmatched])
        return self._output()


class ByteSORT(SORT):
    name = "bytesort"

    def __init__(self, high=0.5, low=0.1, iou=0.3, iou_low=0.5, max_age=10, min_hits=3, use_warp=False):
        super().__init__(high, iou, max_age, min_hits, use_warp)
        self.low, self.iou_low = low, iou_low

    def update(self, dets: np.ndarray, warp: np.ndarray | None = None) -> np.ndarray:
        self.frame += 1
        pred = self._predict(warp)
        high = dets[dets[:, 4] >= self.high]
        low = dets[(dets[:, 4] >= self.low) & (dets[:, 4] < self.high)]
        m1, left, unmatched_high = self._match(list(range(len(self.tracks))), pred, high, self.iou)
        self._commit(m1, high)
        m2, _, _ = self._match(left, pred, low, self.iou_low)  # second pass: only existing tracks
        self._commit(m2, low)
        self._spawn(high[unmatched_high])  # new tracks only from confident detections
        return self._output()


# ---------------------------------------------------------------- baseline


class IoUTracker:
    """Greedy IoU tracker with no motion model. A track ends at its first
    missed frame. Output follows the same min_hits rule as SORT, so the
    comparison is about association, not about what gets reported."""

    name = "iou"

    def __init__(self, high=0.5, iou=0.3, min_hits=3):
        self.high, self.iou, self.min_hits = high, iou, min_hits
        self.tracks: list[dict] = []
        self.next_id = 1

    def update(self, dets: np.ndarray, warp: np.ndarray | None = None) -> np.ndarray:
        dets = dets[dets[:, 4] >= self.high]
        alive, used = [], np.zeros(len(dets), bool)
        if self.tracks and len(dets):
            ious = iou_matrix(np.array([t["box"] for t in self.tracks]), dets[:, :4])
            for ti in np.argsort(-ious.max(axis=1)):
                j = int(np.argmax(np.where(used, -1, ious[ti])))
                if not used[j] and ious[ti, j] >= self.iou:
                    used[j] = True
                    t = self.tracks[ti]
                    t["box"], t["hits"] = dets[j, :4], t["hits"] + 1
                    alive.append(t)
        for j in np.flatnonzero(~used):
            alive.append({"box": dets[j, :4], "id": self.next_id, "hits": 1})
            self.next_id += 1
        self.tracks = alive
        out = [np.r_[t["box"], t["id"]] for t in self.tracks if t["hits"] >= self.min_hits]
        return np.array(out).reshape(-1, 5)
