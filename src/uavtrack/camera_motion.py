"""Camera motion between consecutive frames, with OpenCV.

A drone camera pans, climbs and drifts. To a tracker that has no idea about
this, every parked car appears to move, and a constant-velocity Kalman
filter predicts boxes in the wrong place. Estimating the camera's own motion
and moving every track with it removes that error.

Method: corner features on the previous frame (goodFeaturesToTrack),
followed into the current frame with pyramidal Lucas-Kanade optical flow,
then a similarity transform (rotation, uniform scale, translation) fitted
with RANSAC. RANSAC rejects the points on moving vehicles, because they
disagree with the background. Frames are downscaled by 2 for speed and the
translation scaled back.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def estimate(prev_gray: np.ndarray, gray: np.ndarray, scale: float = 0.5) -> tuple[np.ndarray, int]:
    """2x3 affine mapping previous-frame pixels to current-frame pixels, and
    the number of RANSAC inliers it rests on (0 means it fell back to identity)."""
    a = cv2.resize(prev_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    b = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    pts = cv2.goodFeaturesToTrack(a, maxCorners=1000, qualityLevel=0.01, minDistance=8, blockSize=3)
    if pts is None or len(pts) < 10:
        return IDENTITY.copy(), 0
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None, winSize=(21, 21), maxLevel=3)
    ok = status.ravel() == 1
    if ok.sum() < 10:
        return IDENTITY.copy(), 0
    A, inl = cv2.estimateAffinePartial2D(pts[ok], nxt[ok], method=cv2.RANSAC, ransacReprojThreshold=1.5)
    if A is None:
        return IDENTITY.copy(), 0
    A = A.astype(np.float64)
    A[:, 2] /= scale
    return A, int(inl.sum())


def sequence_warps(frames: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """(n, 2, 3) warps, where warps[k] maps frame k-1 into frame k (warps[0] is
    identity), and (n,) inlier counts."""
    warps = [IDENTITY.copy()]
    inliers = [0]
    prev = cv2.imread(str(frames[0]), cv2.IMREAD_GRAYSCALE)
    for f in frames[1:]:
        g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        A, n = estimate(prev, g)
        warps.append(A)
        inliers.append(n)
        prev = g
    return np.stack(warps), np.array(inliers)
