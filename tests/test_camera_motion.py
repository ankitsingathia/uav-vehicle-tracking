"""The camera-motion estimate must recover a known shift and rotation."""

import cv2
import numpy as np
import pytest

from uavtrack import camera_motion


def _textured(seed=0, size=(480, 640)):
    """A random texture with structure at several scales, like an aerial scene."""
    rng = np.random.default_rng(seed)
    img = np.zeros(size, np.float32)
    for k in (4, 16, 64):
        small = rng.random((size[0] // k + 1, size[1] // k + 1)).astype(np.float32)
        img += cv2.resize(small, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
    img = (255 * (img - img.min()) / (img.max() - img.min())).astype(np.uint8)
    return img


@pytest.mark.parametrize("dx,dy,deg", [(12.0, -7.0, 0.0), (-30.0, 4.0, 1.5)])
def test_recovers_a_known_camera_motion(dx, dy, deg):
    prev = _textured()
    h, w = prev.shape
    A_true = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    A_true[:, 2] += (dx, dy)
    cur = cv2.warpAffine(prev, A_true, (w, h), borderMode=cv2.BORDER_REFLECT)
    A, inliers = camera_motion.estimate(prev, cur)
    assert inliers > 50
    # where does the image centre go?
    c = np.array([w / 2, h / 2, 1.0])
    assert A @ c == pytest.approx(A_true @ c, abs=1.0)


def test_moving_objects_do_not_drag_the_estimate():
    """A static background plus a bright block moving 40 px: RANSAC should
    report the background's (zero) motion, not the block's."""
    prev = _textured(1)
    cur = prev.copy()
    block = np.full((80, 120), 255, np.uint8)
    prev[100:180, 100:220] = block
    cur[100:180, 140:260] = block
    A, _ = camera_motion.estimate(prev, cur)
    assert np.abs(A[:, 2]).max() < 1.0
    assert A[:, :2] == pytest.approx(np.eye(2), abs=0.01)
