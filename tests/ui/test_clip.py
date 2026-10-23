"""Test the synthetic clip: its path, its face size, and the file it writes.

A drawn stand-in replaces the portrait so these need no download.
"""

from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

from abyss.ui.clip import BACKGROUND
from abyss.ui.clip import BASE_SCALE
from abyss.ui.clip import FRAME_HEIGHT
from abyss.ui.clip import FRAME_WIDTH
from abyss.ui.clip import MIN_FACE_WIDTH_PX
from abyss.ui.clip import PORTRAIT_IRIS_MID_PX
from abyss.ui.clip import REST_X
from abyss.ui.clip import REST_Y
from abyss.ui.clip import SEGMENT_FRAMES
from abyss.ui.clip import face_path
from abyss.ui.clip import n_frames
from abyss.ui.clip import render_frame
from abyss.ui.clip import split_mjpeg
from abyss.ui.clip import write_mjpeg


def stand_in() -> np.ndarray:
    """Draw a portrait-sized grey image with one white pixel at the iris midpoint.

    Grey like the clip's border, so the white pixel is the only thing brighter.
    """
    image = np.full((1024, 820, 3), BACKGROUND, np.uint8)
    x, y = (round(v) for v in PORTRAIT_IRIS_MID_PX)
    image[y, x] = 255
    return image


def test_every_movement_starts_and_ends_at_rest() -> None:
    """Each segment is a full period, so the joins have no jump."""
    for segment in range(3):
        point = face_path(segment * SEGMENT_FRAMES)
        assert (point.x_px, point.y_px, point.scale) == (REST_X, REST_Y, BASE_SCALE)


def test_each_movement_moves_one_thing() -> None:
    """Left-right only changes x, up-down only y, the approach only the scale."""
    quarter = SEGMENT_FRAMES // 4
    left = face_path(quarter)
    assert left.x_px < REST_X
    assert (left.y_px, left.scale) == (REST_Y, BASE_SCALE)
    up = face_path(SEGMENT_FRAMES + quarter)
    assert up.y_px < REST_Y
    assert (up.x_px, up.scale) == (REST_X, BASE_SCALE)
    closer = face_path(2 * SEGMENT_FRAMES + quarter)
    assert closer.scale > BASE_SCALE
    assert (closer.x_px, closer.y_px) == (REST_X, REST_Y)


def test_face_never_drops_below_the_size_mediapipe_needs() -> None:
    """The probe found no face at about 110 px; the clip stays well above that."""
    assert (
        min(face_path(i).face_width_px for i in range(n_frames())) >= MIN_FACE_WIDTH_PX
    )


def test_path_is_bounded() -> None:
    """Asking for a frame past the end is an error, not an extrapolation."""
    with pytest.raises(IndexError):
        face_path(n_frames())


def test_render_puts_the_iris_midpoint_on_the_path() -> None:
    """The marked pixel lands within a pixel of the path point.

    Only pixels brighter than the grey border count, weighted by how much
    brighter, since interpolation spreads the one white pixel over its
    neighbours.
    """
    for idx in (0, SEGMENT_FRAMES // 4, SEGMENT_FRAMES + 10, 2 * SEGMENT_FRAMES + 10):
        point = face_path(idx)
        frame = render_frame(stand_in(), point)
        assert frame.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)
        excess = frame[:, :, 0].astype(float) - BACKGROUND
        ys, xs = np.nonzero(excess > 0)
        weights = excess[ys, xs]
        assert abs(np.average(xs, weights=weights) - point.x_px) < 1.0
        assert abs(np.average(ys, weights=weights) - point.y_px) < 1.0


def test_mjpeg_round_trips(tmp_path: Path) -> None:
    """Written frames split back out, decode, and have the frame size."""
    path = tmp_path / "clip.mjpeg"
    assert write_mjpeg(stand_in(), path) == n_frames()
    frames = split_mjpeg(path.read_bytes())
    assert len(frames) == n_frames()
    decoded = cv.imdecode(np.frombuffer(frames[-1], np.uint8), cv.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)
