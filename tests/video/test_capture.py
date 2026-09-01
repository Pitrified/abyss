"""Test the capture checks, with no camera present.

That constraint is the point rather than a limitation: the suite has to pass on
a box with no camera, so the parts worth testing are the ones that judge a
frame. `open_camera` itself is not tested here - it is three `set` calls and a
readback against real hardware, and a mock of it would only assert that the
lines were written in the order they were written in.
"""

from unittest.mock import MagicMock

import cv2 as cv
import numpy as np
import pytest

from abyss.video.capture import BLACK_MEAN_MAX
from abyss.video.capture import BLACK_STD_MAX
from abyss.video.capture import CameraStream
from abyss.video.capture import CaptureIsBlackError
from abyss.video.capture import CaptureModeError
from abyss.video.capture import CaptureReadError
from abyss.video.capture import check_frame_is_live
from abyss.video.capture import check_frame_size

SIZE = (1280, 720)
RNG = np.random.default_rng(0)


def dead_frame(size: tuple[int, int] = SIZE) -> np.ndarray:
    """Build a frame like the ones a cut camera returns.

    Mean 10.7 and standard deviation 2 are what was measured off g7 with the
    session locked, so the fixture is the observation rather than a zero array
    that would be easier to detect than the real thing.

    Args:
        size: The ``(width, height)`` to build.

    Returns:
        A flat, dark BGR frame.
    """
    width, height = size
    noise = RNG.normal(10.7, 2.0, (height, width, 3))
    return np.clip(noise, 0, 255).astype(np.uint8)


def live_frame(size: tuple[int, int] = SIZE) -> np.ndarray:
    """Build a frame with content in it.

    Args:
        size: The ``(width, height)`` to build.

    Returns:
        A structured BGR frame.
    """
    width, height = size
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[: height // 2] = 200
    return frame


def test_a_dead_frame_is_refused() -> None:
    """The failure that reports success, which is why this exists."""
    with pytest.raises(CaptureIsBlackError):
        check_frame_is_live("g7", dead_frame())


def test_a_frame_with_content_passes() -> None:
    """The other half, so the check is not just always raising."""
    check_frame_is_live("g7", live_frame())


def test_a_dark_but_structured_frame_passes() -> None:
    """Dark alone is a dim room and must not raise.

    Both conditions are required together, and this is the test that pins it:
    the mean here is under the threshold, so a check on brightness alone would
    reject a viewer sitting in a badly lit room and call the camera dead.
    """
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:, ::4] = 60

    assert frame.mean() < BLACK_MEAN_MAX
    assert frame.std() > BLACK_STD_MAX
    check_frame_is_live("g7", frame)


def test_a_flat_but_bright_frame_passes() -> None:
    """Flat alone is a blank wall in good light, and must not raise either."""
    frame = np.full((720, 1280, 3), 180, dtype=np.uint8)

    assert frame.std() < BLACK_STD_MAX
    check_frame_is_live("g7", frame)


def test_the_wrong_frame_size_is_refused() -> None:
    """The YUYV clamp, which reports success and hands back 640x480.

    Left unchecked it does not fail, it rescales the measured focal length to a
    resolution it was never measured at and reports a different depth.
    """
    with pytest.raises(CaptureModeError):
        check_frame_size(SIZE, live_frame((640, 480)))


def test_the_right_frame_size_passes() -> None:
    """The size is read off the frame, not off the capture properties."""
    check_frame_size(SIZE, live_frame())


def test_the_stream_returns_a_checked_frame() -> None:
    """The happy path, wired through the stream rather than the free checks."""
    capture = MagicMock(spec=cv.VideoCapture)
    capture.read.return_value = (True, live_frame())

    with CameraStream(capture, SIZE, name="test") as stream:
        assert stream.read().shape == (720, 1280, 3)
    capture.release.assert_called_once()


def test_the_stream_refuses_a_dead_frame() -> None:
    """A camera that dies mid-run must say so rather than look boring."""
    capture = MagicMock(spec=cv.VideoCapture)
    capture.read.return_value = (True, dead_frame())

    with pytest.raises(CaptureIsBlackError), CameraStream(capture, SIZE) as stream:
        stream.read()


def test_a_read_that_fails_is_not_a_black_frame() -> None:
    """The two failures are distinct and get distinct errors.

    `ok=False` means the capture stopped; a black frame means it did not stop
    and that is the whole problem. Collapsing them would lose the distinction
    the module exists to draw.
    """
    capture = MagicMock(spec=cv.VideoCapture)
    capture.read.return_value = (False, None)

    with pytest.raises(CaptureReadError), CameraStream(capture, SIZE) as stream:
        stream.read()


def test_the_device_is_released_when_the_body_raises() -> None:
    """A held camera is not released by the process exiting on some machines."""
    capture = MagicMock(spec=cv.VideoCapture)
    capture.read.return_value = (True, dead_frame())

    with pytest.raises(CaptureIsBlackError), CameraStream(capture, SIZE) as stream:
        stream.read()
    capture.release.assert_called_once()
