"""Opening a camera in a known mode, and noticing when it dies.

Step three of phase 5. Three findings from the calibration sessions are built in
here rather than left to be rediscovered, and each one cost real time to learn
(``plans/01_abyss_expansion/tracking.md``):

- **The pixel format decides the resolution.** OpenCV defaults to YUYV, which
  caps at 640x480 on the g7 webcam and silently ignores a request for anything
  larger, returning 640x480 while reporting success. MJPG reaches 1280x720,
  which is the mode the focal length was measured at. The fourcc must be set
  **before** the frame size or the size is clamped back.
- **The queue is four frames deep.** Offline that produced four identical
  calibration views; live it is 160 ms between the viewer moving and the camera
  admitting it. `CAP_PROP_BUFFERSIZE` of 1 is the fix for a loop that reads
  continuously. A reader that goes idle and comes back still needs an explicit
  flush, which is `calibrate_camera.py`'s problem and not this module's.
- **A dead camera claims success.** With the session locked, `read()` returned
  ``ok=True`` and black frames: mean 10.7 of 255, standard deviation 2, flat
  across 90 consecutive frames. Downstream that is indistinguishable from "no
  face", so the loop would hold a stale position and look boring rather than
  broken. The frame itself has to be checked, not the return flag.

The checks are free functions over a frame so they can be tested with no camera,
which is what keeps this phase's suite runnable on a box that has none.
"""

from pathlib import Path
from types import TracebackType
from typing import Self

import cv2 as cv
from loguru import logger as lg
import numpy as np

MJPG_FOURCC = "MJPG"
"""The only mode on the g7 webcam that reaches 1280x720."""

BLACK_MEAN_MAX = 16.0
"""Mean intensity at or below which a frame is a candidate for being dead.

The measured dead frames sat at 10.7 of 255. This is deliberately not the only
condition: a genuinely dark room is dim and structured, while a dead capture is
dim and flat.
"""

BLACK_STD_MAX = 5.0
"""Standard deviation below which a frame carries no structure at all.

The measured dead frames sat at 2. A real frame of a lit face is an order of
magnitude above this, and so is an underexposed one.
"""

BLACK_CHECK_STRIDE = 8
"""Sample every Nth pixel for the liveness check.

A 1280x720 frame is 2.8 million values and this runs every frame inside a 33 ms
budget, so it takes a strided view instead: 160x90 samples is 43 thousand,
enough to tell a flat frame from a structured one.

Measured rather than assumed, because the margin is not small: the strided
check is **0.26 ms** and mean-plus-standard-deviation over the whole frame is
**7.3 ms**. Reading every pixel to answer "is anything there" would have cost a
fifth of the frame budget, which is the same shape of mistake as the background
fill Q29 removed. Checking whether a frame is blank does not need every pixel.
"""


class CaptureOpenError(RuntimeError):
    """Raised when a capture device cannot be opened at all."""

    def __init__(self, source: int | Path) -> None:
        """Initialise with the source that would not open.

        Args:
            source: Device index or path that was requested.
        """
        super().__init__(f"Could not open capture source {source!r}")


class CaptureModeError(RuntimeError):
    """Raised when a camera did not give back the mode it was asked for.

    Worth its own error because the alternative is silent: the camera reports
    success, hands over a smaller frame, and `focal_px_for_height` cheerfully
    rescales the measured focal length to a resolution it was never measured
    at, changing the reported depth rather than failing.
    """

    def __init__(self, wanted: tuple[int, int], got: tuple[int, int]) -> None:
        """Initialise with both modes.

        Args:
            wanted: The ``(width, height)`` requested.
            got: The ``(width, height)`` the camera actually produced.
        """
        super().__init__(
            f"Asked the camera for {wanted[0]}x{wanted[1]} and got "
            f"{got[0]}x{got[1]}. The pixel format is the usual cause: YUYV caps "
            f"at 640x480 and ignores larger requests without failing"
        )


class CaptureReadError(RuntimeError):
    """Raised when a capture stops handing over frames."""

    def __init__(self, name: str) -> None:
        """Initialise with the stream's name.

        Args:
            name: Name of the stream that stopped.
        """
        super().__init__(f"Capture {name!r} returned no frame")


class CaptureIsBlackError(RuntimeError):
    """Raised when a capture returns frames with nothing in them.

    The failure this exists for reports success. A camera cut out from under
    the process - a lock screen, a suspended session, a lens cap - keeps
    returning ``ok=True`` with flat dark frames, and a face tracker fed those
    reports "no face" rather than an error.
    """

    def __init__(self, name: str, mean: float, std: float) -> None:
        """Initialise with what the frame measured.

        Args:
            name: Name of the stream that produced it.
            mean: Mean intensity of the sampled pixels.
            std: Standard deviation of the sampled pixels.
        """
        super().__init__(
            f"Capture {name!r} returned a frame with no content: mean {mean:.1f}, "
            f"standard deviation {std:.1f}. The camera is likely gone rather "
            f"than looking at something dark"
        )


def check_frame_size(wanted: tuple[int, int], frame: np.ndarray) -> None:
    """Reject a frame that is not the size the camera was asked for.

    Checked against the frame rather than against `CAP_PROP_FRAME_WIDTH`,
    because the properties are what lied in the first place.

    Args:
        wanted: The ``(width, height)`` requested.
        frame: A frame from the capture.

    Raises:
        CaptureModeError: If the frame is a different size.
    """
    height, width = frame.shape[:2]
    if (width, height) != wanted:
        raise CaptureModeError(wanted, (width, height))


def check_frame_is_live(name: str, frame: np.ndarray) -> None:
    """Reject a frame that is flat and dark, which means the camera is gone.

    Both conditions are required. Dark alone is a dim room, and flat alone is a
    blank wall in good light; a dead capture is both at once.

    Args:
        name: Name of the stream, for the message.
        frame: A frame from the capture.

    Raises:
        CaptureIsBlackError: If the frame carries no content.
    """
    sampled = frame[::BLACK_CHECK_STRIDE, ::BLACK_CHECK_STRIDE]
    mean = float(np.mean(sampled))
    std = float(np.std(sampled))
    if mean <= BLACK_MEAN_MAX and std <= BLACK_STD_MAX:
        raise CaptureIsBlackError(name, mean, std)


def open_camera(source: int | Path, size: tuple[int, int]) -> cv.VideoCapture:
    """Open a camera pinned to one mode.

    Args:
        source: Device index, or a path for a file.
        size: The ``(width, height)`` to request.

    Returns:
        The opened capture.

    Raises:
        CaptureOpenError: If the device will not open.
        CaptureModeError: If the camera reports a different mode than asked.
    """
    width, height = size
    capture = cv.VideoCapture(source if isinstance(source, int) else str(source))
    if not capture.isOpened():
        raise CaptureOpenError(source)

    # Order is load bearing: the format caps what sizes are on offer, so a size
    # set first is clamped to whatever the default format allows.
    capture.set(cv.CAP_PROP_FOURCC, cv.VideoWriter.fourcc(*MJPG_FOURCC))
    capture.set(cv.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv.CAP_PROP_BUFFERSIZE, 1)

    got = (
        int(capture.get(cv.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv.CAP_PROP_FRAME_HEIGHT)),
    )
    if got != size:
        capture.release()
        raise CaptureModeError(size, got)

    lg.info(f"Opened {source!r} at {width}x{height} {MJPG_FOURCC}, buffer size 1")
    return capture


class CameraStream:
    """A capture that hands over frames, or says why it cannot.

    Takes an already-open capture rather than opening one itself, which is what
    lets the checks be exercised with no camera present. :func:`open_camera` is
    the other half.

    Args:
        capture: An open capture.
        size: The ``(width, height)`` every frame must be.
        name: Name for logs and error messages.
    """

    def __init__(
        self,
        capture: cv.VideoCapture,
        size: tuple[int, int],
        name: str = "camera",
    ) -> None:
        """Initialise with an open capture.

        Args:
            capture: An open capture.
            size: The ``(width, height)`` every frame must be.
            name: Name for logs and error messages.
        """
        self.capture = capture
        self.size = size
        self.name = name

    def read(self) -> np.ndarray:
        """Take the next frame, checking that it is one.

        Returns:
            A BGR frame of :attr:`size`.

        Raises:
            CaptureReadError: If the capture handed over nothing.
            CaptureModeError: If the frame is the wrong size.
            CaptureIsBlackError: If the frame has no content.
        """
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise CaptureReadError(self.name)
        check_frame_size(self.size, frame)
        check_frame_is_live(self.name, frame)
        return frame

    def close(self) -> None:
        """Release the device."""
        self.capture.release()
        lg.info(f"Released capture {self.name!r}")

    def __enter__(self) -> Self:
        """Enter the context.

        Returns:
            This stream.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the device on the way out.

        Args:
            exc_type: Exception type, if one is propagating.
            exc: The exception, if one is propagating.
            traceback: Its traceback, if one is propagating.
        """
        self.close()
