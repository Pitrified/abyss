"""Where a finished frame goes.

The second seam of phase 4, and the one Q13 argued for: a sink acts on a frame
that already exists, so it knows nothing about screens, eyes or projections.

Two implementations ship together on purpose. A Protocol with a single
implementer is unvalidated - nothing proves the interface fits anything but the
one class that shaped it - and phase 5's window sink is a bad place to discover
that ``write`` should have taken an index, or that ``size`` should have been a
method. `PngSink` and `VideoSink` disagree enough to be a real test of the
shape: one writes many files and counts them, the other holds an open handle
that must be released.

``size`` lives on the protocol rather than being passed alongside it because
the sink is what knows how big a frame it accepts (Q20). `PngSink` reads it
from its config; phase 5's window sink will read it from the window it opened.
"""

from pathlib import Path
from typing import Protocol
from typing import runtime_checkable

import cv2 as cv
from loguru import logger as lg
import numpy as np

from abyss.config.sink import SinkConfig

FRAME_STEM = "frame"
"""Prefix for the per-frame file names written by :class:`PngSink`."""

FRAME_DIGITS = 5
"""Zero padding on frame numbers, so a shell glob sorts them correctly."""

VIDEO_FOURCC = "mp4v"
"""Codec for :class:`VideoSink`, the one already used elsewhere in the repo."""


class FrameSizeMismatchError(ValueError):
    """Raised when a frame handed to a sink is not the size it expects."""

    def __init__(self, expected: tuple[int, int], actual: tuple[int, int]) -> None:
        """Initialise with both sizes.

        Args:
            expected: The ``(width, height)`` the sink was configured for.
            actual: The ``(width, height)`` of the offending frame.
        """
        super().__init__(
            f"Sink expects {expected[0]}x{expected[1]} frames, got "
            f"{actual[0]}x{actual[1]}"
        )


class VideoSinkOpenError(RuntimeError):
    """Raised when the video writer cannot be opened."""

    def __init__(self, path: Path) -> None:
        """Initialise with the path that could not be opened.

        Args:
            path: Where the video was to be written.
        """
        super().__init__(
            f"Could not open a video writer for {path} with codec "
            f"{VIDEO_FOURCC!r}. The codec may be missing from this OpenCV build"
        )


@runtime_checkable
class Sink(Protocol):
    """Somewhere finished frames go.

    Implementations are not expected to be reusable after :meth:`close`.
    """

    @property
    def size(self) -> tuple[int, int]:
        """Frame size this sink accepts, as ``(width_px, height_px)``."""
        ...

    def write(self, frame: np.ndarray) -> None:
        """Accept one finished frame.

        Args:
            frame: A BGR image of exactly :attr:`size`.
        """
        ...

    def close(self) -> None:
        """Release whatever the sink holds. Safe to call more than once."""
        ...


def _check_size(expected: tuple[int, int], frame: np.ndarray) -> None:
    """Reject a frame that is not the size the sink was built for.

    Args:
        expected: The ``(width, height)`` the sink accepts.
        frame: The frame handed in.

    Raises:
        FrameSizeMismatchError: If the frame is a different size.
    """
    height, width = frame.shape[:2]
    if (width, height) != expected:
        raise FrameSizeMismatchError(expected, (width, height))


class PngSink:
    """Write each frame as a numbered PNG.

    Args:
        config: Where to write, and how big the frames are.
    """

    def __init__(self, config: SinkConfig) -> None:
        """Create the output folder and start the frame counter.

        Args:
            config: Where to write, and how big the frames are.
        """
        self.config = config
        self.fol = config.out_fol
        self.fol.mkdir(parents=True, exist_ok=True)
        self.count = 0

    @property
    def size(self) -> tuple[int, int]:
        """Frame size this sink accepts, as ``(width_px, height_px)``."""
        return self.config.size

    def write(self, frame: np.ndarray) -> None:
        """Write one frame as the next numbered PNG.

        Args:
            frame: A BGR image of exactly :attr:`size`.
        """
        _check_size(self.size, frame)
        path = self.fol / f"{FRAME_STEM}_{self.count:0{FRAME_DIGITS}d}.png"
        cv.imwrite(str(path), frame)
        self.count += 1

    def close(self) -> None:
        """Report what was written. There is no handle to release."""
        lg.info(f"Wrote {self.count} frames to {self.fol}")


class VideoSink:
    """Write the frames as a single video file.

    Deliberately thin: a path, a frame rate, and the repo's usual codec. Codec
    choice, quality and per-frame timing are all absent, and wanting any of
    them is the signal that this has stopped being a second implementation of
    the protocol and become a feature.

    Args:
        config: Where to write, how big the frames are, and at what rate.
    """

    def __init__(self, config: SinkConfig) -> None:
        """Open the writer.

        Args:
            config: Where to write, how big the frames are, and at what rate.

        Raises:
            VideoSinkOpenError: If OpenCV cannot open the writer.
        """
        self.config = config
        config.out_fol.mkdir(parents=True, exist_ok=True)
        self.path = config.out_fol / f"{config.name}.mp4"
        self.writer = cv.VideoWriter(
            str(self.path),
            cv.VideoWriter.fourcc(*VIDEO_FOURCC),
            config.fps,
            config.size,
        )
        if not self.writer.isOpened():
            raise VideoSinkOpenError(self.path)

    @property
    def size(self) -> tuple[int, int]:
        """Frame size this sink accepts, as ``(width_px, height_px)``."""
        return self.config.size

    def write(self, frame: np.ndarray) -> None:
        """Append one frame to the video.

        Args:
            frame: A BGR image of exactly :attr:`size`.
        """
        _check_size(self.size, frame)
        self.writer.write(frame)

    def close(self) -> None:
        """Release the writer, flushing the file to disk."""
        if self.writer.isOpened():
            self.writer.release()
            lg.info(f"Wrote {self.path}")
