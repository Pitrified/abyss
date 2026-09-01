"""Sinks that write frames to disk.

Two implementations shipped together on purpose. A Protocol with a single
implementer is unvalidated - nothing proves the interface fits anything but the
one class that shaped it - and phase 5's window sink was a bad place to discover
that ``write`` should have taken an index, or that ``size`` should have been a
method. `PngSink` and `VideoSink` disagree enough to be a real test of the
shape: one writes many files and counts them, the other holds an open handle
that must be released.

It worked: `WindowSink` arrived in phase 5 and needed no change to the seam.

Everything here is safe on a headless box. The one sink that is not lives in
`window.py`.
"""

from pathlib import Path

import cv2 as cv
from loguru import logger as lg
import numpy as np

from abyss.config.sink import SinkConfig
from abyss.sink.base import check_size

FRAME_STEM = "frame"
"""Prefix for the per-frame file names written by :class:`PngSink`."""

FRAME_DIGITS = 5
"""Zero padding on frame numbers, so a shell glob sorts them correctly."""

VIDEO_FOURCC = "mp4v"
"""Codec for :class:`VideoSink`, the one already used elsewhere in the repo."""


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
        check_size(self.size, frame)
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
        check_size(self.size, frame)
        self.writer.write(frame)

    def close(self) -> None:
        """Release the writer, flushing the file to disk."""
        if self.writer.isOpened():
            self.writer.release()
            lg.info(f"Wrote {self.path}")
