"""A frame is a video frame.

With a timestamp and an index.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import cv2 as cv
import mediapipe as mp
from mediapipe.python import Image  # type: ignore
from mediapipe.python import ImageFormat  # type: ignore
import numpy as np


@dataclass
class Frame:
    """A frame is a video frame.

    A MediaPipe Image is used to store the frame data.
    See
    https://cs.opensource.google/mediapipe/mediapipe/+/master:mediapipe/python/pybind/image.cc
    .

    It has the following attributes:
        * image.width
        * image.height
        * image.channels
        * image.step
        * image.image_format
        * image.numpy_view()

    Args:
        image: MediaPipe Image. Uses RGB format.
        msec: Timestamp in milliseconds.
        idx: Index of the frame.
    """

    image: Image
    msec: float
    idx: int

    @classmethod
    def from_np_array(
        cls,
        array: np.ndarray,
        msec: float = 0,
        idx: int = 0,
    ) -> Self:
        """Create a frame from a numpy array."""
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=array)
        return cls(image, msec, idx)

    @classmethod
    def from_opencv(
        cls,
        array: np.ndarray,
        msec: float = 0,
        idx: int = 0,
    ) -> Self:
        """Create a frame from an OpenCV array."""
        image = cv.cvtColor(array, cv.COLOR_BGR2RGB)
        return cls.from_np_array(image, msec, idx)

    @classmethod
    def from_file(
        cls,
        image_path: Path,
        msec: float = 0,
        idx: int = 0,
    ) -> Self:
        """Create a frame from a file."""
        image = mp.Image.create_from_file(str(image_path))
        return cls(image, msec, idx)

    def to_numpy(self) -> np.ndarray:
        """Convert a frame to a numpy array."""
        return self.image.numpy_view()

    def to_opencv(self) -> np.ndarray:
        """Convert a frame to an OpenCV array."""
        return cv.cvtColor(self.to_numpy(), cv.COLOR_RGB2BGR)

    def __str__(self) -> str:
        """Return the string representation of a frame."""
        return f"Frame(idx={self.idx}, msec={self.msec:.3f})"

    def __repr__(self) -> str:
        """Return a detailed string representation of a frame."""
        return str(self)
