"""Utility functions for opencv."""

import math
from pathlib import Path
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def resize(
    image,
    desired_width=640,
    desired_height=480,
) -> np.ndarray:
    """Resize the image to the desired width and height.

    https://colab.research.google.com/drive/1uCuA6We9T5r0WljspEHWPHXCT_2bMKUy
    """
    h, w = image.shape[:2]
    if h < w:
        img = cv.resize(
            image,
            (desired_width, math.floor(h / (w / desired_width))),
        )
    else:
        img = cv.resize(
            image,
            (math.floor(w / (h / desired_height)), desired_height),
        )
    return img


def cv_imshow_rgb(winname: str, image_rgb: np.ndarray) -> None:
    """Show a RGB image in an opencv window."""
    image_bgr = cv.cvtColor(image_rgb, cv.COLOR_RGB2BGR)
    cv.imshow(winname, image_bgr)
