"""A synthetic face clip that moves along a known path.

The fixture every UI test runs on (``plans/03_phone_webapp/00_start.md`` Q7 c):
a public-domain portrait, warped frame by frame so that the midpoint between its
irises follows a path computed here, written as MJPEG.

Why each choice:

- **Generated, not recorded.** It runs anywhere, a cloud session included, and
  the true motion is known exactly. A recording never gives that.
- **The path is on the iris midpoint**, the point the eye position is computed
  from, so a test compares tracked pixels against :func:`face_path` directly.
- **Portrait, 720x1280.** The shape of a phone held upright.
- **MJPEG.** Chromium plays a ``.mjpeg`` file as a fake camera, and OpenCV writes
  one with no ffmpeg. At about 85 KB a frame it is a tenth the size of the
  ``.y4m`` alternative. The page can also split it into frames itself, which is
  what makes a frame-exact mode possible.
- **A minimum face size.** MediaPipe found no face at all in a probe where the
  face was about 110 px across in a 1280x720 frame, and every frame at about
  270 px (``plans/03_phone_webapp/tracking.md``, 2026-10-22). The scale never
  drops below :data:`MIN_FACE_WIDTH_PX`, and a test holds it there.

Run as ``python -m abyss.ui.clip``, which is what ``make clip`` does.
"""

from dataclasses import dataclass
import math
from pathlib import Path

import cv2 as cv
from loguru import logger as lg
import numpy as np

from abyss.params.abyss_params import get_abyss_paths
from abyss.ui.vendor import load_manifest
from abyss.ui.vendor import verified_file

FRAME_WIDTH = 720
FRAME_HEIGHT = 1280
FPS = 30
SEGMENT_FRAMES = 40
"""Frames per movement. Three movements, so four seconds at 30 fps."""

BACKGROUND = 128
JPEG_QUALITY = 90

PORTRAIT_IRIS_MID_PX = (406.39, 174.76)
"""Midpoint between the two iris centres in ``portrait.jpg``, in its pixels.

Measured once with the Python face landmarker (landmarks 468 and 473) on
2026-10-22. The path is defined on this point.
"""

PORTRAIT_FACE_WIDTH_PX = 191.75
"""Width of the landmark bounding box in ``portrait.jpg``, measured the same way."""

BASE_SCALE = 1.5
"""Portrait-to-frame scale at rest: a face of about 290 px across."""

SWEEP_X_PX = 140.0
SWEEP_Y_PX = 160.0
APPROACH_SCALE = 1.3
"""Peak scale relative to :data:`BASE_SCALE` during the approach movement."""

MIN_FACE_WIDTH_PX = 250.0
"""No frame has a face narrower than this. See the module docstring."""

REST_X = FRAME_WIDTH / 2
REST_Y = FRAME_HEIGHT * 0.4


class FrameOutsideClipError(IndexError):
    """Raised when a frame index is outside the clip."""

    def __init__(self, idx: int, count: int) -> None:
        """Initialise with the index and the clip length.

        Args:
            idx: The requested frame.
            count: Frames in the clip.
        """
        super().__init__(f"frame {idx} outside 0..{count - 1}")


class ClipEncodeError(RuntimeError):
    """Raised when OpenCV fails to encode a frame."""

    def __init__(self, idx: int) -> None:
        """Initialise with the frame index.

        Args:
            idx: The frame that failed.
        """
        super().__init__(f"JPEG encoding failed for frame {idx}")


@dataclass(frozen=True)
class PathPoint:
    """Where the iris midpoint is in one frame, and at what scale."""

    x_px: float
    y_px: float
    scale: float

    @property
    def face_width_px(self) -> float:
        """Apparent face width in this frame."""
        return PORTRAIT_FACE_WIDTH_PX * self.scale


def n_frames() -> int:
    """Total frames in the clip.

    Returns:
        Three movements of :data:`SEGMENT_FRAMES` each.
    """
    return 3 * SEGMENT_FRAMES


def face_path(idx: int) -> PathPoint:
    """Return the true position of the iris midpoint in a frame.

    Three movements from rest and back, each a full sine period so that it
    starts and ends still: left then right, up then down, closer then back.

    Args:
        idx: Frame index, ``0 <= idx < n_frames()``.

    Returns:
        The point and scale for that frame.

    Raises:
        FrameOutsideClipError: If the index is outside the clip.
    """
    if not 0 <= idx < n_frames():
        raise FrameOutsideClipError(idx, n_frames())
    segment, step = divmod(idx, SEGMENT_FRAMES)
    wave = math.sin(2 * math.pi * step / SEGMENT_FRAMES)
    x, y, scale = REST_X, REST_Y, BASE_SCALE
    if segment == 0:
        x -= SWEEP_X_PX * wave
    elif segment == 1:
        y -= SWEEP_Y_PX * wave
    else:
        scale *= 1 + (APPROACH_SCALE - 1) * max(wave, 0.0)
    return PathPoint(x, y, scale)


def render_frame(portrait: np.ndarray, point: PathPoint) -> np.ndarray:
    """Warp the portrait so its iris midpoint lands on the path point.

    Args:
        portrait: BGR portrait image.
        point: Where the iris midpoint goes, and the scale.

    Returns:
        A BGR frame of :data:`FRAME_WIDTH` x :data:`FRAME_HEIGHT`.
    """
    ix, iy = PORTRAIT_IRIS_MID_PX
    s = point.scale
    affine = np.array([[s, 0.0, point.x_px - s * ix], [0.0, s, point.y_px - s * iy]])
    return cv.warpAffine(
        portrait,
        affine,
        (FRAME_WIDTH, FRAME_HEIGHT),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=(BACKGROUND, BACKGROUND, BACKGROUND),
    )


def write_mjpeg(portrait: np.ndarray, path: Path) -> int:
    """Write the whole clip as concatenated JPEG frames.

    Args:
        portrait: BGR portrait image.
        path: Output file, conventionally ``.mjpeg``.

    Returns:
        Number of frames written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    params = [cv.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    with path.open("wb") as handle:
        for idx in range(n_frames()):
            ok, encoded = cv.imencode(
                ".jpg", render_frame(portrait, face_path(idx)), params
            )
            if not ok:
                raise ClipEncodeError(idx)
            handle.write(encoded.tobytes())
    return n_frames()


def split_mjpeg(data: bytes) -> list[bytes]:
    """Split concatenated JPEG frames, as written by :func:`write_mjpeg`.

    Splits on start-of-image markers. Safe for baseline JPEGs from OpenCV,
    whose entropy-coded data byte-stuffs every ``0xFF``.

    Args:
        data: The file's bytes.

    Returns:
        One ``bytes`` per frame.
    """
    soi = b"\xff\xd8\xff"
    starts = []
    pos = data.find(soi)
    while pos != -1:
        starts.append(pos)
        pos = data.find(soi, pos + 1)
    return [data[a:b] for a, b in zip(starts, [*starts[1:], len(data)], strict=True)]


def load_portrait() -> np.ndarray:
    """Read the vendored portrait, checked against its pin.

    Returns:
        BGR image.

    Raises:
        FileNotFoundError: If it cannot be decoded.
    """
    path = verified_file(load_manifest().file("portrait"))
    image = cv.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return image


def default_clip_path() -> Path:
    """Where ``make clip`` writes the clip.

    Returns:
        ``build/clips/synthetic_face.mjpeg``.
    """
    return get_abyss_paths().clips_fol / "synthetic_face.mjpeg"


def write_clip(path: Path | None = None) -> Path:
    """Write the clip, replacing any earlier one.

    Always rewritten rather than cached: it takes about a second, and a cached
    clip would silently disagree with :func:`face_path` after the path changed.

    Args:
        path: Output file. Defaults to :func:`default_clip_path`.

    Returns:
        The clip's path.
    """
    path = path or default_clip_path()
    count = write_mjpeg(load_portrait(), path)
    lg.info(f"Wrote {count} frames to {path}")
    return path


if __name__ == "__main__":
    write_clip()
