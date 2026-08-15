"""Camera assumptions for the viewer pipeline.

Placeholder for the camera config phase 2 will introduce. It holds five
numbers and no behaviour beyond deriving the focal length, and it exists so
that every assumption the eye position depends on sits in one visible place
instead of being scattered through the extraction code.

Measured facts behind the defaults, recorded in the phase 1 plan
(``plans/01_abyss_expansion/01_viewer_position.md``):

- MediaPipe's landmarker receives no intrinsics, so it assumes a 63 degree
  vertical field of view. Its focal length therefore follows the frame height,
  ``(H / 2) / tan(31.5 deg)``, confirmed on a 1080-tall and a 1920-tall clip to
  a consistent 2%. It is computed, never configured.
- Any change to the frame height changes the depth MediaPipe reports. Padding a
  1920x1080 frame to 1920x1920 with identical content moved reported depth from
  0.507 m to 0.881 m.
"""

from dataclasses import dataclass
import math

MEDIAPIPE_VERTICAL_FOV_DEG = 63.0
"""Vertical field of view MediaPipe assumes when it is given no intrinsics."""

DEFAULT_IPD_M = 0.063
"""Mean adult interpupillary distance, the scale reference chosen in Q6."""


class InvalidFrameSizeError(ValueError):
    """Raised when a frame size is not usable as a camera geometry."""

    def __init__(self, width: int, height: int) -> None:
        """Initialise with the offending size.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
        """
        super().__init__(f"Frame size must be positive, got {width}x{height}")


@dataclass(frozen=True)
class CameraAssumptions:
    """What the viewer pipeline assumes about the camera and the viewer.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.
        focal_px: True focal length in pixels. ``None`` means "unknown", and
            MediaPipe's own assumed focal length is used instead, which leaves
            depth scaled by ``f_real / f_assumed``.
        ipd_m: Interpupillary distance of the viewer, in metres.
        mirrored: Whether the capture is mirrored, as a front-facing phone
            camera usually is. Flips the sign of X and nothing else.
    """

    width: int
    height: int
    focal_px: float | None = None
    ipd_m: float = DEFAULT_IPD_M
    mirrored: bool = False

    def __post_init__(self) -> None:
        """Validate the frame size.

        Raises:
            InvalidFrameSizeError: If either dimension is not positive.
        """
        if self.width <= 0 or self.height <= 0:
            raise InvalidFrameSizeError(self.width, self.height)

    @property
    def assumed_focal_px(self) -> float:
        """Focal length MediaPipe is using, derived from the frame height."""
        half_fov_rad = math.radians(MEDIAPIPE_VERTICAL_FOV_DEG / 2)
        return (self.height / 2) / math.tan(half_fov_rad)

    @property
    def focal(self) -> float:
        """Focal length to convert pixels with: the real one if it is known."""
        return self.focal_px if self.focal_px is not None else self.assumed_focal_px

    @property
    def principal_point(self) -> tuple[float, float]:
        """Principal point in pixels, assumed to be the frame centre."""
        return self.width / 2, self.height / 2
