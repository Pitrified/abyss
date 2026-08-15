"""What a capture device contributes to the geometry.

The one number anything downstream consumes is a focal length in pixels, and it
is **not** a property of the camera alone: it scales with the frame height, so
1100 px at 1280x720 is 1650 px on the same lens at 1920x1080. The camera is
therefore described by its field of view, which is resolution independent, and
:class:`FrameGeometry` binds that description to an actual frame.

Measured facts behind the fallback, from the phase 1 plan
(``plans/01_abyss_expansion/01_viewer_position.md``):

- MediaPipe's landmarker receives no intrinsics, so it assumes a 63 degree
  vertical field of view. Its focal length follows the frame height,
  ``(H / 2) / tan(31.5 deg)``, confirmed on a 1080-tall and a 1920-tall clip to
  a consistent 2%.
- Any change to the frame height changes the depth MediaPipe reports. Padding a
  1920x1080 frame to 1920x1920 with identical content moved reported depth from
  0.507 m to 0.881 m. Letterboxing is not a no-op.
"""

from dataclasses import dataclass
import math

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

MEDIAPIPE_VERTICAL_FOV_DEG = 63.0
"""Vertical field of view MediaPipe assumes when it is given no intrinsics."""

STRAIGHT_ANGLE_DEG = 180.0
"""Upper bound for a field of view: at 180 degrees the focal length vanishes."""


class InvalidFrameSizeError(ValueError):
    """Raised when a frame size is not usable as a camera geometry."""

    def __init__(self, width: int, height: int) -> None:
        """Initialise with the offending size.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
        """
        super().__init__(f"Frame size must be positive, got {width}x{height}")


class AmbiguousIntrinsicsError(ValueError):
    """Raised when a camera is given both a field of view and a focal length."""

    def __init__(self, name: str) -> None:
        """Initialise with the offending camera name.

        Args:
            name: Name of the camera entry.
        """
        super().__init__(
            f"Camera {name!r} sets both fov_vertical_deg and focal_px; "
            f"they are two ways to say the same thing, so set one"
        )


class MissingFocalHeightError(ValueError):
    """Raised when a measured focal length does not say what height it is for."""

    def __init__(self, name: str) -> None:
        """Initialise with the offending camera name.

        Args:
            name: Name of the camera entry.
        """
        super().__init__(
            f"Camera {name!r} sets focal_px without focal_measured_at_height; "
            f"a focal length in pixels is meaningless without its resolution"
        )


class CameraConfig(BaseModel):
    """A capture device, described independently of any particular frame.

    Note there is no resolution field. One camera feeds several streams at
    several resolutions, so the frame size arrives with the frames and is bound
    in through :class:`FrameGeometry`. The only resolution stored here is the
    one a measured focal length was measured at, which is what makes that
    measurement interpretable.

    Args:
        name: Device name, used in logs and as its registry key.
        fov_vertical_deg: Vertical field of view in degrees. **Vertical**: a
            published spec is almost always the diagonal, and feeding a
            diagonal figure in unconverted is roughly 20% wrong on a 16:9
            sensor. ``None`` falls back to MediaPipe's own assumption.
        focal_px: Focal length in pixels, if it has been measured. Requires
            ``focal_measured_at_height``.
        focal_measured_at_height: Frame height ``focal_px`` was measured at.
        mirrored: Whether the capture is mirrored, as a front-facing phone
            camera usually is. Flips the sign of X and nothing else.
        provenance: Where the numbers came from, in words. Say "provisional"
            when they are a published spec rather than a measurement.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    fov_vertical_deg: float | None = Field(default=None, gt=0, lt=STRAIGHT_ANGLE_DEG)
    focal_px: float | None = Field(default=None, gt=0)
    focal_measured_at_height: int | None = Field(default=None, gt=0)
    mirrored: bool = False
    provenance: str = "unmeasured, using MediaPipe's assumed field of view"

    @model_validator(mode="after")
    def _check_intrinsics(self) -> "CameraConfig":
        """Reject intrinsics that are ambiguous or uninterpretable.

        Note that pydantic re-wraps whatever a validator raises into a
        ``ValidationError``, so the caller sees that type and the message, not
        the classes below. They exist to say what went wrong at the point it
        went wrong, which is the half of the rule that survives.

        Returns:
            The validated model.

        Raises:
            AmbiguousIntrinsicsError: If both a field of view and a focal
                length are given.
            MissingFocalHeightError: If a focal length is given without the
                height it was measured at.
        """
        if self.fov_vertical_deg is not None and self.focal_px is not None:
            raise AmbiguousIntrinsicsError(self.name)
        if self.focal_px is not None and self.focal_measured_at_height is None:
            raise MissingFocalHeightError(self.name)
        return self

    @property
    def is_measured(self) -> bool:
        """Whether this camera has real intrinsics rather than the fallback."""
        return self.fov_vertical_deg is not None or self.focal_px is not None

    def focal_px_for_height(self, height: int) -> float:
        """Focal length in pixels for a frame of this height.

        Args:
            height: Frame height in pixels.

        Returns:
            The measured focal length rescaled to this height, or the one the
            field of view implies, or MediaPipe's assumption when neither is
            known.

        Raises:
            InvalidFrameSizeError: If the height is not positive.
        """
        if height <= 0:
            raise InvalidFrameSizeError(height, height)
        if self.focal_px is not None and self.focal_measured_at_height is not None:
            # Focal length in pixels scales with the sampling of the sensor.
            return self.focal_px * height / self.focal_measured_at_height
        fov = self.fov_vertical_deg
        if fov is None:
            fov = MEDIAPIPE_VERTICAL_FOV_DEG
        return (height / 2) / math.tan(math.radians(fov / 2))


@dataclass(frozen=True)
class FrameGeometry:
    """A camera bound to the frames it actually produced.

    Not config: this is built at run time, once the frame size is known, and it
    is what the conversion functions consume. It exists because a focal length
    and a principal point are both functions of the frame, not of the device.

    Args:
        camera: The device the frames came from.
        width: Frame width in pixels.
        height: Frame height in pixels.
    """

    camera: CameraConfig
    width: int
    height: int

    def __post_init__(self) -> None:
        """Validate the frame size.

        Raises:
            InvalidFrameSizeError: If either dimension is not positive.
        """
        if self.width <= 0 or self.height <= 0:
            raise InvalidFrameSizeError(self.width, self.height)

    @property
    def focal(self) -> float:
        """Focal length in pixels for this frame size."""
        return self.camera.focal_px_for_height(self.height)

    @property
    def principal_point(self) -> tuple[float, float]:
        """Principal point in pixels, assumed to be the frame centre."""
        return self.width / 2, self.height / 2

    @property
    def mirrored(self) -> bool:
        """Whether the capture is mirrored."""
        return self.camera.mirrored
