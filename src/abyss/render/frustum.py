"""Off-axis projection: where the viewer is, to what they should see.

Head-coupled perspective makes the screen behave like a window rather than a
picture, by rebuilding the projection from wherever the eye actually is. The
maths is an asymmetric frustum, and it is four divisions. The delicate part is
that three coordinate frames meet here and two of them disagree about which way
is up.

**Camera frame**, what phase 1 produces: ``+X`` image right, ``+Y`` image
**down**, ``+Z`` away from the lens, metres. ``ScreenConfig.camera_to_centre_m``
is expressed in it too, which is why a camera sitting *above* a panel gives a
*positive* Y offset.

**Screen frame**, used here and nowhere else: origin at the panel centre, ``+X``
the viewer's right, ``+Y`` up, ``+Z`` out of the panel toward the viewer. The
scene lives at negative Z, behind the window.

Converting between them flips **both X and Y** and leaves Z alone. Y because the
two frames disagree about down. X because the camera looks *at* the viewer, so
the viewer's right hand lands on the left of an unmirrored image, the same way a
person facing you has their right hand on your left. Together that is a 180
degree rotation about Z, which is the right shape for two frames that face each
other.

The X sign additionally depends on whether the capture was mirrored, and that is
**already applied upstream** by :func:`abyss.viewer.eye_position.eye_position_m`.
Nothing here may flip it a second time: the bug would be invisible on a laptop
webcam and wrong on a front-facing phone.

Every public function here takes the eye in the **camera** frame. The
screen-frame vector is deliberately not something callers hold, so that the one
place the two frames can be confused stays inside this module (Q19).
"""

from dataclasses import dataclass

import numpy as np

from abyss.config.screen import ScreenConfig

DEFAULT_NEAR_M = 0.05
"""Near clip plane, well inside any plausible eye distance."""

DEFAULT_FAR_M = 100.0
"""Far clip plane. Nothing has a scene yet, so this is simply generous."""

CAMERA_TO_SCREEN = np.array([-1.0, -1.0, 1.0])
"""Axis signs taking a camera-frame vector to the screen frame.

See the module docstring: X and Y flip, Z does not.
"""


class EyeBehindScreenError(ValueError):
    """Raised when the eye is not in front of the screen."""

    def __init__(self, z_m: float) -> None:
        """Initialise with the offending depth.

        Args:
            z_m: Eye depth in the screen frame, in metres.
        """
        super().__init__(
            f"Eye must be in front of the screen, got z={z_m:.4f} m. At or "
            f"behind the panel there is no frustum to build"
        )


class InvalidClipPlanesError(ValueError):
    """Raised when the near and far planes cannot bound a frustum."""

    def __init__(self, near_m: float, far_m: float) -> None:
        """Initialise with the offending planes.

        Args:
            near_m: Near plane distance in metres.
            far_m: Far plane distance in metres.
        """
        super().__init__(f"Need 0 < near < far, got near={near_m} m, far={far_m} m")


class PointNotInFrontError(ValueError):
    """Raised when a point cannot be projected because it is not in front."""

    def __init__(self, count: int) -> None:
        """Initialise with how many points were behind the eye.

        Args:
            count: Number of offending points.
        """
        super().__init__(
            f"{count} point(s) at or behind the eye cannot be projected. "
            f"Clipping belongs to whoever builds the scene, not here"
        )


@dataclass(frozen=True)
class Frustum:
    """An asymmetric view frustum, as extents on the near plane.

    Args:
        left: Left extent at the near plane, in metres.
        right: Right extent at the near plane, in metres.
        bottom: Bottom extent at the near plane, in metres.
        top: Top extent at the near plane, in metres.
        near: Near plane distance in metres, positive.
        far: Far plane distance in metres, positive.
    """

    left: float
    right: float
    bottom: float
    top: float
    near: float
    far: float

    @property
    def is_symmetric(self) -> bool:
        """Whether the frustum is on axis, i.e. the eye faces the centre."""
        return bool(
            np.isclose(self.left, -self.right) and np.isclose(self.bottom, -self.top)
        )


def eye_in_screen_frame(
    eye_camera_m: np.ndarray,
    screen: ScreenConfig,
) -> np.ndarray:
    """Convert an eye position from the camera frame to the screen frame.

    The one interesting function in this module. Everything after it is
    arithmetic that works the first time.

    Args:
        eye_camera_m: Eye position ``[x, y, z]`` in the camera frame, metres.
        screen: The display, carrying the camera to panel centre offset.

    Returns:
        Eye position in the screen frame, metres. Not intended to travel
        outside this module: see the module docstring.
    """
    relative = np.asarray(eye_camera_m, dtype=float) - np.asarray(
        screen.camera_to_centre_m, dtype=float
    )
    return relative * CAMERA_TO_SCREEN


def frustum_for_eye(
    screen: ScreenConfig,
    eye_camera_m: np.ndarray,
    near_m: float = DEFAULT_NEAR_M,
    far_m: float = DEFAULT_FAR_M,
) -> Frustum:
    """Build the frustum that makes this screen a window for this eye.

    The screen rectangle sits at distance ``ez`` and the near plane at ``n``,
    so the extents are the rectangle scaled by ``n / ez``. The near plane is
    not the screen plane, and does not need to be.

    Args:
        screen: The display being looked through.
        eye_camera_m: Eye position in the **camera** frame, metres.
        near_m: Near clip plane distance in metres.
        far_m: Far clip plane distance in metres.

    Returns:
        The asymmetric frustum.

    Raises:
        InvalidClipPlanesError: If the planes do not satisfy 0 < near < far.
        EyeBehindScreenError: If the eye is not in front of the panel.
    """
    if near_m <= 0 or far_m <= near_m:
        raise InvalidClipPlanesError(near_m, far_m)

    eye = eye_in_screen_frame(eye_camera_m, screen)
    eye_x, eye_y, eye_z = (float(v) for v in eye)
    if eye_z <= 0:
        raise EyeBehindScreenError(eye_z)

    scale = near_m / eye_z
    half_w = screen.width_m / 2
    half_h = screen.height_m / 2
    return Frustum(
        left=(-half_w - eye_x) * scale,
        right=(half_w - eye_x) * scale,
        bottom=(-half_h - eye_y) * scale,
        top=(half_h - eye_y) * scale,
        near=near_m,
        far=far_m,
    )


def projection_matrix(frustum: Frustum) -> np.ndarray:
    """Build the 4x4 projection for a frustum, in ``glFrustum`` convention.

    Maps eye space, where the viewer sits at the origin looking down ``-Z``, to
    clip space. Compose it with a translation to get a matrix for screen frame
    points: :func:`view_projection_matrix` does that.

    Args:
        frustum: The extents to project.

    Returns:
        A 4x4 matrix.
    """
    left, right = frustum.left, frustum.right
    bottom, top = frustum.bottom, frustum.top
    near, far = frustum.near, frustum.far
    return np.array(
        [
            [2 * near / (right - left), 0, (right + left) / (right - left), 0],
            [0, 2 * near / (top - bottom), (top + bottom) / (top - bottom), 0],
            [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
            [0, 0, -1, 0],
        ],
        dtype=float,
    )


def view_projection_matrix(
    screen: ScreenConfig,
    eye_camera_m: np.ndarray,
    near_m: float = DEFAULT_NEAR_M,
    far_m: float = DEFAULT_FAR_M,
) -> np.ndarray:
    """Build the matrix taking **screen frame** world points to clip space.

    This is the one phase 4 wants. It folds in the translation putting the eye
    at the origin, so the caller works in the screen frame throughout and never
    handles an eye-space coordinate.

    Args:
        screen: The display being looked through.
        eye_camera_m: Eye position in the **camera** frame, metres.
        near_m: Near clip plane distance in metres.
        far_m: Far clip plane distance in metres.

    Returns:
        A 4x4 matrix.
    """
    frustum = frustum_for_eye(screen, eye_camera_m, near_m, far_m)
    eye = eye_in_screen_frame(eye_camera_m, screen)

    translation = np.eye(4)
    translation[:3, 3] = -eye
    return projection_matrix(frustum) @ translation


def project_points(
    matrix: np.ndarray,
    points_m: np.ndarray,
    width_px: int,
    height_px: int,
) -> np.ndarray:
    """Project screen frame points to pixel coordinates.

    Does the perspective divide and the viewport transform, so that phase 4
    does not reimplement either. Pixel Y runs **down**, as images do, which is
    the last sign flip in the chain.

    No clipping: a point outside the frustum projects to a pixel outside the
    image rather than being dropped, because what to do about that belongs to
    whoever builds the scene.

    Args:
        matrix: A view projection matrix from :func:`view_projection_matrix`.
        points_m: Points ``(N, 3)`` in the screen frame, metres.
        width_px: Output image width in pixels.
        height_px: Output image height in pixels.

    Returns:
        Pixel coordinates ``(N, 2)``.

    Raises:
        PointNotInFrontError: If any point is at or behind the eye, where the
            perspective divide is undefined.
    """
    points = np.atleast_2d(np.asarray(points_m, dtype=float))
    homogeneous = np.hstack([points, np.ones((len(points), 1))])
    clip = homogeneous @ matrix.T

    w = clip[:, 3]
    behind = int(np.sum(w <= 0))
    if behind:
        raise PointNotInFrontError(behind)

    ndc = clip[:, :3] / w[:, None]
    pixels = np.empty((len(points), 2))
    pixels[:, 0] = (ndc[:, 0] + 1) / 2 * width_px
    pixels[:, 1] = (1 - ndc[:, 1]) / 2 * height_px
    return pixels
