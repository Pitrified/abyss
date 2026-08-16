"""Test the off-axis projection.

Two families of test here, and neither substitutes for the other. Established
by breaking each sign in turn and recording what went red, rather than assumed.

:func:`test_screen_corners_map_to_viewport_corners` is a **self-consistency**
check: it pins that the frustum, the matrix and the viewport transform agree.
It catches near-plane scaling, a wrong matrix and a missing eye translation,
where all 45 swept cases fail at once. It is **blind to the camera to screen
conversion**, because a wrong eye position produces a frustum that is wrong to
match, and the corners still fill the image perfectly.

The directional tests are what pin the conversion, by asserting which way the
world moves. Flipping an axis sign fails exactly three tests, and all three are
of that family.
"""

import numpy as np
import pytest

from abyss.config.screen import ScreenConfig
from abyss.params.abyss_devices import get_screen
from abyss.render.frustum import DEFAULT_FAR_M
from abyss.render.frustum import DEFAULT_NEAR_M
from abyss.render.frustum import EyeBehindScreenError
from abyss.render.frustum import InvalidClipPlanesError
from abyss.render.frustum import PointNotInFrontError
from abyss.render.frustum import eye_in_screen_frame
from abyss.render.frustum import frustum_for_eye
from abyss.render.frustum import project_points
from abyss.render.frustum import projection_matrix
from abyss.render.frustum import view_projection_matrix

WIDTH_PX, HEIGHT_PX = 1280, 720


@pytest.fixture
def screen() -> ScreenConfig:
    """Build a screen with the camera centred above it, for simple arithmetic."""
    return ScreenConfig(
        name="test",
        width_m=0.4,
        height_m=0.2,
        camera_to_centre_m=(0.0, 0.1, 0.0),
        provenance="invented for tests",
    )


def centred_eye(screen: ScreenConfig, distance_m: float = 0.5) -> np.ndarray:
    """Camera-frame eye position that sits straight in front of the centre.

    Args:
        screen: The display to centre on.
        distance_m: How far in front of the panel to place the eye.

    Returns:
        The eye position in the camera frame, metres.
    """
    offset = np.asarray(screen.camera_to_centre_m, dtype=float)
    return np.array([offset[0], offset[1], distance_m])


def screen_corners(screen: ScreenConfig) -> np.ndarray:
    """Build the four panel corners in the screen frame.

    Args:
        screen: The display.

    Returns:
        Corners ``(4, 3)`` as top left, top right, bottom right, bottom left.
    """
    half_w, half_h = screen.width_m / 2, screen.height_m / 2
    return np.array(
        [
            [-half_w, half_h, 0.0],
            [half_w, half_h, 0.0],
            [half_w, -half_h, 0.0],
            [-half_w, -half_h, 0.0],
        ]
    )


def test_the_conversion_flips_x_and_y_but_not_z(screen: ScreenConfig) -> None:
    """Two frames that face each other differ by a rotation about Z."""
    # One metre right and half a metre up in the camera frame, where +Y is down.
    eye_camera = np.array([1.0, screen.camera_to_centre_m[1] - 0.5, 0.6])
    eye = eye_in_screen_frame(eye_camera, screen)
    assert eye[0] == pytest.approx(-1.0)
    assert eye[1] == pytest.approx(0.5)
    assert eye[2] == pytest.approx(0.6)


def test_a_camera_above_the_panel_sees_a_centred_eye_as_centred(
    screen: ScreenConfig,
) -> None:
    """The Y offset cancels, which is the sign the docstring warns about."""
    eye = eye_in_screen_frame(centred_eye(screen), screen)
    assert eye[0] == pytest.approx(0.0)
    assert eye[1] == pytest.approx(0.0)
    assert eye[2] == pytest.approx(0.5)


def test_a_centred_eye_gives_a_symmetric_frustum(screen: ScreenConfig) -> None:
    """On axis, the frustum reduces to the ordinary symmetric case."""
    frustum = frustum_for_eye(screen, centred_eye(screen))
    assert frustum.is_symmetric
    assert frustum.left == pytest.approx(-frustum.right)
    assert frustum.bottom == pytest.approx(-frustum.top)


def test_a_centred_frustum_matches_the_screen_aspect(screen: ScreenConfig) -> None:
    """The window's shape is the panel's shape, not the image's."""
    frustum = frustum_for_eye(screen, centred_eye(screen))
    aspect = (frustum.right - frustum.left) / (frustum.top - frustum.bottom)
    assert aspect == pytest.approx(screen.width_m / screen.height_m)


def test_moving_the_eye_right_shifts_the_frustum_left(screen: ScreenConfig) -> None:
    """Moving right reveals what is to the left, as through a real window.

    The viewer's right is camera ``-X``, so this moves the eye there, and the
    frustum extents must both decrease.
    """
    base = frustum_for_eye(screen, centred_eye(screen))
    moved = frustum_for_eye(screen, centred_eye(screen) + np.array([-0.1, 0, 0]))
    assert moved.left < base.left
    assert moved.right < base.right
    # The width is unchanged: the eye moved, it did not get closer.
    assert moved.right - moved.left == pytest.approx(base.right - base.left)


def test_moving_the_eye_up_shifts_the_frustum_down(screen: ScreenConfig) -> None:
    """The Y counterpart, moving up in the world, which is camera -Y."""
    base = frustum_for_eye(screen, centred_eye(screen))
    moved = frustum_for_eye(screen, centred_eye(screen) + np.array([0, -0.1, 0]))
    assert moved.bottom < base.bottom
    assert moved.top < base.top


def test_moving_closer_widens_the_view(screen: ScreenConfig) -> None:
    """Nearer the window, more of the world is visible through it."""
    near_eye = frustum_for_eye(screen, centred_eye(screen, 0.3))
    far_eye = frustum_for_eye(screen, centred_eye(screen, 0.9))
    assert near_eye.right > far_eye.right


@pytest.mark.parametrize("dx", [-0.25, -0.05, 0.0, 0.05, 0.25])
@pytest.mark.parametrize("dy", [-0.2, 0.0, 0.2])
@pytest.mark.parametrize("distance", [0.25, 0.5, 1.2])
def test_screen_corners_map_to_viewport_corners(
    screen: ScreenConfig,
    dx: float,
    dy: float,
    distance: float,
) -> None:
    """The defining property: the panel always fills the image exactly.

    Whatever the eye does, the four physical corners of the panel must land on
    the four corners of the rendered image. That is what makes the screen a
    window rather than a picture.

    Note what this does **not** cover, established by mutation rather than
    assumed: a wrong camera to screen conversion builds a frustum that is wrong
    to match, so the corners still land perfectly. The directional tests cover
    that. This one covers the near-plane scaling, the matrix and the viewport.
    """
    eye_camera = centred_eye(screen, distance) + np.array([dx, dy, 0.0])
    matrix = view_projection_matrix(screen, eye_camera)
    pixels = project_points(matrix, screen_corners(screen), WIDTH_PX, HEIGHT_PX)

    expected = np.array(
        [
            [0.0, 0.0],
            [WIDTH_PX, 0.0],
            [WIDTH_PX, HEIGHT_PX],
            [0.0, HEIGHT_PX],
        ]
    )
    assert pixels == pytest.approx(expected, abs=1e-6)


def test_the_screen_centre_projects_to_the_image_centre(screen: ScreenConfig) -> None:
    """Ordinary case, stated separately so a failure is easy to read."""
    matrix = view_projection_matrix(screen, centred_eye(screen))
    pixels = project_points(matrix, np.zeros((1, 3)), WIDTH_PX, HEIGHT_PX)
    assert pixels[0] == pytest.approx([WIDTH_PX / 2, HEIGHT_PX / 2])


def test_a_point_behind_the_screen_stays_put_when_the_eye_moves(
    screen: ScreenConfig,
) -> None:
    """Parallax has a direction, and this is it.

    A point deep behind the window drifts across the image as the eye moves,
    the same way a distant tree does. Moving to the viewer's right, camera
    ``-X``, must send it right in the image.
    """
    far_point = np.array([[0.0, 0.0, -2.0]])
    base = project_points(
        view_projection_matrix(screen, centred_eye(screen)),
        far_point,
        WIDTH_PX,
        HEIGHT_PX,
    )
    moved = project_points(
        view_projection_matrix(screen, centred_eye(screen) + np.array([-0.1, 0, 0])),
        far_point,
        WIDTH_PX,
        HEIGHT_PX,
    )
    assert moved[0][0] > base[0][0]


def test_the_projection_matrix_is_the_gl_frustum_form(screen: ScreenConfig) -> None:
    """Pin the convention, so a later rewrite cannot quietly change it."""
    frustum = frustum_for_eye(screen, centred_eye(screen))
    matrix = projection_matrix(frustum)
    assert matrix[3, 2] == pytest.approx(-1.0)
    assert matrix[3, 3] == pytest.approx(0.0)
    width = frustum.right - frustum.left
    assert matrix[0, 0] == pytest.approx(2 * frustum.near / width)
    assert matrix[2, 2] == pytest.approx(
        -(frustum.far + frustum.near) / (frustum.far - frustum.near)
    )


def test_the_eye_must_be_in_front_of_the_screen(screen: ScreenConfig) -> None:
    """At the panel there is no frustum, so say so rather than divide by zero."""
    for depth in (0.0, -0.3):
        eye = centred_eye(screen)
        eye[2] = depth
        with pytest.raises(EyeBehindScreenError):
            frustum_for_eye(screen, eye)


@pytest.mark.parametrize(
    ("near", "far"),
    [(0.0, 10.0), (-1.0, 10.0), (5.0, 5.0), (9.0, 1.0)],
)
def test_the_clip_planes_must_bound_something(
    screen: ScreenConfig,
    near: float,
    far: float,
) -> None:
    """0 < near < far, or there is nothing to project into."""
    with pytest.raises(InvalidClipPlanesError):
        frustum_for_eye(screen, centred_eye(screen), near_m=near, far_m=far)


def test_points_behind_the_eye_are_refused(screen: ScreenConfig) -> None:
    """The perspective divide is undefined there, so it must not be guessed."""
    matrix = view_projection_matrix(screen, centred_eye(screen))
    behind = np.array([[0.0, 0.0, 5.0]])
    with pytest.raises(PointNotInFrontError):
        project_points(matrix, behind, WIDTH_PX, HEIGHT_PX)


def test_the_near_plane_is_not_the_screen_plane(screen: ScreenConfig) -> None:
    """Changing the near plane rescales the extents and nothing else."""
    a = frustum_for_eye(screen, centred_eye(screen), near_m=0.05)
    b = frustum_for_eye(screen, centred_eye(screen), near_m=0.10)
    assert b.right == pytest.approx(a.right * 2)
    # The view itself is unchanged, so the same corners still fill the image.
    corners = screen_corners(screen)
    for near in (0.05, 0.10):
        matrix = view_projection_matrix(screen, centred_eye(screen), near_m=near)
        pixels = project_points(matrix, corners, WIDTH_PX, HEIGHT_PX)
        assert pixels[0] == pytest.approx([0.0, 0.0], abs=1e-6)


def test_the_real_g7_screen_gives_a_plausible_view() -> None:
    """Sanity check against the measured device, loosely.

    A 344 mm panel at 0.5 m subtends about 38 degrees horizontally.
    """
    screen = get_screen("g7_internal")
    frustum = frustum_for_eye(screen, centred_eye(screen, 0.5))
    half_angle = np.degrees(np.arctan(frustum.right / frustum.near))
    assert 2 * half_angle == pytest.approx(38.0, abs=1.0)
    assert frustum.near == DEFAULT_NEAR_M
    assert frustum.far == DEFAULT_FAR_M
