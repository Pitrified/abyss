"""Test the wireframe renderer.

Split the way phase 3's lesson demands. The corner invariant is deliberately
**not** re-tested here: phase 3 already sweeps 45 cases of it, and it is the
family the mutation pass proved blind to a wrong frame conversion. What is
tested here is what only appears once something is drawn - that the frame is
not empty, and that things at different depths move by different amounts when
the eye moves.

Measurements are taken off the rendered pixels rather than off the projected
coordinates, since coordinates would only re-test phase 3. The scene's colours
are what makes that possible: the cube is orange and the frame marker cyan, so
each can be found in the image without knowing where it was drawn.
"""

import numpy as np
import pytest

from abyss.config.screen import ScreenConfig
from abyss.params.abyss_devices import get_screen
from abyss.render.renderer import BACKGROUND_BGR
from abyss.render.renderer import AspectMismatchError
from abyss.render.renderer import Renderer
from abyss.render.renderer import WireframeRenderer
from abyss.render.renderer import check_aspect
from abyss.render.renderer import render_frame
from abyss.render.scene import window_box

WIDTH_PX, HEIGHT_PX = 640, 320
HUE_MARGIN = 40
"""How far a channel must lead to count as that colour, after fading."""


@pytest.fixture
def screen() -> ScreenConfig:
    """Build a 2:1 screen, matching the test render size exactly."""
    return ScreenConfig(
        name="test",
        width_m=0.4,
        height_m=0.2,
        camera_to_centre_m=(0.0, 0.1, 0.0),
        provenance="invented for tests",
    )


@pytest.fixture
def renderer(screen: ScreenConfig) -> WireframeRenderer:
    """Build the renderer under test, over the default scene."""
    return WireframeRenderer(window_box(screen))


def centred_eye(screen: ScreenConfig, right_m: float = 0.0) -> np.ndarray:
    """Camera-frame eye, optionally moved to the viewer's right.

    Args:
        screen: The display to sit in front of.
        right_m: How far to the viewer's right, which is camera ``-X``.

    Returns:
        The eye position in the camera frame.
    """
    offset = np.asarray(screen.camera_to_centre_m, dtype=float)
    return offset + np.array([-right_m, 0.0, 0.6])


def centroid(frame: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Centre of mass of the masked pixels.

    Args:
        frame: The rendered image.
        mask: Boolean mask over it.

    Returns:
        The ``(u, v)`` centroid in pixels.
    """
    rows, cols = np.nonzero(mask)
    return float(cols.mean()), float(rows.mean())


def orange_mask(frame: np.ndarray) -> np.ndarray:
    """Find the cube, which is the only orange thing in the scene."""
    return frame[:, :, 2].astype(int) - frame[:, :, 0].astype(int) > HUE_MARGIN


def cyan_mask(frame: np.ndarray) -> np.ndarray:
    """Find the frame marker, which is the only cyan thing in the scene."""
    return frame[:, :, 0].astype(int) - frame[:, :, 2].astype(int) > HUE_MARGIN


def test_the_wireframe_renderer_satisfies_the_protocol(
    renderer: WireframeRenderer,
) -> None:
    """The seam, satisfied by the implementation that shaped it."""
    assert isinstance(renderer, Renderer)


def test_the_frame_is_the_size_asked_for(
    screen: ScreenConfig,
    renderer: WireframeRenderer,
) -> None:
    """The sink will refuse anything else, so get it right here."""
    frame = render_frame(screen, centred_eye(screen), renderer, (WIDTH_PX, HEIGHT_PX))
    assert frame.shape == (HEIGHT_PX, WIDTH_PX, 3)
    assert frame.dtype == np.uint8


def test_the_frame_is_not_blank(
    screen: ScreenConfig,
    renderer: WireframeRenderer,
) -> None:
    """The guard against the silent failure that leaves a suite green.

    If everything projected off-screen, or the scene were empty, every other
    test here could still pass while the artefact was a black rectangle.
    """
    frame = render_frame(screen, centred_eye(screen), renderer, (WIDTH_PX, HEIGHT_PX))
    drawn = np.any(frame != np.array(BACKGROUND_BGR, dtype=np.uint8), axis=2)
    assert drawn.mean() > 0.005


def test_moving_the_eye_right_moves_the_cube_right(
    screen: ScreenConfig,
    renderer: WireframeRenderer,
) -> None:
    """Parallax has a direction, measured off the pixels.

    A cube behind the window drifts across the image as the eye moves, the way
    a distant tree does when you walk past a window.
    """
    size = (WIDTH_PX, HEIGHT_PX)
    base = render_frame(screen, centred_eye(screen), renderer, size)
    moved = render_frame(screen, centred_eye(screen, right_m=0.12), renderer, size)

    base_u, _ = centroid(base, orange_mask(base))
    moved_u, _ = centroid(moved, orange_mask(moved))
    assert moved_u > base_u + 1


def test_the_cube_and_the_window_move_by_different_amounts(
    screen: ScreenConfig,
    renderer: WireframeRenderer,
) -> None:
    """Equal motion would mean depth never reached the projection.

    The frame marker sits in the panel plane, so it is welded to the image
    whatever the eye does. The cube is 0.25 m behind it and must not be. If
    both moved together, the render would be a picture panning rather than a
    window being looked through.
    """
    size = (WIDTH_PX, HEIGHT_PX)
    base = render_frame(screen, centred_eye(screen), renderer, size)
    moved = render_frame(screen, centred_eye(screen, right_m=0.12), renderer, size)

    cube_shift = centroid(moved, orange_mask(moved))[0] - centroid(
        base, orange_mask(base)
    )[0]
    marker_shift = centroid(moved, cyan_mask(moved))[0] - centroid(
        base, cyan_mask(base)
    )[0]
    assert abs(marker_shift) < 0.5
    assert abs(cube_shift) > 5 * max(abs(marker_shift), 0.5)


def test_backing_away_makes_the_cube_fill_more_of_the_window(
    screen: ScreenConfig,
    renderer: WireframeRenderer,
) -> None:
    """Depth is in the projection, not just the offset.

    Backing away shrinks the cube and the window together, but not at the same
    rate: the cube is 0.25 m further off, so the window shrinks faster and the
    cube ends up covering more of it. In the image the cube therefore grows as
    the eye retreats, which is the opposite of what a picture would do and is
    the whole reason the frustum is rebuilt per frame.
    """
    size = (WIDTH_PX, HEIGHT_PX)
    offset = np.asarray(screen.camera_to_centre_m, dtype=float)

    def cube_width(distance_m: float) -> float:
        eye = offset + np.array([0, 0, distance_m])
        frame = render_frame(screen, eye, renderer, size)
        cols = np.nonzero(orange_mask(frame))[1]
        return float(cols.max() - cols.min())

    assert cube_width(1.2) > cube_width(0.3) * 1.2


def test_a_non_grey_background_is_rendered_in_that_colour(
    screen: ScreenConfig,
) -> None:
    """Nothing varies this parameter today, which is why it is worth pinning.

    The cheapest possible fill is `np.full(shape, background[0])`, which is
    four times faster than the per-channel one and exactly equivalent while the
    background stays grey. This is the test that fails when someone takes it,
    rather than a viewer noticing the room came out blue.
    """
    background = (10, 60, 200)
    renderer = WireframeRenderer(window_box(screen), background=background)
    frame = render_frame(screen, centred_eye(screen), renderer, (WIDTH_PX, HEIGHT_PX))

    # Not sampled at the corners: the box's corner connectors run right into
    # them. The scene is forty lines on an otherwise empty frame, so the fill is
    # almost all of it, and the scalar shortcut would leave none of it matching.
    filled = np.all(frame == np.array(background, dtype=np.uint8), axis=2)
    assert filled.mean() > 0.9


def test_each_render_returns_its_own_frame(
    screen: ScreenConfig,
    renderer: WireframeRenderer,
) -> None:
    """Filling one reused buffer would be free, and would corrupt the caller.

    `render` returns the frame and callers keep it: phase 4's `render_run`
    holds selected frames for a contact sheet while writing the same frame to
    two sinks. A shared buffer would rewrite what had already been handed over,
    so this guards against it being reintroduced as an optimisation.
    """
    size = (WIDTH_PX, HEIGHT_PX)
    first = render_frame(screen, centred_eye(screen), renderer, size)
    kept = first.copy()
    render_frame(screen, centred_eye(screen, right_m=0.12), renderer, size)

    assert np.array_equal(first, kept)


def test_the_real_g7_panel_passes_the_aspect_check() -> None:
    """0.26% off 16:9, which is why the check cannot be an equality."""
    check_aspect(get_screen("g7_internal"), 1280, 720)


def test_a_mismatched_aspect_is_refused(screen: ScreenConfig) -> None:
    """It would fill the frame and stretch rather than letterbox."""
    with pytest.raises(AspectMismatchError):
        check_aspect(screen, 640, 480)


def test_the_aspect_check_runs_before_anything_is_drawn(
    screen: ScreenConfig,
    renderer: WireframeRenderer,
) -> None:
    """The entry point is where the screen and the output size meet."""
    with pytest.raises(AspectMismatchError):
        render_frame(screen, centred_eye(screen), renderer, (640, 480))
