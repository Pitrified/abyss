"""Test the scene geometry."""

import numpy as np
import pytest

from abyss.config.screen import ScreenConfig
from abyss.render.scene import DEFAULT_DEPTH_M
from abyss.render.scene import FRAME_MARKER_SCALE
from abyss.render.scene import MalformedSceneError
from abyss.render.scene import Scene
from abyss.render.scene import SceneInFrontOfWindowError
from abyss.render.scene import window_box


@pytest.fixture
def screen() -> ScreenConfig:
    """Build a screen with round numbers, so the arithmetic checks by eye."""
    return ScreenConfig(
        name="test",
        width_m=0.4,
        height_m=0.2,
        camera_to_centre_m=(0.0, 0.1, 0.0),
        provenance="invented for tests",
    )


def test_the_scene_stays_behind_the_window(screen: ScreenConfig) -> None:
    """The property that makes a clipper unnecessary.

    Not a style point: with the eye in front and the scene behind, no point can
    be at or behind the eye, so ``project_points`` cannot raise.
    """
    scene = window_box(screen)
    assert scene.segments[..., 2].max() <= 0


def test_the_scene_reaches_back_exactly_as_far_as_asked(screen: ScreenConfig) -> None:
    """The depth argument is the back wall, not a suggestion."""
    scene = window_box(screen, depth_m=0.5)
    assert scene.segments[..., 2].min() == pytest.approx(-0.5)


def test_a_scene_in_front_of_the_window_is_refused() -> None:
    """Refused at construction, where it can still be explained."""
    segments = np.array([[[0.0, 0.0, 0.1], [0.0, 0.0, -0.1]]])
    with pytest.raises(SceneInFrontOfWindowError):
        Scene(segments=segments, colours=np.zeros((1, 3)))


def test_segments_and_colours_must_line_up() -> None:
    """One colour per segment, or the draw loop would silently truncate."""
    segments = np.zeros((3, 2, 3))
    with pytest.raises(MalformedSceneError):
        Scene(segments=segments, colours=np.zeros((2, 3)))


def test_segments_must_be_pairs_of_3d_points() -> None:
    """The one shape the renderer knows how to project."""
    with pytest.raises(MalformedSceneError):
        Scene(segments=np.zeros((3, 2)), colours=np.zeros((3, 3)))


def test_the_frame_marker_sits_inside_the_panel(screen: ScreenConfig) -> None:
    """It is the mouth pulled in, and the gap is what makes it visible."""
    scene = window_box(screen)
    at_panel = scene.segments[scene.segments[..., 2] == 0]
    marker = at_panel[np.abs(at_panel[:, 0]) < screen.width_m / 2]
    assert len(marker)
    assert np.abs(marker[:, 0]).max() == pytest.approx(
        screen.width_m / 2 * FRAME_MARKER_SCALE
    )


def test_the_mouth_corners_are_present_but_its_edges_are_not(
    screen: ScreenConfig,
) -> None:
    """The mouth edges clip away exactly on the image boundary.

    The corner connectors start there and run backwards, so they draw. A
    segment lying *in* the panel plane at full size would not, which is why the
    scene must not contain one.
    """
    scene = window_box(screen)
    half_w = screen.width_m / 2
    on_edge = np.isclose(np.abs(scene.segments[..., 0]), half_w) & np.isclose(
        scene.segments[..., 2], 0.0
    )
    # Corners at the mouth exist,
    assert on_edge.any()
    # but no segment has both its endpoints in the panel plane at full width.
    both_ends = on_edge.all(axis=1)
    assert not both_ends.any()


def test_colour_fades_with_depth(screen: ScreenConfig) -> None:
    """The depth cue, which is what replaced painter ordering.

    Both groups here are the *same base colour*, the room, at two depths. An
    earlier version of this test compared whatever was near against whatever
    was far, which is a proxy: the near things were the cyan marker and the far
    things the grey grid, so it passed on base colour alone and stayed green
    with the fade removed entirely. Established by mutation, not by reasoning.
    """
    scene = window_box(screen, depth_m=DEFAULT_DEPTH_M)
    x, y, z = (scene.segments[..., i] for i in range(3))
    on_frame = np.isclose(np.abs(x), screen.width_m / 2).all(axis=1) & np.isclose(
        np.abs(y), screen.height_m / 2
    ).all(axis=1)

    # The back wall's own edges, flat against the far end.
    back_edge = on_frame & np.isclose(z, -DEFAULT_DEPTH_M).all(axis=1)
    # The corners and wall lines, which run from the mouth to the back wall.
    spanning = np.isclose(z.min(axis=1), -DEFAULT_DEPTH_M) & np.isclose(
        z.max(axis=1), 0.0
    )
    assert back_edge.any()
    assert spanning.any()
    assert scene.colours[back_edge].mean() < scene.colours[spanning].mean() * 0.8


def test_the_scene_is_small_enough_to_read(screen: ScreenConfig) -> None:
    """A minimal scene, and a number that says what minimal means."""
    assert len(window_box(screen)) < 50
