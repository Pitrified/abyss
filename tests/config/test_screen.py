"""Test the screen config."""

import pytest

from abyss.config.screen import ScreenConfig


def make(width_m: float = 0.309, height_m: float = 0.173) -> ScreenConfig:
    """Build a screen config, defaulting to this box's panel."""
    return ScreenConfig(
        name="test",
        width_m=width_m,
        height_m=height_m,
        camera_to_centre_m=(0.0, 0.0965, 0.0),
        provenance="test",
    )


def test_it_holds_metres() -> None:
    """Sizes are metres, the unit phase 1 fixed for the whole package."""
    screen = make()
    assert screen.width_m == 0.309
    assert screen.height_m == 0.173


def test_the_camera_sits_above_the_panel_centre() -> None:
    """+Y points down the image, so a camera above the panel gives +Y."""
    assert make().camera_to_centre_m[1] > 0


@pytest.mark.parametrize("value", [0.0, -0.3])
def test_a_degenerate_width_is_rejected(value: float) -> None:
    """A screen with no area cannot be a window onto anything."""
    with pytest.raises(ValueError, match="width_m"):
        make(width_m=value)


@pytest.mark.parametrize("value", [0.0, -0.3])
def test_a_degenerate_height_is_rejected(value: float) -> None:
    """A screen with no area cannot be a window onto anything."""
    with pytest.raises(ValueError, match="height_m"):
        make(height_m=value)


def test_the_config_is_frozen() -> None:
    """Config is read at run time, never written."""
    screen = make()
    with pytest.raises(ValueError, match="frozen"):
        screen.width_m = 0.4
