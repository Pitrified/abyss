"""Test the sink config."""

from pathlib import Path

from pydantic import ValidationError
import pytest

from abyss.config.sink import SinkConfig


def make(width_px: int = 1280, height_px: int = 720, fps: float = 25.0) -> SinkConfig:
    """Build a sink config, defaulting to the size the sweep renders at."""
    return SinkConfig(
        name="test",
        out_fol=Path("out"),
        width_px=width_px,
        height_px=height_px,
        fps=fps,
    )


def test_the_size_is_width_then_height() -> None:
    """The order matters: OpenCV takes (width, height) and numpy the reverse."""
    assert make().size == (1280, 720)


def test_the_aspect_is_width_over_height() -> None:
    """Width over height, the same way a screen aspect is written."""
    assert make().aspect == pytest.approx(16 / 9)


@pytest.mark.parametrize("value", [0, -1])
def test_a_degenerate_width_is_rejected(value: int) -> None:
    """As the other three config models reject theirs."""
    with pytest.raises(ValidationError):
        make(width_px=value)


@pytest.mark.parametrize("value", [0, -1])
def test_a_degenerate_height_is_rejected(value: int) -> None:
    """As the other three config models reject theirs."""
    with pytest.raises(ValidationError):
        make(height_px=value)


def test_the_frame_rate_must_be_positive() -> None:
    """A zero rate would make a video of no duration."""
    with pytest.raises(ValidationError):
        make(fps=0)


def test_the_config_is_frozen() -> None:
    """Config is passed around, so it must not be edited in flight."""
    with pytest.raises(ValidationError):
        make().width_px = 640
