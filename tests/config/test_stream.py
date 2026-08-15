"""Test the stream config."""

from pathlib import Path

import pytest

from abyss.config.stream import StreamConfig


def test_a_clip_is_not_live() -> None:
    """A recorded clip is a path, and paths do not stream."""
    stream = StreamConfig(name="clip", camera="test", source=Path("face01.mp4"))
    assert stream.is_live is False


def test_a_device_index_is_live() -> None:
    """An integer is what OpenCV takes for a capture device."""
    stream = StreamConfig(name="webcam", camera="test", source=0)
    assert stream.is_live is True


def test_a_missing_clip_is_still_a_valid_config() -> None:
    """The clips live outside the repo and are not on every machine."""
    stream = StreamConfig(name="clip", camera="test", source=Path("/nope.mp4"))
    assert stream.source == Path("/nope.mp4")


def test_the_config_is_frozen() -> None:
    """Config is read at run time, never written."""
    stream = StreamConfig(name="webcam", camera="test", source=0)
    with pytest.raises(ValueError, match="frozen"):
        stream.source = 1
