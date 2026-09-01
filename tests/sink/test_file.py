"""Test the sinks.

The point of testing both implementations together is the Protocol itself: a
Protocol with one implementer is unvalidated, and these two disagree enough to
be a real check of the shape. One writes many files and counts them, the other
holds a handle that must be released.
"""

from pathlib import Path

import numpy as np
import pytest

from abyss.config.sink import SinkConfig
from abyss.sink.base import FrameSizeMismatchError
from abyss.sink.base import Sink
from abyss.sink.file import PngSink
from abyss.sink.file import VideoSink

WIDTH_PX, HEIGHT_PX = 64, 36


@pytest.fixture
def config(tmp_path: Path) -> SinkConfig:
    """Build a sink config writing into a temporary folder."""
    return SinkConfig(
        name="run",
        out_fol=tmp_path / "out",
        width_px=WIDTH_PX,
        height_px=HEIGHT_PX,
    )


def frame() -> np.ndarray:
    """Build one frame of the right size."""
    return np.zeros((HEIGHT_PX, WIDTH_PX, 3), dtype=np.uint8)


def test_both_sinks_satisfy_the_protocol(config: SinkConfig) -> None:
    """The reason two implementations ship together."""
    for sink in (PngSink(config), VideoSink(config)):
        assert isinstance(sink, Sink)
        assert sink.size == (WIDTH_PX, HEIGHT_PX)
        sink.close()


def test_the_png_sink_numbers_frames_so_a_glob_sorts(config: SinkConfig) -> None:
    """Zero padding, or frame 10 sorts before frame 2."""
    sink = PngSink(config)
    for _ in range(11):
        sink.write(frame())
    sink.close()

    written = sorted(p.name for p in config.out_fol.glob("*.png"))
    assert len(written) == 11
    assert written[0] == "frame_00000.png"
    assert written[-1] == "frame_00010.png"


def test_the_video_sink_writes_one_file(config: SinkConfig) -> None:
    """One handle, one file, unlike the PNG sink's many."""
    sink = VideoSink(config)
    for _ in range(5):
        sink.write(frame())
    sink.close()

    assert (config.out_fol / "run.mp4").exists()


def test_closing_twice_is_safe(config: SinkConfig) -> None:
    """A run that fails halfway will close in a finally and then again."""
    sink = VideoSink(config)
    sink.write(frame())
    sink.close()
    sink.close()


@pytest.mark.parametrize("sink_type", [PngSink, VideoSink])
def test_a_frame_of_the_wrong_size_is_refused(config: SinkConfig, sink_type) -> None:
    """OpenCV's video writer drops mismatched frames silently, so check first."""
    sink = sink_type(config)
    wrong = np.zeros((HEIGHT_PX + 1, WIDTH_PX, 3), dtype=np.uint8)
    with pytest.raises(FrameSizeMismatchError):
        sink.write(wrong)
    sink.close()


def test_the_sink_creates_its_output_folder(config: SinkConfig) -> None:
    """The config names a folder; making it is the sink's job."""
    assert not config.out_fol.exists()
    PngSink(config).close()
    assert config.out_fol.is_dir()
