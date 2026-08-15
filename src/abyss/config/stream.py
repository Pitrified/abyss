"""Where the frames come from, for one run.

Split from the camera because one camera feeds both a recorded clip and a live
capture (Q11), and the intrinsics are the same either way. What changes per run
is only the source.

Frame rate and frame size are deliberately absent: both are read from the
capture, so a config value could only disagree with the frames in hand.
"""

from pathlib import Path

from pydantic import BaseModel
from pydantic import ConfigDict


class StreamConfig(BaseModel):
    """One source of frames.

    Args:
        name: Name of this stream, used in logs and as its registry key.
        camera: Registry key of the camera the frames come from.
        source: A path to a clip, or an integer device index for a live
            capture. Both are what OpenCV's ``VideoCapture`` accepts.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    camera: str
    source: int | Path

    @property
    def is_live(self) -> bool:
        """Whether this stream is a camera device rather than a file."""
        return isinstance(self.source, int)
