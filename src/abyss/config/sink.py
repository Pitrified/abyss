"""What happens to a frame once it has been rendered.

Split from the screen because the two belong to different stages (Q13): screen
geometry is an *input* to the rendering, since the frustum is built from it,
while a sink only acts on a frame that already exists. Handing the renderer one
object that also carried PNG paths would give it information it has no business
seeing.

The resolution lives here because pixels are what a sink writes, while metres
are what a screen is (Q20). The `Sink` protocol exposes it as ``size``, so a
sink that did not choose its own resolution - a window, in phase 5 - can report
the one it was given instead.
"""

from pathlib import Path

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class SinkConfig(BaseModel):
    """Where finished frames go, and how big they are.

    Args:
        name: Name of this run, used in logs, filenames and as a registry key.
        out_fol: Folder to write into. Created by the sink, not by this model.
        width_px: Output frame width in pixels.
        height_px: Output frame height in pixels.
        fps: Frame rate for sinks that have one. Ignored by still-image sinks.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    out_fol: Path
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    fps: float = Field(default=25.0, gt=0)

    @property
    def size(self) -> tuple[int, int]:
        """Output size as ``(width_px, height_px)``."""
        return self.width_px, self.height_px

    @property
    def aspect(self) -> float:
        """Output aspect ratio, width over height."""
        return self.width_px / self.height_px
