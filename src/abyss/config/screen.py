"""Where the screen is, relative to the camera.

The screen is the window the scene is seen through, so phase 3's off-axis
frustum is built from this: a rectangle of known size at a known place in the
camera's frame - ``+X`` right, ``+Y`` down, ``+Z`` away, metres, the convention
phase 1 established.

Rotation is deliberately not modelled. For a laptop the camera sits in the lid
bezel, so it does not move relative to the panel when the lid tilts and one
offset holds for every lid angle. A tilted external monitor would need more,
and can have it when one exists.

Panel sizes come from the display's own EDID rather than a tape measure - run
``scripts/read_edid.py`` on the machine and paste the result here. The camera
offset has no such source and has to be measured by hand.
"""

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class ScreenConfig(BaseModel):
    """A display device, as a rectangle placed in the camera's frame.

    Args:
        name: Device name, used in logs and as its registry key.
        width_m: Active panel width in metres.
        height_m: Active panel height in metres.
        camera_to_centre_m: Offset from the camera's optical centre to the
            centre of the panel, in metres, in the camera frame. For a laptop
            the camera sits above the screen, so with ``+Y`` pointing down the
            image this is a positive Y.
        provenance: Where the numbers came from, in words.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    width_m: float = Field(gt=0)
    height_m: float = Field(gt=0)
    camera_to_centre_m: tuple[float, float, float]
    provenance: str
