"""The person in front of the camera.

A viewer is not a device, which is why this is its own model rather than a
field on the camera. Q6 chose interpupillary distance as the scale reference:
MediaPipe fits an identity-dependent mesh, so the head size implicit in its
metric output varies per person - 66.9 mm against 57.7 mm of implied
interpupillary distance between the two subjects in the sample clips, 16%
apart. Comparing that implied size against the viewer's real one is what puts a
track on the right scale.

Choosing *which* viewer is in frame is deferred: there is one person today, and
the per-session estimator derives their scale from the frames themselves.
"""

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

DEFAULT_IPD_M = 0.063
"""Mean adult interpupillary distance, the scale reference chosen in Q6."""


class ViewerConfig(BaseModel):
    """A viewer, described by what the scale correction needs.

    Args:
        name: Who this is, used in logs and as the registry key.
        ipd_m: Interpupillary distance in metres.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    ipd_m: float = Field(default=DEFAULT_IPD_M, gt=0)
