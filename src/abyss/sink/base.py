"""The sink seam itself, and the check every implementation shares.

The second seam of phase 4, and the one Q13 argued for: a sink acts on a frame
that already exists, so it knows nothing about screens, eyes or projections.

``size`` lives on the protocol rather than being passed alongside it because
the sink is what knows how big a frame it accepts (Q20). `PngSink` reads it
from its config; `WindowSink` reads it from the window it opened.
"""

from typing import Protocol
from typing import runtime_checkable

import numpy as np


class FrameSizeMismatchError(ValueError):
    """Raised when a frame handed to a sink is not the size it expects."""

    def __init__(self, expected: tuple[int, int], actual: tuple[int, int]) -> None:
        """Initialise with both sizes.

        Args:
            expected: The ``(width, height)`` the sink was configured for.
            actual: The ``(width, height)`` of the offending frame.
        """
        super().__init__(
            f"Sink expects {expected[0]}x{expected[1]} frames, got "
            f"{actual[0]}x{actual[1]}"
        )


@runtime_checkable
class Sink(Protocol):
    """Somewhere finished frames go.

    Implementations are not expected to be reusable after :meth:`close`.
    """

    @property
    def size(self) -> tuple[int, int]:
        """Frame size this sink accepts, as ``(width_px, height_px)``."""
        ...

    def write(self, frame: np.ndarray) -> None:
        """Accept one finished frame.

        Args:
            frame: A BGR image of exactly :attr:`size`.
        """
        ...

    def close(self) -> None:
        """Release whatever the sink holds. Safe to call more than once."""
        ...


def check_size(expected: tuple[int, int], frame: np.ndarray) -> None:
    """Reject a frame that is not the size the sink was built for.

    Args:
        expected: The ``(width, height)`` the sink accepts.
        frame: The frame handed in.

    Raises:
        FrameSizeMismatchError: If the frame is a different size.
    """
    height, width = frame.shape[:2]
    if (width, height) != expected:
        raise FrameSizeMismatchError(expected, (width, height))
