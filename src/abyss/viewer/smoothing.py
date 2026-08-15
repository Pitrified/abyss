"""Causal smoothing for a 3D position track.

Jitter in an eye position becomes jitter in a projection matrix, which is
visible immediately, so the track is filtered before anything downstream uses
it.

Built on ``pose_tools.utils.np_signal``: a left-triangle filter whose weights
rise towards the newest sample and sum to 1. It is causal, so an offline run
and a live loop produce the same numbers - nothing here peeks at future frames.

Deliberately not ``pose_tools.geometry.signal_tracker.SignalTracker``, despite
the name: that is a gesture classifier whose ``update()`` returns a thresholded
derivative, and it would need three threshold parameters that mean nothing for a
position. It is a consumer of the same primitives used here.
"""

import numpy as np
from pose_tools.utils.np_signal import create_left_triangle_filter
from pose_tools.utils.np_signal import roll_append_smooth

DEFAULT_FILTER_SIZE = 5
"""Filter width in frames: 0.2 s at 25 fps, wide enough to see on the plots."""

AXES = 3


class PositionSmoother:
    """Smooth a 3D position track, one causal filter per axis.

    The history starts from the first sample rather than from zeros, so the
    output does not ramp up from the origin over the first few frames.

    Args:
        filter_size: Number of taps in the smoothing filter.
    """

    def __init__(self, filter_size: int = DEFAULT_FILTER_SIZE) -> None:
        """Initialize the smoother.

        Args:
            filter_size: Number of taps in the smoothing filter.
        """
        self.filter_size = filter_size
        self.filt = create_left_triangle_filter(filter_size)
        self.history = np.zeros((AXES, filter_size), dtype=float)
        self.history_smooth = np.zeros((AXES, filter_size), dtype=float)
        self.started = False

    def update(self, position: np.ndarray) -> np.ndarray:
        """Feed one position and return the smoothed one.

        Args:
            position: Array ``[x, y, z]``.

        Returns:
            The smoothed position, same shape.
        """
        if not self.started:
            # Prime both histories with the first sample, so the filter starts
            # settled instead of climbing out of zero.
            self.history[:] = position[:, None]
            self.history_smooth[:] = position[:, None]
            self.started = True

        smoothed = np.empty(AXES, dtype=float)
        for axis in range(AXES):
            self.history[axis], self.history_smooth[axis] = roll_append_smooth(
                self.history[axis],
                self.history_smooth[axis],
                float(position[axis]),
                self.filt,
            )
            smoothed[axis] = self.history_smooth[axis][-1]
        return smoothed

    def hold(self) -> np.ndarray | None:
        """Keep the state unchanged across a frame with no face.

        The filter assumes evenly spaced samples, and a gap is not a new
        measurement: holding keeps the last estimate rather than inventing one
        or resetting to zero.

        Returns:
            The most recent smoothed position, or ``None`` if nothing has been
            fed yet.
        """
        if not self.started:
            return None
        return self.history_smooth[:, -1].copy()
