"""Test the position smoother."""

import numpy as np
import pytest

from abyss.viewer.smoothing import DEFAULT_FILTER_SIZE
from abyss.viewer.smoothing import PositionSmoother


def test_first_sample_passes_through() -> None:
    """A primed history means no ramp up from the origin."""
    smoother = PositionSmoother()
    first = np.array([0.1, -0.2, 0.5])
    assert np.allclose(smoother.update(first), first)


def test_no_ramp_from_zero() -> None:
    """Every early sample stays near the signal, not near zero."""
    smoother = PositionSmoother()
    position = np.array([0.0, 0.0, 0.5])
    for _ in range(DEFAULT_FILTER_SIZE):
        smoothed = smoother.update(position)
        assert smoothed[2] == pytest.approx(0.5)


def test_a_constant_signal_stays_constant() -> None:
    """A steady input comes out steady, not attenuated."""
    smoother = PositionSmoother()
    position = np.array([0.02, -0.03, 0.42])
    smoothed = position
    for _ in range(20):
        smoothed = smoother.update(position)
    assert np.allclose(smoothed, position)


def test_noise_is_reduced() -> None:
    """The whole point: less jitter out than in."""
    rng = np.random.default_rng(seed=17)
    smoother = PositionSmoother()
    raw, out = [], []
    for _ in range(200):
        position = np.array([0.0, 0.0, 0.5]) + rng.normal(0, 0.01, 3)
        raw.append(position)
        out.append(smoother.update(position))
    assert np.std(np.array(out)[:, 2]) < np.std(np.array(raw)[:, 2])


def test_a_step_is_followed() -> None:
    """Smoothing lags, but it must converge, not damp the signal away."""
    smoother = PositionSmoother()
    for _ in range(10):
        smoother.update(np.array([0.0, 0.0, 0.4]))
    smoothed = np.zeros(3)
    for _ in range(10):
        smoothed = smoother.update(np.array([0.0, 0.0, 0.8]))
    assert smoothed[2] == pytest.approx(0.8, abs=1e-6)


def test_is_causal() -> None:
    """Output at each step depends only on samples already fed."""
    a = PositionSmoother()
    b = PositionSmoother()
    positions = [np.array([0.0, 0.0, z]) for z in (0.4, 0.5, 0.6, 0.7)]
    prefix = [a.update(p) for p in positions[:2]]
    for p in positions:
        b.update(p)
    b2 = PositionSmoother()
    again = [b2.update(p) for p in positions[:2]]
    assert np.allclose(prefix, again)


class TestHold:
    """Frames with no face."""

    def test_before_any_sample_there_is_nothing_to_hold(self) -> None:
        assert PositionSmoother().hold() is None

    def test_returns_the_last_estimate(self) -> None:
        smoother = PositionSmoother()
        last = np.zeros(3)
        for z in (0.4, 0.5, 0.6):
            last = smoother.update(np.array([0.0, 0.0, z]))
        held = smoother.hold()
        assert held is not None
        assert np.allclose(held, last)

    def test_does_not_advance_the_filter(self) -> None:
        smoother = PositionSmoother()
        for z in (0.4, 0.5, 0.6):
            smoother.update(np.array([0.0, 0.0, z]))
        held_once = smoother.hold()
        held_twice = smoother.hold()
        assert held_once is not None
        assert held_twice is not None
        assert np.allclose(held_once, held_twice)

    def test_the_track_resumes_after_a_gap(self) -> None:
        smoother = PositionSmoother()
        for _ in range(10):
            smoother.update(np.array([0.0, 0.0, 0.5]))
        smoother.hold()
        resumed = smoother.update(np.array([0.0, 0.0, 0.5]))
        assert resumed[2] == pytest.approx(0.5)
