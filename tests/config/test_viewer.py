"""Test the viewer config."""

import pytest

from abyss.config.viewer import DEFAULT_IPD_M
from abyss.config.viewer import ViewerConfig


def test_the_default_is_the_population_mean() -> None:
    """The default interpupillary distance is the Q6 reference."""
    assert ViewerConfig(name="test").ipd_m == DEFAULT_IPD_M


def test_the_default_matches_phase_one() -> None:
    """Phase 1 used 63 mm, and its outputs have to stay reproducible."""
    assert DEFAULT_IPD_M == 0.063


@pytest.mark.parametrize("ipd", [0.0, -0.06])
def test_a_degenerate_ipd_is_rejected(ipd: float) -> None:
    """The scale correction divides by this, so it cannot be zero."""
    with pytest.raises(ValueError, match="ipd_m"):
        ViewerConfig(name="test", ipd_m=ipd)


def test_the_config_is_frozen() -> None:
    """Config is read at run time, never written."""
    viewer = ViewerConfig(name="test")
    with pytest.raises(ValueError, match="frozen"):
        viewer.ipd_m = 0.07
