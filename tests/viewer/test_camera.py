"""Test the camera assumptions."""

import math

import pytest

from abyss.viewer.camera import DEFAULT_IPD_M
from abyss.viewer.camera import CameraAssumptions
from abyss.viewer.camera import InvalidFrameSizeError


class TestAssumedFocal:
    """The focal length MediaPipe uses when it is given no intrinsics."""

    @pytest.mark.parametrize("height", [480, 720, 1080, 1920])
    def test_follows_frame_height(self, height: int) -> None:
        camera = CameraAssumptions(width=1280, height=height)
        expected = (height / 2) / math.tan(math.radians(31.5))
        assert camera.assumed_focal_px == pytest.approx(expected)

    def test_matches_the_measured_values(self) -> None:
        # Fitted by reprojection: 900 px on a 1080-tall clip, 1600 px on a
        # 1920-tall one, both 2% above the law. See the phase 1 plan.
        landscape = CameraAssumptions(width=1920, height=1080)
        portrait = CameraAssumptions(width=1080, height=1920)
        assert landscape.assumed_focal_px == pytest.approx(881, abs=1)
        assert portrait.assumed_focal_px == pytest.approx(1567, abs=1)

    def test_width_does_not_affect_it(self) -> None:
        narrow = CameraAssumptions(width=640, height=1080)
        wide = CameraAssumptions(width=3840, height=1080)
        assert narrow.assumed_focal_px == wide.assumed_focal_px


class TestFocal:
    """Which focal length the conversion uses."""

    def test_falls_back_to_the_assumption(self) -> None:
        camera = CameraAssumptions(width=1920, height=1080)
        assert camera.focal == camera.assumed_focal_px

    def test_prefers_a_known_focal(self) -> None:
        camera = CameraAssumptions(width=1920, height=1080, focal_px=1500.0)
        assert camera.focal == 1500.0


def test_principal_point_is_the_centre() -> None:
    """The principal point is assumed to be the frame centre."""
    camera = CameraAssumptions(width=1920, height=1080)
    assert camera.principal_point == (960.0, 540.0)


def test_default_ipd_is_the_population_mean() -> None:
    """The default interpupillary distance is the Q6 reference."""
    assert CameraAssumptions(width=640, height=480).ipd_m == DEFAULT_IPD_M


@pytest.mark.parametrize(("width", "height"), [(0, 480), (640, 0), (-1, -1)])
def test_a_degenerate_frame_raises(width: int, height: int) -> None:
    """A frame with a non-positive dimension is rejected."""
    with pytest.raises(InvalidFrameSizeError):
        CameraAssumptions(width=width, height=height)
