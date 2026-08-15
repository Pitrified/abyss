"""Test the camera config and the frame geometry built from it."""

import math

import pytest

from abyss.config.camera import MEDIAPIPE_VERTICAL_FOV_DEG
from abyss.config.camera import CameraConfig
from abyss.config.camera import FrameGeometry
from abyss.config.camera import InvalidFrameSizeError


def unmeasured() -> CameraConfig:
    """Build a camera with no intrinsics, which is every camera today."""
    return CameraConfig(name="test")


class TestFallbackFocal:
    """What happens when nothing has been measured."""

    @pytest.mark.parametrize("height", [480, 720, 1080, 1920])
    def test_follows_frame_height(self, height: int) -> None:
        expected = (height / 2) / math.tan(math.radians(31.5))
        assert unmeasured().focal_px_for_height(height) == pytest.approx(expected)

    def test_matches_the_measured_values(self) -> None:
        # Fitted by reprojection: 900 px on a 1080-tall clip, 1600 px on a
        # 1920-tall one, both 2% above the law. See the phase 1 plan.
        camera = unmeasured()
        assert camera.focal_px_for_height(1080) == pytest.approx(881, abs=1)
        assert camera.focal_px_for_height(1920) == pytest.approx(1567, abs=1)

    def test_the_assumption_is_the_documented_one(self) -> None:
        assert MEDIAPIPE_VERTICAL_FOV_DEG == 63.0

    def test_is_not_measured(self) -> None:
        assert unmeasured().is_measured is False


class TestFieldOfView:
    """A camera described by its field of view."""

    def test_round_trips_to_a_focal_length(self) -> None:
        camera = CameraConfig(name="test", fov_vertical_deg=60.0)
        focal = camera.focal_px_for_height(1080)
        recovered = 2 * math.degrees(math.atan((1080 / 2) / focal))
        assert recovered == pytest.approx(60.0)

    @pytest.mark.parametrize("height", [720, 1080, 1920])
    def test_scales_with_height(self, height: int) -> None:
        camera = CameraConfig(name="test", fov_vertical_deg=60.0)
        ratio = camera.focal_px_for_height(height) / height
        assert ratio == pytest.approx(camera.focal_px_for_height(1080) / 1080)

    def test_a_wider_view_is_a_shorter_focal(self) -> None:
        narrow = CameraConfig(name="n", fov_vertical_deg=40.0)
        wide = CameraConfig(name="w", fov_vertical_deg=90.0)
        assert wide.focal_px_for_height(1080) < narrow.focal_px_for_height(1080)

    def test_is_measured(self) -> None:
        assert CameraConfig(name="test", fov_vertical_deg=60.0).is_measured is True


class TestMeasuredFocal:
    """A camera described by a focal length someone measured."""

    def test_returns_it_at_the_height_it_was_measured_at(self) -> None:
        camera = CameraConfig(
            name="test", focal_px=1100.0, focal_measured_at_height=720
        )
        assert camera.focal_px_for_height(720) == pytest.approx(1100.0)

    def test_rescales_to_another_height(self) -> None:
        # The same lens at 1080 samples 1.5x more rows.
        camera = CameraConfig(
            name="test", focal_px=1100.0, focal_measured_at_height=720
        )
        assert camera.focal_px_for_height(1080) == pytest.approx(1650.0)


class TestValidation:
    """Malformed entries fail at construction, not ten frames into a run."""

    @pytest.mark.parametrize("fov", [0.0, -10.0, 180.0, 200.0])
    def test_a_degenerate_field_of_view_is_rejected(self, fov: float) -> None:
        with pytest.raises(ValueError, match="fov_vertical_deg"):
            CameraConfig(name="test", fov_vertical_deg=fov)

    def test_a_non_positive_focal_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="focal_px"):
            CameraConfig(name="test", focal_px=0.0, focal_measured_at_height=1080)

    def test_both_intrinsics_at_once_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="two ways to say the same thing"):
            CameraConfig(
                name="test",
                fov_vertical_deg=60.0,
                focal_px=1000.0,
                focal_measured_at_height=1080,
            )

    def test_a_focal_without_its_height_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="meaningless without its resolution"):
            CameraConfig(name="test", focal_px=1000.0)

    def test_the_config_is_frozen(self) -> None:
        camera = unmeasured()
        with pytest.raises(ValueError, match="frozen"):
            camera.mirrored = True


class TestFrameGeometry:
    """Binding a camera to the frames it actually produced."""

    def test_focal_comes_from_the_frame_height(self) -> None:
        geometry = FrameGeometry(camera=unmeasured(), width=1920, height=1080)
        assert geometry.focal == pytest.approx(881, abs=1)

    def test_the_same_camera_gives_a_different_focal_per_orientation(self) -> None:
        camera = unmeasured()
        landscape = FrameGeometry(camera=camera, width=1920, height=1080)
        portrait = FrameGeometry(camera=camera, width=1080, height=1920)
        assert portrait.focal > landscape.focal

    def test_principal_point_is_the_centre(self) -> None:
        geometry = FrameGeometry(camera=unmeasured(), width=1920, height=1080)
        assert geometry.principal_point == (960.0, 540.0)

    def test_width_does_not_affect_the_focal(self) -> None:
        camera = unmeasured()
        narrow = FrameGeometry(camera=camera, width=640, height=1080)
        wide = FrameGeometry(camera=camera, width=3840, height=1080)
        assert narrow.focal == wide.focal

    def test_mirrored_passes_through(self) -> None:
        camera = CameraConfig(name="test", mirrored=True)
        geometry = FrameGeometry(camera=camera, width=1920, height=1080)
        assert geometry.mirrored is True

    @pytest.mark.parametrize(("width", "height"), [(0, 480), (640, 0), (-1, -1)])
    def test_a_degenerate_frame_raises(self, width: int, height: int) -> None:
        with pytest.raises(InvalidFrameSizeError):
            FrameGeometry(camera=unmeasured(), width=width, height=height)

    def test_a_degenerate_height_raises_on_the_config_too(self) -> None:
        with pytest.raises(InvalidFrameSizeError):
            unmeasured().focal_px_for_height(0)
