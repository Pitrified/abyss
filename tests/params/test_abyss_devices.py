"""Test the device registry."""

import pytest

from abyss.params.abyss_devices import CAMERAS
from abyss.params.abyss_devices import SAMPLE_CLIPS
from abyss.params.abyss_devices import SCREENS
from abyss.params.abyss_devices import VIEWERS
from abyss.params.abyss_devices import UnknownDeviceError
from abyss.params.abyss_devices import get_camera
from abyss.params.abyss_devices import get_screen
from abyss.params.abyss_devices import get_viewer
from abyss.params.abyss_devices import sample_stream


def test_every_entry_is_keyed_by_its_own_name() -> None:
    """A key that disagrees with the name inside it would confuse every log."""
    for registry in (CAMERAS, SCREENS, VIEWERS):
        for key, entry in registry.items():
            assert key == entry.name


def test_the_clip_camera_is_unmeasured() -> None:
    """The sample clips have to stay on phase 1's fallback path.

    This matters more now that a real camera is measured: the regression
    baseline is only comparable while the clips keep using MediaPipe's assumed
    field of view, so a measurement must never leak into them.
    """
    assert get_camera("unknown_clip").is_measured is False


def test_the_g7_webcam_is_measured() -> None:
    """Measured on g7 by ChArUco calibration, two runs agreeing to 0.5%."""
    camera = get_camera("g7_webcam")
    assert camera.is_measured is True
    assert camera.focal_px == pytest.approx(945.0)
    assert camera.focal_measured_at_height == 720


def test_the_g7_webcam_focal_rescales_with_height() -> None:
    """A focal in pixels is only valid at its own resolution.

    The camera also offers 640x480, and the two modes were never shown to
    share a vertical field of view, so this pins the arithmetic rather than
    the claim that rescaling to 480 is meaningful here.
    """
    camera = get_camera("g7_webcam")
    assert camera.focal_px_for_height(1440) == pytest.approx(1890.0)
    assert camera.focal_px_for_height(360) == pytest.approx(472.5)


def test_the_phone_camera_is_mirrored() -> None:
    """A front-facing phone camera mirrors its preview."""
    assert get_camera("pixel7pro_front").mirrored is True


def test_this_box_screen_matches_its_edid() -> None:
    """Read from the panel with scripts/read_edid.py: 309x173 mm."""
    screen = get_screen("g4_internal")
    assert screen.width_m == pytest.approx(0.309)
    assert screen.height_m == pytest.approx(0.173)


def test_the_screen_offset_is_flagged_as_provisional() -> None:
    """The bezel is a guess until someone measures it, and must say so."""
    assert "PROVISIONAL" in get_screen("g4_internal").provenance


def test_the_default_viewer_exists() -> None:
    """One viewer today, which is why nothing selects between them."""
    assert get_viewer().name == "default"


def test_a_sample_stream_points_into_the_pose_folder() -> None:
    """Clips live outside the repo, under AbyssPaths.pose_fol."""
    stream = sample_stream(SAMPLE_CLIPS[0])
    assert stream.camera == "unknown_clip"
    assert stream.is_live is False
    assert str(stream.source).endswith(SAMPLE_CLIPS[0])


@pytest.mark.parametrize(
    ("lookup", "kind"),
    [(get_camera, "camera"), (get_screen, "screen"), (get_viewer, "viewer")],
)
def test_an_unknown_name_names_what_is_known(lookup, kind: str) -> None:
    """A typo should say what the options were, not just fail."""
    with pytest.raises(UnknownDeviceError, match=kind):
        lookup("nope")
