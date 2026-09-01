"""Test the live head scale estimator.

Its own file rather than an addition to `test_eye_position.py`, because that
module patches the pose-tools accessors for every test in it and none of this
needs a landmarker result at all - `LiveScale` consumes samples, not frames.
"""

import pytest

from abyss.config.camera import CameraConfig
from abyss.config.camera import FrameGeometry
from abyss.config.viewer import ViewerConfig
from abyss.viewer.eye_position import FRONT_FACING_YAW_DEG
from abyss.viewer.eye_position import EyeSample
from abyss.viewer.eye_position import HeadScaleNotReadyError
from abyss.viewer.eye_position import LiveScale
from abyss.viewer.eye_position import estimate_head_scale

NEEDED = 5


@pytest.fixture
def geometry() -> FrameGeometry:
    """Build a measured camera bound to a 1280x720 frame."""
    camera = CameraConfig(
        name="test",
        focal_px=945.0,
        focal_measured_at_height=720,
        provenance="g7's measured focal, for tests",
    )
    return FrameGeometry(camera=camera, width=1280, height=720)


@pytest.fixture
def viewer() -> ViewerConfig:
    """Build the person the scale corrects towards."""
    return ViewerConfig(name="test")


def sample(ipd_px: float, yaw_deg: float = 0.0, depth_m: float = 0.6) -> EyeSample:
    """Build one sample.

    Args:
        ipd_px: Apparent interpupillary distance in pixels.
        yaw_deg: Head yaw, which is what gates the bootstrap.
        depth_m: Eye depth before scaling.

    Returns:
        The sample.
    """
    return EyeSample(
        idx=0,
        msec=0.0,
        u_px=640.0,
        v_px=360.0,
        ipd_px=ipd_px,
        depth_m=depth_m,
        yaw_deg=yaw_deg,
    )


def test_it_is_not_ready_before_the_bootstrap(geometry, viewer) -> None:
    """Reading it early raises rather than returning 1.0.

    A default of 1.0 would render the whole scene at the wrong depth and look
    like a working loop, which is the failure this class exists to prevent.
    """
    live = LiveScale(geometry, viewer, needed=NEEDED)

    assert not live.is_ready
    assert live.progress == (0, NEEDED)
    with pytest.raises(HeadScaleNotReadyError):
        _ = live.scale


def test_it_freezes_once_it_has_enough(geometry, viewer) -> None:
    """And reports progress on the way there, which the loop displays."""
    live = LiveScale(geometry, viewer, needed=NEEDED)
    for i in range(NEEDED - 1):
        live.update(sample(100.0))
        assert live.progress == (i + 1, NEEDED)
        assert not live.is_ready

    live.update(sample(100.0))
    assert live.is_ready
    assert live.scale > 0


def test_a_later_outlier_does_not_move_it(geometry, viewer) -> None:
    """Frozen means frozen: the scene must not breathe mid-run (Q23)."""
    live = LiveScale(geometry, viewer, needed=NEEDED)
    for _ in range(NEEDED):
        live.update(sample(100.0))
    frozen = live.scale

    for _ in range(50):
        live.update(sample(220.0))

    assert live.scale == frozen


def test_the_order_of_the_bootstrap_samples_does_not_matter(geometry, viewer) -> None:
    """It is a median over the collected set, not a running value."""
    widths = [96.0, 104.0, 100.0, 112.0, 92.0]

    forward = LiveScale(geometry, viewer, needed=NEEDED)
    backward = LiveScale(geometry, viewer, needed=NEEDED)
    for width in widths:
        forward.update(sample(width))
    for width in reversed(widths):
        backward.update(sample(width))

    assert forward.scale == pytest.approx(backward.scale)


def test_turned_away_frames_do_not_count(geometry, viewer) -> None:
    """Apparent interpupillary distance shrinks under yaw.

    Counting those would bootstrap the scale off a foreshortened face and
    freeze the error in for the whole run, which is worse than waiting.
    """
    live = LiveScale(geometry, viewer, needed=NEEDED)
    for _ in range(20):
        live.update(sample(100.0, yaw_deg=FRONT_FACING_YAW_DEG + 5))

    assert not live.is_ready
    assert live.progress == (0, NEEDED)


def test_it_agrees_with_the_offline_estimator(geometry, viewer) -> None:
    """Live and offline must not drift apart in how they compute it.

    `LiveScale` calls `estimate_head_scale` on its buffer rather than
    reimplementing the formula, and this pins that they stay the same answer
    for the same samples - which is what makes a clip replay comparable to a
    live run.
    """
    widths = [96.0, 104.0, 100.0, 112.0, 92.0]
    samples = [sample(width) for width in widths]

    live = LiveScale(geometry, viewer, needed=NEEDED)
    for one in samples:
        live.update(one)

    assert live.scale == pytest.approx(
        estimate_head_scale(samples, geometry, viewer)
    )


def test_a_reset_bootstraps_again(geometry, viewer) -> None:
    """The key the live run offers when the wrong person sat down."""
    live = LiveScale(geometry, viewer, needed=NEEDED)
    for _ in range(NEEDED):
        live.update(sample(100.0))
    assert live.is_ready

    live.reset()

    assert not live.is_ready
    assert live.progress == (0, NEEDED)


def test_it_can_start_frozen(geometry, viewer) -> None:
    """For replaying a clip at a scale measured over the whole of it."""
    live = LiveScale(geometry, viewer, scale=0.942)

    assert live.is_ready
    assert live.scale == pytest.approx(0.942)
