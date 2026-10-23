"""The page's eye tracking, held to the Python pipeline and to the clip's path.

Three checks, each catching what the others cannot:

- **Against Python, frame by frame.** The page and ``abyss.ui.reference`` run
  the same model over the same JPEG bytes, and the whole chain after the
  landmarker (head scale bootstrap, eye position, smoothing) is replayed in
  Python on Python's own samples. A port bug shows up as a difference.
- **Against the path.** The clip moves along a known path, so direction and
  relative depth are known exactly. Catches an error both implementations
  share, which the first check cannot see.
- **The port's functions on golden inputs** (``test_port.py``), for the
  branches a clip never reaches: a mirrored camera, a measured focal length,
  a viewer turned away.

Tolerances were measured on 2026-10-22 and are set about twice the worst value
seen; ``plans/03_phone_webapp/03_tracking_in_page.md`` has the numbers.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from playwright.sync_api import Browser
import pytest

from abyss.config.camera import FrameGeometry
from abyss.params.abyss_devices import get_camera
from abyss.params.abyss_devices import get_viewer
from abyss.ui.browser import PageLog
from abyss.ui.browser import open_page
from abyss.ui.browser import wait_until
from abyss.ui.clip import BASE_SCALE
from abyss.ui.clip import FRAME_HEIGHT
from abyss.ui.clip import FRAME_WIDTH
from abyss.ui.clip import SEGMENT_FRAMES
from abyss.ui.clip import face_path
from abyss.ui.reference import track_mjpeg
from abyss.viewer.eye_position import EyeSample
from abyss.viewer.eye_position import LiveScale
from abyss.viewer.eye_position import eye_position_m
from abyss.viewer.smoothing import PositionSmoother

CLIP_CAMERA = "unknown_clip"
"""The clip has no real camera, so both sides use MediaPipe's assumed one."""

CLIP_TIMEOUT_MS = 180_000

TOLERANCE = {
    "u_px": 0.5,
    "v_px": 0.5,
    "ipd_px": 0.5,
    "depth_m": 0.001,
    "yaw_deg": 0.15,
}
"""Per-field limits on |page - Python|. Worst seen: 0.20 px, 0.10 px, 0.26 px,
0.30 mm, 0.054 deg."""

POSITION_TOLERANCE_M = 0.002
"""Limit on |page - Python| for eye positions, raw and smoothed."""

SCALE_TOLERANCE = 0.005
"""Limit on the difference between the two frozen head scales."""

REST_TOLERANCE_PX = 2.0
"""Pixel error allowed where the clip is still.

Elsewhere VIDEO mode lags: at peak speed the clip moves 22 px a frame and both
implementations trail it by up to 10.5 px. That is MediaPipe's tracking, not
the port, so pixel position is only checked at rest.
"""

DEPTH_RATIO_TOLERANCE = 0.06
"""Relative error allowed on depth against the path's scale. Seen: 3.5%."""


@dataclass
class ClipRun:
    """One run of the page over the clip, and what it logged."""

    track: list[dict]
    log: PageLog
    config: dict


def run_clip(browser: Browser, base_url: str, clip: Path, query: str = "") -> ClipRun:
    """Open the page on the clip and wait for it to finish every frame.

    Args:
        browser: The browser.
        base_url: The server.
        clip: The clip, served from ``/clips/``.
        query: Extra query parameters.

    Returns:
        The page's track, log and config.
    """
    url = f"{base_url}/phone_web/?source=/clips/{clip.name}&camera={CLIP_CAMERA}{query}"
    page, log = open_page(browser, url)
    wait_until(page, "['done', 'error'].includes(window.abyss.status)", CLIP_TIMEOUT_MS)
    assert page.evaluate("window.abyss.status") == "done", page.evaluate(
        "window.abyss.error"
    )
    run = ClipRun(
        page.evaluate("window.abyss.track"), log, page.evaluate("window.abyss.config")
    )
    page.context.close()
    return run


@pytest.fixture(scope="module")
def page_run(browser: Browser, base_url: str, clip: Path) -> ClipRun:
    """Run the page over the clip, bootstrapping its own head scale."""
    return run_clip(browser, base_url, clip)


@pytest.fixture(scope="module")
def python_samples(clip: Path) -> list[EyeSample | None]:
    """Run the Python pipeline over the same clip."""
    return track_mjpeg(clip, get_camera(CLIP_CAMERA))


@dataclass
class Replay:
    """The Python chain after the landmarker, frame by frame."""

    scale: float | None
    positions: list[list[float] | None]
    smoothed: list[list[float] | None]


@pytest.fixture(scope="module")
def python_replay(python_samples: list[EyeSample | None], page_run: ClipRun) -> Replay:
    """Replay head scale, eye position and smoothing in Python, as the page does."""
    geometry = FrameGeometry(get_camera(CLIP_CAMERA), FRAME_WIDTH, FRAME_HEIGHT)
    scale = LiveScale(geometry, get_viewer())
    smoother = PositionSmoother(page_run.config["constants"]["filter_size"])
    positions: list[list[float] | None] = []
    smoothed: list[list[float] | None] = []
    for sample in python_samples:
        if sample is None:
            held = smoother.hold()
            positions.append(None)
            smoothed.append(None if held is None else held.tolist())
            continue
        scale.update(sample)
        if not scale.is_ready:
            positions.append(None)
            smoothed.append(None)
            continue
        position = eye_position_m(sample, geometry, scale.scale)
        positions.append(position.tolist())
        smoothed.append(smoother.update(position).tolist())
    return Replay(scale.scale if scale.is_ready else None, positions, smoothed)


def test_every_frame_is_tracked_by_both(
    page_run: ClipRun, python_samples: list[EyeSample | None]
) -> None:
    """Same frame count, and a face found in exactly the same frames."""
    assert len(page_run.track) == len(python_samples)
    page_found = [entry["sample"] is not None for entry in page_run.track]
    assert page_found == [s is not None for s in python_samples]
    assert all(page_found)


def test_samples_match_python(
    page_run: ClipRun, python_samples: list[EyeSample | None]
) -> None:
    """Iris pixels, apparent IPD, depth and yaw agree within the tolerances."""
    for entry, reference in zip(page_run.track, python_samples, strict=True):
        assert reference is not None
        for field, limit in TOLERANCE.items():
            got, want = entry["sample"][field], getattr(reference, field)
            assert abs(got - want) <= limit, (
                f"frame {entry['idx']} {field}: {got} vs {want}"
            )


def test_head_scale_matches_python(page_run: ClipRun, python_replay: Replay) -> None:
    """Both bootstraps freeze, at the same frame, on the same value."""
    assert python_replay.scale is not None
    page_first = next(
        i for i, e in enumerate(page_run.track) if e["position"] is not None
    )
    python_first = next(
        i for i, p in enumerate(python_replay.positions) if p is not None
    )
    assert page_first == python_first
    page_scale = (
        page_run.track[-1]["position"][2] / page_run.track[-1]["sample"]["depth_m"]
    )
    assert abs(page_scale - python_replay.scale) <= SCALE_TOLERANCE


def test_positions_match_python(page_run: ClipRun, python_replay: Replay) -> None:
    """Eye positions, raw and smoothed, agree frame by frame."""
    for entry, raw, smooth in zip(
        page_run.track, python_replay.positions, python_replay.smoothed, strict=True
    ):
        for got, want in ((entry["position"], raw), (entry["smoothed"], smooth)):
            assert (got is None) == (want is None), f"frame {entry['idx']}"
            if got is not None and want is not None:
                for axis in range(3):
                    assert abs(got[axis] - want[axis]) <= POSITION_TOLERANCE_M, (
                        f"frame {entry['idx']} axis {axis}: {got} vs {want}"
                    )


def test_still_frames_sit_on_the_path(page_run: ClipRun) -> None:
    """Where the clip has been still for a while, the iris midpoint is on the path."""
    still = range(3 * SEGMENT_FRAMES - 10, 3 * SEGMENT_FRAMES)
    for idx in still:
        sample = page_run.track[idx]["sample"]
        point = face_path(idx)
        assert abs(sample["u_px"] - point.x_px) <= REST_TOLERANCE_PX, idx
        assert abs(sample["v_px"] - point.y_px) <= REST_TOLERANCE_PX, idx


def test_eye_moves_the_way_the_path_does(page_run: ClipRun) -> None:
    """Left in the image is towards -x, up is towards -y, closer is less depth.

    Movement is measured from the rest position, which sits above the image
    centre, so it is the change in position that has a sign, not the position.
    The clip's camera is not mirrored, so image left is camera-frame left.
    Checked where the path is at least 50 px from rest, so lag cannot flip it.
    """
    rest = face_path(SEGMENT_FRAMES)
    rest_entry = page_run.track[SEGMENT_FRAMES]
    rest_position = rest_entry["position"]
    rest_depth = rest_entry["sample"]["depth_m"]
    for entry in page_run.track:
        point = face_path(entry["idx"])
        position = entry["position"]
        if position is None:
            continue
        if abs(point.x_px - rest.x_px) > 50:
            moved_right = position[0] > rest_position[0]
            assert moved_right == (point.x_px > rest.x_px), entry["idx"]
        if abs(point.y_px - rest.y_px) > 50:
            moved_down = position[1] > rest_position[1]
            assert moved_down == (point.y_px > rest.y_px), entry["idx"]
        expected_ratio = BASE_SCALE / point.scale
        ratio = entry["sample"]["depth_m"] / rest_depth
        assert abs(ratio / expected_ratio - 1) <= DEPTH_RATIO_TOLERANCE, entry["idx"]


def test_usage_log_is_attempted_and_refused(page_run: ClipRun, base_url: str) -> None:
    """Closing the landmarker flushes its usage log, and the CSP stops it leaving.

    Both halves matter: without the attempt, a passing request check would say
    nothing about the policy.
    """
    assert page_run.log.off_server(base_url) == []
    refused = [
        line for line in page_run.log.console if "odml.pa.googleapis.com" in line
    ]
    assert refused, "tasks-vision did not try to send its log; this test checks nothing"
    assert all("Content Security Policy" in line for line in refused)


@pytest.fixture
def live_page(browser: Browser, base_url: str) -> Iterator:
    """Open the page on the fake camera, as a phone would open it on its own."""
    page, _ = open_page(browser, f"{base_url}/phone_web/?camera={CLIP_CAMERA}")
    yield page
    page.context.close()


def test_camera_path_tracks_and_calibrates(live_page) -> None:
    """Through getUserMedia: a face is found, the scale bootstraps, positions follow."""
    wait_until(live_page, "window.abyss.status === 'running'")
    wait_until(
        live_page, "window.abyss.scale && window.abyss.scale.ready", CLIP_TIMEOUT_MS
    )
    latest = wait_until(
        live_page, "window.abyss.latest && window.abyss.latest.position"
    )
    assert isinstance(latest, list)
    assert 0.2 < latest[2] < 1.0
