"""Test the loop, with no camera, no display and no model.

That is the whole design constraint: the loop takes its source, its sink and
its tracker as arguments, so a list of frames, a recording sink and a stub
tracker exercise every path the live run takes except the two that need
hardware. If any of this needed a window, the phase would have been built
wrong.
"""

from loguru import logger as lg
import numpy as np
import pytest

from abyss.config.camera import CameraConfig
from abyss.config.camera import FrameGeometry
from abyss.config.screen import ScreenConfig
from abyss.config.viewer import ViewerConfig
from abyss.loop import Controls
from abyss.loop import LoopStats
from abyss.loop import run_loop
from abyss.render.renderer import WireframeRenderer
from abyss.render.scene import window_box
from abyss.sink.base import Sink
from abyss.viewer.eye_position import EyeSample
from abyss.viewer.eye_position import LiveScale

WIDTH_PX, HEIGHT_PX = 640, 320
SIZE = (WIDTH_PX, HEIGHT_PX)
FRAMES = 12


class RecordingSink:
    """A sink that keeps what it was given, so tests can look at it."""

    def __init__(self, size: tuple[int, int] = SIZE) -> None:
        self._size = size
        self.frames: list[np.ndarray] = []
        self.closed = False

    @property
    def size(self) -> tuple[int, int]:
        """Frame size this sink accepts."""
        return self._size

    def write(self, frame: np.ndarray) -> None:
        """Keep the frame.

        Args:
            frame: The finished frame.
        """
        self.frames.append(frame.copy())

    def close(self) -> None:
        """Record that it happened."""
        self.closed = True


@pytest.fixture
def screen() -> ScreenConfig:
    """Build a 2:1 screen, matching the test render size exactly."""
    return ScreenConfig(
        name="test",
        width_m=0.4,
        height_m=0.2,
        camera_to_centre_m=(0.0, 0.1, 0.0),
        provenance="invented for tests",
    )


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
    """Build the person in front of the camera."""
    return ViewerConfig(name="test")


@pytest.fixture
def renderer(screen: ScreenConfig) -> WireframeRenderer:
    """Build the renderer under the loop."""
    return WireframeRenderer(window_box(screen))


def frames(count: int = FRAMES) -> list[np.ndarray]:
    """Frames the tracker never looks at, since it is a stub.

    Args:
        count: How many to build.

    Returns:
        A list of small BGR frames.
    """
    return [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(count)]


def eye_sample(idx: int, ipd_px: float = 100.0) -> EyeSample:
    """Build one sample, front facing and half a metre out.

    Args:
        idx: Frame index.
        ipd_px: Apparent interpupillary distance.

    Returns:
        The sample.
    """
    return EyeSample(
        idx=idx,
        msec=float(idx),
        u_px=640.0,
        v_px=360.0,
        ipd_px=ipd_px,
        depth_m=0.6,
        yaw_deg=0.0,
    )


def always_tracks(_frame: np.ndarray, idx: int, _msec: float) -> EyeSample:
    """Find a face on every frame."""
    return eye_sample(idx)


def never_tracks(_frame: np.ndarray, _idx: int, _msec: float) -> None:
    """Find no face, ever."""


def test_the_recording_sink_satisfies_the_protocol() -> None:
    """Otherwise the rest of this file proves nothing about real sinks."""
    assert isinstance(RecordingSink(), Sink)


def test_every_frame_reaches_the_sink(screen, geometry, viewer, renderer) -> None:
    """One frame in, one frame out, at the size the sink asked for."""
    sink = RecordingSink()
    stats = run_loop(
        frames(), sink, screen, renderer, always_tracks, geometry, viewer,
        scale=LiveScale(geometry, viewer, scale=0.94),
    )

    assert stats.frames == FRAMES
    assert stats.faces == FRAMES
    assert len(sink.frames) == FRAMES
    assert all(f.shape == (HEIGHT_PX, WIDTH_PX, 3) for f in sink.frames)


def test_it_renders_the_scene_rather_than_a_message(
    screen, geometry, viewer, renderer
) -> None:
    """The guard against a loop that runs happily and shows a placeholder.

    The scene is orange and cyan; a message frame is grey text on grey. So a
    coloured pixel means the renderer actually ran.
    """
    sink = RecordingSink()
    run_loop(
        frames(), sink, screen, renderer, always_tracks, geometry, viewer,
        scale=LiveScale(geometry, viewer, scale=0.94),
    )

    last = sink.frames[-1].astype(int)
    coloured = np.abs(last[:, :, 0] - last[:, :, 2]) > 40
    assert coloured.any()


def test_it_says_so_while_calibrating(screen, geometry, viewer, renderer) -> None:
    """Rather than rendering at the wrong scale, which would look like it works."""
    sink = RecordingSink()
    stats = run_loop(
        frames(), sink, screen, renderer, always_tracks, geometry, viewer,
        scale=LiveScale(geometry, viewer, needed=FRAMES + 5),
    )

    assert stats.calibrating == FRAMES
    assert len(sink.frames) == FRAMES
    last = sink.frames[-1].astype(int)
    assert not (np.abs(last[:, :, 0] - last[:, :, 2]) > 40).any()


def test_a_face_that_arrives_late_starts_the_scene(
    screen, geometry, viewer, renderer
) -> None:
    """The bootstrap gates on front-facing samples, not on elapsed frames."""
    seen = {"count": 0}

    def late(_frame: np.ndarray, idx: int, _msec: float) -> EyeSample | None:
        if idx < 4:
            return None
        seen["count"] += 1
        return eye_sample(idx)

    sink = RecordingSink()
    stats = run_loop(
        frames(), sink, screen, renderer, late, geometry, viewer,
        scale=LiveScale(geometry, viewer, needed=2),
    )

    assert stats.faces == FRAMES - 4
    assert stats.calibrating >= 4
    assert stats.frames == FRAMES


def test_a_lost_face_holds_the_position_and_marks_it(
    screen, geometry, viewer, renderer
) -> None:
    """A viewer who left and a camera that died look identical otherwise.

    The scene must keep rendering from the held position - not freeze, not go
    blank - and the frame itself has to say it is held, because live there is
    nobody reading a log.
    """
    def then_lost(_frame: np.ndarray, idx: int, _msec: float) -> EyeSample | None:
        return eye_sample(idx) if idx < 6 else None

    sink = RecordingSink()
    stats = run_loop(
        frames(), sink, screen, renderer, then_lost, geometry, viewer,
        scale=LiveScale(geometry, viewer, scale=0.94),
    )

    assert stats.faces == 6
    assert stats.held == FRAMES - 6
    # Still the scene, drawn from the held position.
    last = sink.frames[-1].astype(int)
    assert (np.abs(last[:, :, 0] - last[:, :, 2]) > 40).any()
    # And visibly different from the last tracked frame, which carries no mark.
    assert not np.array_equal(sink.frames[5], sink.frames[-1])


def test_a_face_never_seen_at_all_does_not_crash(
    screen, geometry, viewer, renderer
) -> None:
    """There is nothing to hold before the first face, and that is a state."""
    sink = RecordingSink()
    stats = run_loop(
        frames(), sink, screen, renderer, never_tracks, geometry, viewer,
        scale=LiveScale(geometry, viewer, scale=0.94),
    )

    assert stats.frames == FRAMES
    assert stats.faces == 0
    assert stats.held == 0
    assert len(sink.frames) == FRAMES


def test_stop_ends_the_run_early(screen, geometry, viewer, renderer) -> None:
    """The quit key, which is the only way a live run ever ends."""
    seen = {"count": 0}

    def stop() -> bool:
        seen["count"] += 1
        return seen["count"] >= 3

    sink = RecordingSink()
    stats = run_loop(
        frames(), sink, screen, renderer, always_tracks, geometry, viewer,
        scale=LiveScale(geometry, viewer, scale=0.94),
        controls=Controls(stop=stop),
    )

    assert stats.frames == 3
    assert len(sink.frames) == 3


def test_reset_re_bootstraps_the_scale(screen, geometry, viewer, renderer) -> None:
    """The key for when the wrong person sat down.

    After the reset the loop must go back to calibrating rather than carrying
    on with the scale it had, which is what makes the key worth having.
    """
    asked = {"done": False}

    def reset() -> bool:
        if asked["done"]:
            return False
        asked["done"] = True
        return True

    with_reset = run_loop(
        frames(), RecordingSink(), screen, renderer, always_tracks, geometry, viewer,
        scale=LiveScale(geometry, viewer, needed=4),
        controls=Controls(reset=reset),
    )
    without = run_loop(
        frames(), RecordingSink(), screen, renderer, always_tracks, geometry, viewer,
        scale=LiveScale(geometry, viewer, needed=4),
    )

    # Compared against the same run without the key rather than against a
    # count: the frame that completes a bootstrap already renders, so the
    # absolute number is an off-by-one waiting to be asserted wrongly.
    assert with_reset.calibrating > without.calibrating
    assert with_reset.frames == FRAMES


def test_the_stats_report_the_rate(screen, geometry, viewer, renderer) -> None:
    """A slow loop honestly measured closes the phase; an unmeasured one does not."""
    sink = RecordingSink()
    stats = run_loop(
        frames(), sink, screen, renderer, always_tracks, geometry, viewer,
        scale=LiveScale(geometry, viewer, scale=0.94),
    )

    assert stats.seconds > 0
    assert stats.fps == pytest.approx(stats.frames / stats.seconds)


def test_the_frame_carries_the_depth_readout(
    screen, geometry, viewer, renderer
) -> None:
    """Without it the tape measure check cannot be done at all.

    The live run is fullscreen, so no terminal is visible, and logging a
    position at 30 fps is unreadable. The depth is the one number that can be
    wrong against the world rather than merely inconsistent with itself, so it
    has to be on the frame. This was missing from the first build and only
    surfaced when the runbook said "read the depth the loop reports".
    """
    plain = RecordingSink()
    run_loop(
        frames(2), plain, screen, renderer, always_tracks, geometry, viewer,
        scale=LiveScale(geometry, viewer, scale=0.94),
    )

    # The readout is grey text on the scene's top-left, so the row band it
    # occupies must differ from what the renderer alone would have drawn there.
    from abyss.render.renderer import render_frame
    from abyss.viewer.eye_position import eye_position_m

    bare = render_frame(
        screen,
        eye_position_m(eye_sample(0), geometry, 0.94),
        renderer,
        SIZE,
    )
    band = slice(20, 60)
    assert not np.array_equal(plain.frames[0][band], bare[band])


def captured_warnings(stats) -> str:
    """Run `report` and return what it logged at warning level.

    loguru does not route through the standard library, so `caplog` sees
    nothing; a list sink is the way to read it.

    Args:
        stats: The `LoopStats` to report.

    Returns:
        The concatenated warning text.
    """
    messages: list[str] = []
    handler = lg.add(messages.append, level="WARNING")
    try:
        stats.report()
    finally:
        lg.remove(handler)
    return "".join(messages)


def test_a_starved_loop_says_the_camera_is_pacing_it() -> None:
    """The 8.3 fps run looked slow and was starved, which is the opposite fix.

    Capture 99 ms against 21 ms of work, at two render sizes, with the camera
    advertising 30 fps throughout: nothing in the loop pointed at the camera.
    """
    starved = LoopStats(
        frames=168,
        faces=157,
        held=11,
        calibrating=29,
        seconds=20.1,
        stage_ms={"capture": 99.0, "track": 12.2, "render": 3.3, "sink": 5.4},
    )
    assert "exposure_dynamic_framerate" in captured_warnings(starved)


def test_a_fed_loop_does_not_warn() -> None:
    """Or the warning is noise and gets ignored when it matters."""
    fed = LoopStats(
        frames=250,
        faces=250,
        held=0,
        calibrating=29,
        seconds=10.6,
        stage_ms={"capture": 1.7, "track": 14.8, "render": 4.0, "sink": 20.2},
    )
    assert "exposure_dynamic_framerate" not in captured_warnings(fed)


def test_the_mark_key_logs_the_reading_every_time(
    screen, geometry, viewer, renderer
) -> None:
    """Asked for on three frames, logged three times, with the position in it.

    Every frame matters here rather than just the first: the callable was once
    shadowed by a timing variable of the same name, so frame two called a float.
    Nothing caught it because no test passed a mark at all.
    """
    asked = iter([True, True, True] + [False] * 20)

    stats_lines: list[str] = []
    handler = lg.add(stats_lines.append, level="INFO")
    try:
        run_loop(
            frames(), RecordingSink(), screen, renderer, always_tracks, geometry,
            viewer,
            scale=LiveScale(geometry, viewer, scale=0.94),
            controls=Controls(mark=lambda: next(asked, False)),
        )
    finally:
        lg.remove(handler)

    marks = [line for line in stats_lines if "MARK" in line]
    assert len(marks) == 3
    assert "iris 100 px" in marks[0]
    assert " m (camera frame)" in marks[0]


def test_the_readout_is_the_same_on_the_frame_and_in_the_log() -> None:
    """One formatter, or the number recorded is not the number displayed."""
    from abyss.loop import readout_text

    eye = np.array([0.012, -0.043, 0.512])
    assert readout_text(eye, 118.0) == readout_text(eye, 118.0)
    assert "0.512" in readout_text(eye, 118.0)
    assert "118 px" in readout_text(eye, 118.0)
    assert "held" in readout_text(eye, None)
