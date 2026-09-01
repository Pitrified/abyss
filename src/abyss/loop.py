"""Capture to sink: the loop that closes phase 5.

**The source and the sink are arguments**, which is what keeps this phase
testable at all. The same loop over a clip with a `PngSink` is phase 4's output
and needs no camera and no display; over a camera with a `WindowSink` it is the
live effect. If the loop could only be exercised through a window, the phase
would have been built wrong.

The tracker is an argument too, for the same reason one level down. The loop's
job is orchestration - what happens in what order, and what to do when a step
declines to produce anything - and a landmarker inside it would make every test
need a model file. `track_with_landmarker` builds the real one.

    capture -> landmark -> eye sample -> scale -> smooth -> frustum -> render -> sink

One thread, synchronous, and measured rather than paced (Q26). It runs as fast
as the source allows and reports what it achieved; the camera's own 30 fps cap
is the pacing.

Three states per frame, and the distinction between the last two is the point:

- **a face**: track, smooth, render
- **no face**: hold the last smoothed position and mark the frame, so a viewer
  who left looks different from a viewer who is still there
- **not yet calibrated**: say so on the frame rather than render at the wrong
  scale, which would look like a working loop (Q23)
"""

from collections.abc import Callable
from collections.abc import Iterable
from dataclasses import dataclass
import time

import cv2 as cv
from loguru import logger as lg
from mediapipe.tasks.python.vision.core.image import Image
from mediapipe.tasks.python.vision.core.image import ImageFormat
import numpy as np
from pose_tools.landmark.face import FaceLandmarkerFrame
from pose_tools.video.frame import Frame

from abyss.config.camera import FrameGeometry
from abyss.config.screen import ScreenConfig
from abyss.config.viewer import ViewerConfig
from abyss.render.renderer import Renderer
from abyss.render.renderer import render_frame
from abyss.sink.base import Sink
from abyss.viewer.eye_position import EyeSample
from abyss.viewer.eye_position import LiveScale
from abyss.viewer.eye_position import extract_eye_sample
from abyss.viewer.eye_position import eye_position_m
from abyss.viewer.smoothing import PositionSmoother

Tracker = Callable[[np.ndarray, int, float], EyeSample | None]
"""Turns a frame, its index and its timestamp into a sample, or nothing."""

MESSAGE_BGR = (200, 200, 200)
HELD_BGR = (60, 200, 255)
MESSAGE_ORIGIN = (24, 48)
HELD_ORIGIN = (24, 96)
MESSAGE_SCALE = 0.9
MESSAGE_THICKNESS = 2

MS_PER_S = 1000.0

CAPTURE_STALL_MS = 40.0
"""Capture median above which the camera is pacing the run, not the loop.

Above one camera frame at 30 fps, with a margin. A loop doing less work than it
spends waiting is not slow, it is starved, and the two have opposite fixes:
optimising a starved loop changes nothing at all.
"""

STAGE_NAMES = ("capture", "track", "render", "sink")
"""Stages timed inside the loop, in the order they run.

The benchmark times everything except the sink, because only `PngSink` exists
to time offline and a window is not a file. That gap is exactly where the first
live run lost its time - 120 ms per frame against a predicted 17 - so the loop
now measures itself and says so on exit. A stage nobody can measure is a stage
that will be the bottleneck.

The eye conversion is inside `render` rather than a stage of its own. The
benchmark measured it at 0.09 ms, so it will never be the answer, and a stage
that cannot be the bottleneck is noise in the report.

`capture` is pulling the next frame from the source, which for a camera is the
blocking `read`. It is timed here rather than in the source because the source
is an iterable and the loop is what waits on it. Expect it to be large and for
that to be correct: a loop running faster than 30 fps spends the remainder
waiting for the camera, which is the pacing, not a cost.
"""


def track_with_landmarker(
    landmarker: FaceLandmarkerFrame,
    geometry: FrameGeometry,
) -> Tracker:
    """Build the real tracker, over an open face landmarker.

    Timestamps are forced to increase by at least a millisecond per frame.
    MediaPipe's ``VIDEO`` mode requires strictly increasing timestamps, and the
    loop's wall clock does not guarantee that: replaying a clip faster than a
    thousand frames a second would hand it the same millisecond twice. The
    clamp costs nothing live, where frames are 33 ms apart.

    Args:
        landmarker: An open landmarker, built with
            ``output_facial_transformation_matrixes=True``.
        geometry: Frame geometry of the source.

    Returns:
        A tracker for :func:`run_loop`.
    """
    last = -1.0

    def track(frame: np.ndarray, idx: int, msec: float) -> EyeSample | None:
        nonlocal last
        stamp = max(msec, last + 1.0)
        last = stamp
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = landmarker.detect(
            Frame(
                image=Image(image_format=ImageFormat.SRGB, data=rgb),
                msec=stamp,
                idx=idx,
            )
        )
        return extract_eye_sample(result, geometry, idx, stamp)

    return track


@dataclass(frozen=True)
class Controls:
    """What the viewer asked for, as the sink reports it.

    One object rather than three arguments because they are one concept: keys
    pressed in the window, read between frames. The loop never learns what kind
    of sink it has, which is what keeps it runnable over a clip with no display
    and no keyboard - the default is all three absent.

    Args:
        stop: Returns true when the run should end.
        reset: Returns true when the head scale bootstrap should run again.
        mark: Returns true when the current reading should be logged, which is
            how the tape measure check takes its numbers.
    """

    stop: Callable[[], bool] | None = None
    reset: Callable[[], bool] | None = None
    mark: Callable[[], bool] | None = None


@dataclass(frozen=True)
class LoopStats:
    """What a run achieved.

    Args:
        frames: Frames pulled from the source.
        faces: Frames a face was found in.
        held: Frames rendered from a held position, no face seen.
        calibrating: Frames shown before the head scale had bootstrapped.
        seconds: Wall clock time of the run.
    """

    frames: int
    faces: int
    held: int
    calibrating: int
    seconds: float
    stage_ms: dict[str, float]

    @property
    def fps(self) -> float:
        """Frames per second achieved over the whole run."""
        return self.frames / self.seconds if self.seconds > 0 else 0.0

    def report(self) -> None:
        """Log the run, which is a phase exit criterion rather than a nicety."""
        lg.info(
            f"{self.frames} frames in {self.seconds:.1f} s, {self.fps:.1f} fps: "
            f"{self.faces} with a face, {self.held} held, "
            f"{self.calibrating} calibrating"
        )
        self._warn_if_starved()
        timed = " ".join(
            f"{name} {self.stage_ms[name]:.1f}" for name in STAGE_NAMES
            if name in self.stage_ms
        )
        accounted = sum(self.stage_ms.values())
        per_frame = self.seconds * MS_PER_S / self.frames if self.frames else 0.0
        lg.info(
            f"median ms per frame: {timed} | measured {accounted:.1f} of "
            f"{per_frame:.1f} actual"
        )

    def _warn_if_starved(self) -> None:
        """Say so when the camera, not the loop, is setting the rate.

        Measured on g7 on 2026-09-01: capture 99 ms against 21 ms of work, so
        8.3 fps where the loop could manage 47. The cause was
        ``exposure_dynamic_framerate``, a UVC control that lets the camera drop
        its own rate to buy longer exposures in a dim room. It defaults to off
        and was on, the camera advertised 30 fps throughout, and the frame rate
        was identical at two render sizes - so nothing about the loop pointed
        at it. Naming it here costs one branch and saves the next person the
        afternoon it cost this time.
        """
        capture = self.stage_ms.get("capture", 0.0)
        work = sum(ms for name, ms in self.stage_ms.items() if name != "capture")
        if capture <= CAPTURE_STALL_MS or capture <= work:
            return
        lg.warning(
            f"Capture is {capture:.0f} ms against {work:.0f} ms of work: the "
            f"camera is pacing this run, so the loop could reach "
            f"{MS_PER_S / work:.0f} fps if fed. On a UVC webcam the usual cause "
            f"is exposure_dynamic_framerate, which trades frame rate for "
            f"exposure in low light. Check with "
            f"'v4l2-ctl -d /dev/video0 --list-ctrls' and turn it off with "
            f"'v4l2-ctl -d /dev/video0 -c exposure_dynamic_framerate=0'"
        )


def message_frame(
    size: tuple[int, int],
    text: str,
    background: tuple[int, int, int] = (16, 16, 16),
) -> np.ndarray:
    """Build a frame that says something instead of showing the scene.

    Args:
        size: The ``(width_px, height_px)`` to build.
        text: What to write on it.
        background: Fill colour, BGR.

    Returns:
        A BGR image of that size.
    """
    width, height = size
    frame = np.empty((height, width, 3), dtype=np.uint8)
    for channel, value in enumerate(background):
        frame[:, :, channel] = value
    cv.putText(
        frame,
        text,
        MESSAGE_ORIGIN,
        cv.FONT_HERSHEY_SIMPLEX,
        MESSAGE_SCALE,
        MESSAGE_BGR,
        MESSAGE_THICKNESS,
        cv.LINE_AA,
    )
    return frame


def readout_text(eye_camera_m: np.ndarray, ipd_px: float | None) -> str:
    """Format the tracked position for the frame and for the log.

    One function so the number pressing the mark key records is character for
    character the number on screen. Two formatters would drift.

    Args:
        eye_camera_m: Eye position in the camera frame, metres.
        ipd_px: Apparent interpupillary distance, or ``None`` when held.

    Returns:
        The readout line.
    """
    x, y, z = (float(v) for v in eye_camera_m)
    iris = f"{ipd_px:.0f} px" if ipd_px is not None else "held"
    return f"eye {x:+.3f} {y:+.3f} {z:.3f} m (camera frame)   iris {iris}"


def annotate_position(
    frame: np.ndarray,
    eye_camera_m: np.ndarray,
    ipd_px: float | None,
) -> None:
    """Write the tracked position onto the frame, in place.

    **This is what makes the tape measure check possible at all.** The depth is
    the one number in the loop that can be wrong against the world rather than
    merely inconsistent with itself, and there is nowhere else to read it: the
    run is fullscreen, so a terminal is not visible, and per-frame logging at
    30 fps is unreadable. The apparent iris separation goes next to it because
    the prediction table is written in those terms, so the frame carries both
    halves of the comparison.

    Args:
        frame: The image to draw on.
        eye_camera_m: Eye position in the camera frame, metres.
        ipd_px: Apparent interpupillary distance, or ``None`` when the position
            is held rather than measured.
    """
    cv.putText(
        frame,
        readout_text(eye_camera_m, ipd_px),
        MESSAGE_ORIGIN,
        cv.FONT_HERSHEY_SIMPLEX,
        MESSAGE_SCALE,
        MESSAGE_BGR,
        MESSAGE_THICKNESS,
        cv.LINE_AA,
    )


def mark_held(frame: np.ndarray) -> None:
    """Write the held-position warning onto a frame, in place.

    A camera that died and a viewer who left produce the same still scene, and
    only the frame itself can say which. Phase 4 established this on the offline
    render; live it matters more, because there is nobody reading a log.

    Args:
        frame: The image to draw on.
    """
    cv.putText(
        frame,
        "HELD: no face",
        HELD_ORIGIN,
        cv.FONT_HERSHEY_SIMPLEX,
        MESSAGE_SCALE,
        HELD_BGR,
        MESSAGE_THICKNESS,
        cv.LINE_AA,
    )


def draw_scene(
    screen: ScreenConfig,
    renderer: Renderer,
    size: tuple[int, int],
    smoothed: np.ndarray,
    sample: EyeSample | None,
) -> np.ndarray:
    """Render one frame and put the readouts on it.

    Args:
        screen: The panel being looked through.
        renderer: What draws the scene.
        size: Output ``(width_px, height_px)``.
        smoothed: Smoothed eye position in the camera frame, metres.
        sample: This frame's measurements, or ``None`` when the position is
            held rather than measured.

    Returns:
        A BGR image of that size.
    """
    out = render_frame(screen, smoothed, renderer, size)
    annotate_position(out, smoothed, sample.ipd_px if sample is not None else None)
    if sample is None:
        mark_held(out)
    return out


def frame_for(
    sample: EyeSample | None,
    live_scale: LiveScale,
    smoother: PositionSmoother,
    screen: ScreenConfig,
    renderer: Renderer,
    geometry: FrameGeometry,
    size: tuple[int, int],
) -> tuple[np.ndarray, str, str | None]:
    """Decide what this frame should show, and draw it.

    The three states of the loop live here, which is why they are one function
    rather than a branch in the middle of the timing code: a face, no face but
    a position to hold, and no scale yet to render at.

    Args:
        sample: This frame's measurements, or ``None`` when no face was found.
        live_scale: The head scale estimator.
        smoother: The position smoother.
        screen: The panel being looked through.
        renderer: What draws the scene.
        geometry: Frame geometry of the source.
        size: Output ``(width_px, height_px)``.

    Returns:
        The frame, which state produced it (``"calibrating"``, ``"held"`` or
        ``"tracked"``), and the readout line when there is a position.
    """
    if not live_scale.is_ready:
        have, need = live_scale.progress
        text = f"Look at the camera: {have}/{need}"
        return message_frame(size, text), "calibrating", None

    position = (
        eye_position_m(sample, geometry, live_scale.scale)
        if sample is not None
        else None
    )
    smoothed = smoother.update(position) if position is not None else smoother.hold()
    if smoothed is None:
        # Nothing has ever been seen, so there is no position to hold.
        return message_frame(size, "Waiting for a face"), "calibrating", None

    kind = "tracked" if sample is not None else "held"
    ipd_px = sample.ipd_px if sample is not None else None
    drawn = draw_scene(screen, renderer, size, smoothed, sample)
    return drawn, kind, readout_text(smoothed, ipd_px)


def apply_controls(
    live_scale: LiveScale,
    reading: str | None,
    controls: Controls,
) -> bool:
    """Act on the keys the sink reports, between frames.

    Args:
        live_scale: The head scale estimator, which a reset clears.
        reading: This frame's readout line, which a mark records.
        controls: What the viewer asked for.

    Returns:
        Whether the scale was reset, so the caller can clear the smoother too.
    """
    if controls.mark is not None and controls.mark():
        lg.info(f"MARK  {reading or 'no position yet'}")
    if controls.reset is not None and controls.reset():
        live_scale.reset()
        return True
    return False


def stage_medians(timings: dict[str, list[float]]) -> dict[str, float]:
    """Reduce the per-frame timings to one median per stage.

    Args:
        timings: Durations in milliseconds, per stage.

    Returns:
        The median for each stage that ran at all.
    """
    return {
        name: float(np.median(values)) for name, values in timings.items() if values
    }


def run_loop(
    frames: Iterable[np.ndarray],
    sink: Sink,
    screen: ScreenConfig,
    renderer: Renderer,
    track: Tracker,
    geometry: FrameGeometry,
    viewer: ViewerConfig,
    *,
    scale: LiveScale | None = None,
    controls: Controls | None = None,
) -> LoopStats:
    """Run frames through the whole chain and into the sink.

    Args:
        frames: The source. Anything iterable of BGR frames: a clip reader, a
            camera, or a list in a test.
        sink: Where finished frames go. Its `size` is what gets rendered.
        screen: The panel being looked through.
        renderer: What draws the scene.
        track: Turns a frame into an eye sample, or nothing.
        geometry: Frame geometry of the source.
        viewer: The person in front of the camera.
        scale: The head scale estimator. A fresh bootstrapping one by default.
        controls: What the viewer asked for, checked after each frame. All
            absent by default, which is the offline case.

    Returns:
        What the run achieved.
    """
    live_scale = scale if scale is not None else LiveScale(geometry, viewer)
    keys = controls if controls is not None else Controls()
    smoother = PositionSmoother()
    counts = {"frames": 0, "faces": 0, "held": 0, "calibrating": 0}
    started = time.perf_counter()

    timings: dict[str, list[float]] = {name: [] for name in STAGE_NAMES}
    source = iter(frames)
    idx = -1

    while True:
        started_stage = time.perf_counter()
        frame = next(source, None)
        capture_ms = (time.perf_counter() - started_stage) * MS_PER_S
        if frame is None:
            break
        idx += 1
        timings["capture"].append(capture_ms)
        counts["frames"] += 1
        msec = (time.perf_counter() - started) * MS_PER_S

        started_stage = time.perf_counter()
        sample = track(frame, idx, msec)
        timings["track"].append((time.perf_counter() - started_stage) * MS_PER_S)

        if sample is not None:
            counts["faces"] += 1
            live_scale.update(sample)

        started_stage = time.perf_counter()
        out, kind, reading = frame_for(
            sample, live_scale, smoother, screen, renderer, geometry, sink.size
        )
        timings["render"].append((time.perf_counter() - started_stage) * MS_PER_S)
        if kind in counts:
            counts[kind] += 1

        started_stage = time.perf_counter()
        sink.write(out)
        timings["sink"].append((time.perf_counter() - started_stage) * MS_PER_S)

        if apply_controls(live_scale, reading, keys):
            smoother = PositionSmoother()
        if keys.stop is not None and keys.stop():
            break

    stats = LoopStats(
        frames=counts["frames"],
        faces=counts["faces"],
        held=counts["held"],
        calibrating=counts["calibrating"],
        seconds=time.perf_counter() - started,
        stage_ms=stage_medians(timings),
    )
    stats.report()
    return stats
