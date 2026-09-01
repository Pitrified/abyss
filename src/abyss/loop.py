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
    x, y, z = (float(v) for v in eye_camera_m)
    iris = f"{ipd_px:.0f} px" if ipd_px is not None else "held"
    cv.putText(
        frame,
        f"eye {x:+.3f} {y:+.3f} {z:.3f} m (camera frame)   iris {iris}",
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
    stop: Callable[[], bool] | None = None,
    reset: Callable[[], bool] | None = None,
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
        stop: Called after each frame; the run ends when it returns true.
        reset: Called after each frame; the scale re-bootstraps when it returns
            true.

    Returns:
        What the run achieved.
    """
    live_scale = scale if scale is not None else LiveScale(geometry, viewer)
    smoother = PositionSmoother()
    counts = {"frames": 0, "faces": 0, "held": 0, "calibrating": 0}
    started = time.perf_counter()

    for idx, frame in enumerate(frames):
        counts["frames"] += 1
        msec = (time.perf_counter() - started) * MS_PER_S
        sample = track(frame, idx, msec)

        if sample is not None:
            counts["faces"] += 1
            live_scale.update(sample)

        if not live_scale.is_ready:
            counts["calibrating"] += 1
            have, need = live_scale.progress
            sink.write(
                message_frame(sink.size, f"Look at the camera: {have}/{need}")
            )
        else:
            position = (
                eye_position_m(sample, geometry, live_scale.scale)
                if sample is not None
                else None
            )
            smoothed = (
                smoother.update(position) if position is not None else smoother.hold()
            )
            if smoothed is None:
                # Nothing has ever been seen, so there is no position to hold.
                counts["calibrating"] += 1
                sink.write(message_frame(sink.size, "Waiting for a face"))
            else:
                out = render_frame(screen, smoothed, renderer, sink.size)
                annotate_position(
                    out, smoothed, sample.ipd_px if sample is not None else None
                )
                if sample is None:
                    counts["held"] += 1
                    mark_held(out)
                sink.write(out)

        if reset is not None and reset():
            live_scale.reset()
            smoother = PositionSmoother()
        if stop is not None and stop():
            break

    stats = LoopStats(
        frames=counts["frames"],
        faces=counts["faces"],
        held=counts["held"],
        calibrating=counts["calibrating"],
        seconds=time.perf_counter() - started,
    )
    stats.report()
    return stats
