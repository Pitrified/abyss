"""Time the phase 5 loop's stages against a recorded clip.

Step one of phase 5, and deliberately not part of the loop. Everything here runs
off a clip, so it needs no camera, no display and nobody sitting still, which is
what lets the same script run on any machine in the fleet and produce comparable
rows. Results go to ``cache/benchmark/`` and into the plan log, per machine.

    uv run --no-sync python scripts/benchmark_landmarker.py
    uv run --no-sync python scripts/benchmark_landmarker.py --frames 120
    uv run --no-sync python scripts/benchmark_landmarker.py --input-sizes 1280x720

The question it answers has already changed once, which is the argument for
building it rather than guessing. It was going to be "is the GPU delegate
faster"; a probe run before any of this settled that (the wheel is built without
GPU support, and CPU inference is 11.2 ms at 1920x1080 anyway), so the job now is
to find where the time actually goes. The delegate axis is dropped rather than
skipped: a GPU row could only ever record the same build-flags failure.

**Two size axes, not one.** The plan had a single frame-size axis, which does not
survive contact: the tracker stages scale with the *capture* size while the
render stages scale with the *output* size, and they are different numbers in the
live loop - 1280x720 in, 1920x1080 out. Worse, the plan's cheap fallback of
640x480 is 4:3, so rendering into it raises `AspectMismatchError` against a 16:9
panel. Splitting the axis is what makes both halves measurable.

**There is no capture stage here, on purpose.** Timing an mp4 decode and calling
it capture would be a proxy, and a bad one: a V4L2 MJPG read costs queue latency
and JPEG decode that a seek-free file read does not. The decode is timed and
named `decode`, which is what it is. Real capture timing arrives with
``abyss/video/capture.py`` in step 2, where it can use the real opener rather
than a second copy of it.
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import platform
import time

import cv2 as cv
from loguru import logger as lg
from mediapipe.tasks.python.vision.core.image import Image
from mediapipe.tasks.python.vision.core.image import ImageFormat
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode as VisionRunningMode,
)
import numpy as np
from pose_tools.landmark.face import FaceLandmarkerFrame
from pose_tools.landmark.model_manager import ModelManager
from pose_tools.video.frame import Frame

from abyss.config.camera import FrameGeometry
from abyss.config.screen import ScreenConfig
from abyss.config.sink import SinkConfig
from abyss.params.abyss_devices import get_camera
from abyss.params.abyss_devices import get_screen
from abyss.params.abyss_params import get_abyss_paths
from abyss.render.frustum import CAMERA_TO_SCREEN
from abyss.render.frustum import view_projection_matrix
from abyss.render.renderer import WireframeRenderer
from abyss.render.renderer import check_aspect
from abyss.render.scene import window_box
from abyss.sink.file import PngSink
from abyss.viewer.eye_position import extract_eye_sample
from abyss.viewer.eye_position import eye_position_m

DEFAULT_CLIP = "face01.mp4"
DEFAULT_CAMERA = "g7_webcam"
DEFAULT_SCREEN = "g7_internal"

DEFAULT_INPUT_SIZES = "1280x720,640x480"
"""Capture sizes to time the tracker stages at.

Both are real modes on the g7 webcam. 1280x720 is the one the focal length was
measured at and the one phase 5 pins; 640x480 is the cheap fallback, kept as a
row because knowing what it would buy is the point of measuring it.
"""

DEFAULT_OUTPUT_SIZES = "1920x1080,1280x720"
"""Render sizes to time the render stages at.

The panel's native resolution first, since phase 5 renders at it rather than
letting OpenCV upscale. Both are 16:9 and pass the panel's aspect check.
"""

DEFAULT_FRAMES = 60
WARMUP_FRAMES = 5
"""Frames discarded before timing starts, so lazy allocation is not measured."""

CAMERA_BUDGET_MS = 1000.0 / 30
"""The frame budget the camera imposes.

The Chicony webcam caps at 30 fps in every mode, read from the device with
v4l2-ctl, so a stage is only worth optimising against this rather than against
whatever the CPU could manage.
"""

P95 = 95

TRACKER_STAGES = ("decode", "landmark", "eye_position")
RENDER_STAGES = ("projection", "render", "sink")

CSV_FIELDS = [
    "machine",
    "path",
    "stage",
    "width_px",
    "height_px",
    "frames",
    "median_ms",
    "p95_ms",
    "stage_fps",
]

BENCH_SWEEP_M = 0.08
"""Amplitude of the eye path the render stages are driven with.

Only that it varies matters here - a fixed eye position would let nothing vary
between frames and could flatter a cache. The amplitude that keeps the scene
composed is `render_scene.py`'s concern, not this script's.
"""

BENCH_DISTANCE_M = 0.55


class NoFramesTimedError(RuntimeError):
    """Raised when a clip yielded no frames to time."""

    def __init__(self, clip: Path) -> None:
        """Initialise with the clip that produced nothing.

        Args:
            clip: The clip that was read.
        """
        super().__init__(
            f"{clip} yielded no frames past the {WARMUP_FRAMES} warm-up frames"
        )


@dataclass
class StageTimes:
    """Timings for one stage at one size.

    Args:
        path: Which half of the loop this stage belongs to.
        stage: Stage name.
        width: Frame width the stage ran at.
        height: Frame height the stage ran at.
        samples_ms: One duration per frame, in milliseconds.
    """

    path: str
    stage: str
    width: int
    height: int
    samples_ms: list[float]

    @property
    def median_ms(self) -> float:
        """Median duration in milliseconds."""
        return float(np.median(self.samples_ms))

    @property
    def p95_ms(self) -> float:
        """95th percentile duration in milliseconds."""
        return float(np.percentile(self.samples_ms, P95))

    @property
    def stage_fps(self) -> float:
        """Frame rate this stage alone would allow."""
        return 1000.0 / self.median_ms if self.median_ms > 0 else float("inf")

    def as_row(self, machine: str) -> dict:
        """Render as a CSV row.

        Args:
            machine: Hostname the timings came from.

        Returns:
            One dict keyed by :data:`CSV_FIELDS`.
        """
        return {
            "machine": machine,
            "path": self.path,
            "stage": self.stage,
            "width_px": self.width,
            "height_px": self.height,
            "frames": len(self.samples_ms),
            "median_ms": round(self.median_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "stage_fps": round(self.stage_fps, 1),
        }


def parse_sizes(text: str) -> list[tuple[int, int]]:
    """Parse a comma-separated list of ``WIDTHxHEIGHT`` sizes.

    Args:
        text: For example ``"1280x720,640x480"``.

    Returns:
        The sizes as ``(width, height)`` pairs.
    """
    sizes = []
    for part in text.split(","):
        width, height = part.strip().lower().split("x")
        sizes.append((int(width), int(height)))
    return sizes


def time_tracker_path(
    clip: Path,
    camera_name: str,
    size: tuple[int, int],
    frames: int,
) -> list[StageTimes]:
    """Time decode, landmarking and the eye conversion at one capture size.

    The clip is decoded at its own resolution and resized to `size`, and the
    resize is **not** timed: it stands in for a camera that would have handed
    over that size directly, so charging the loop for it would be measuring the
    stand-in rather than the loop.

    Args:
        clip: Video file to read.
        camera_name: Registry key for the camera the geometry comes from.
        size: Capture size to time at, as ``(width, height)``.
        frames: How many frames to time, past the warm-up.

    Returns:
        One :class:`StageTimes` per tracker stage.

    Raises:
        NoFramesTimedError: If the clip ran out during the warm-up.
    """
    width, height = size
    capture = cv.VideoCapture(str(clip))
    fps = capture.get(cv.CAP_PROP_FPS)
    geometry = FrameGeometry(camera=get_camera(camera_name), width=width, height=height)
    # No head scale: `estimate_head_scale` needs the whole clip, and live it is
    # a frozen constant, so per frame it costs one multiply either way.
    model_path = ModelManager().ensure_model("face_landmarker")

    samples: dict[str, list[float]] = {stage: [] for stage in TRACKER_STAGES}
    faces = 0
    landmarker_kwargs = {
        "running_mode": VisionRunningMode.VIDEO,
        "num_faces": 1,
        "output_facial_transformation_matrixes": True,
    }
    with FaceLandmarkerFrame(model_path, landmarker_kwargs) as landmarker:
        for idx in range(frames + WARMUP_FRAMES):
            start = time.perf_counter()
            ok, bgr = capture.read()
            decode_s = time.perf_counter() - start
            if not ok:
                break

            bgr = cv.resize(bgr, (width, height))
            msec = idx * 1000.0 / fps

            start = time.perf_counter()
            rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
            frame = Frame(
                image=Image(image_format=ImageFormat.SRGB, data=rgb),
                msec=msec,
                idx=idx,
            )
            result = landmarker.detect(frame)
            landmark_s = time.perf_counter() - start

            start = time.perf_counter()
            sample = extract_eye_sample(result, geometry, idx, msec)
            if sample is not None:
                eye_position_m(sample, geometry, 1.0)
            eye_s = time.perf_counter() - start

            if idx < WARMUP_FRAMES:
                continue
            faces += sample is not None
            samples["decode"].append(decode_s * 1000)
            samples["landmark"].append(landmark_s * 1000)
            samples["eye_position"].append(eye_s * 1000)
    capture.release()

    if not samples["decode"]:
        raise NoFramesTimedError(clip)
    counted = len(samples["decode"])
    lg.info(f"tracker {width}x{height}: {faces}/{counted} frames with a face")
    return [
        StageTimes("tracker", stage, width, height, samples[stage])
        for stage in TRACKER_STAGES
    ]


def bench_eye_positions(screen: ScreenConfig, frames: int) -> np.ndarray:
    """Build an eye path for the render stages, in the camera frame.

    Args:
        screen: The display, which fixes where the camera sits.
        frames: How many positions to produce.

    Returns:
        Eye positions ``(frames, 3)`` in the camera frame, metres.
    """
    turns = np.linspace(0, 2 * np.pi, frames, endpoint=False)
    viewer = np.stack(
        [
            BENCH_SWEEP_M * np.sin(turns),
            BENCH_SWEEP_M * np.sin(2 * turns),
            np.full(frames, BENCH_DISTANCE_M),
        ],
        axis=1,
    )
    # CAMERA_TO_SCREEN is its own inverse, so it converts either way.
    return np.asarray(screen.camera_to_centre_m) + viewer * CAMERA_TO_SCREEN


def time_render_path(
    screen: ScreenConfig,
    size: tuple[int, int],
    frames: int,
    out_fol: Path,
) -> list[StageTimes]:
    """Time the projection, the wireframe render and the sink at one size.

    The sink timed here is `PngSink`, because it is the sink that exists. It is
    an upper bound rather than the live cost: phase 5's window sink hands a
    frame to a display instead of encoding it to disk, and it cannot be timed
    before it is written.

    Args:
        screen: The display being rendered for.
        size: Output size, as ``(width, height)``.
        frames: How many frames to time, past the warm-up.
        out_fol: Folder the PNGs go into.

    Returns:
        One :class:`StageTimes` per render stage.
    """
    width, height = size
    check_aspect(screen, width, height)
    renderer = WireframeRenderer(window_box(screen))
    sink = PngSink(
        SinkConfig(
            name=f"bench_{width}x{height}",
            out_fol=out_fol / f"{width}x{height}",
            width_px=width,
            height_px=height,
        )
    )
    positions = bench_eye_positions(screen, frames + WARMUP_FRAMES)

    samples: dict[str, list[float]] = {stage: [] for stage in RENDER_STAGES}
    for idx, eye in enumerate(positions):
        start = time.perf_counter()
        matrix = view_projection_matrix(screen, eye)
        projection_s = time.perf_counter() - start

        start = time.perf_counter()
        frame = renderer.render(matrix, width, height)
        render_s = time.perf_counter() - start

        start = time.perf_counter()
        sink.write(frame)
        sink_s = time.perf_counter() - start

        if idx < WARMUP_FRAMES:
            continue
        samples["projection"].append(projection_s * 1000)
        samples["render"].append(render_s * 1000)
        samples["sink"].append(sink_s * 1000)
    sink.close()

    return [
        StageTimes("render", stage, width, height, samples[stage])
        for stage in RENDER_STAGES
    ]


def report(timings: list[StageTimes]) -> None:
    """Log the per-stage table.

    Args:
        timings: Every timed stage.
    """
    lg.info(f"{'path':8} {'stage':13} {'size':11} {'median':>9} {'p95':>9} {'fps':>8}")
    for entry in timings:
        size = f"{entry.width}x{entry.height}"
        lg.info(
            f"{entry.path:8} {entry.stage:13} {size:11} "
            f"{entry.median_ms:8.2f}m {entry.p95_ms:8.2f}m {entry.stage_fps:8.1f}"
        )


def loop_budget_ms(
    timings: list[StageTimes],
    in_size: tuple[int, int],
    out_size: tuple[int, int],
) -> float:
    """Median milliseconds per frame for one capture/output pairing.

    Excludes the sink, for the reason given in :func:`report_budget`.

    Args:
        timings: Every timed stage.
        in_size: Capture size, as ``(width, height)``.
        out_size: Output size, as ``(width, height)``.

    Returns:
        The summed median cost of every stage both loops pay.
    """
    total = 0.0
    for entry in timings:
        if entry.stage == "sink":
            continue
        wanted = in_size if entry.path == "tracker" else out_size
        if (entry.width, entry.height) == wanted:
            total += entry.median_ms
    return total


def report_budget(timings: list[StageTimes]) -> None:
    """Log what each capture/output pairing would cost end to end.

    The point of the benchmark is this table rather than any single stage: a
    stage is only interesting relative to the 33 ms the camera allows.

    **The sink is excluded from the total, and the excluded figure is printed
    beside it.** Only `PngSink` exists to time, and encoding a PNG to disk is
    not what the live window sink will do, so folding it in would report a
    budget for a loop nobody is going to run. What is left - decode, landmark,
    eye position, projection, render - is what both loops pay, so the remaining
    headroom is what the window sink has to fit into.

    Args:
        timings: Every timed stage.
    """
    tracker = [t for t in timings if t.path == "tracker"]
    render = [t for t in timings if t.path == "render"]
    input_sizes = sorted({(t.width, t.height) for t in tracker}, reverse=True)
    output_sizes = sorted({(t.width, t.height) for t in render}, reverse=True)

    lg.info(f"Budget, against {CAMERA_BUDGET_MS:.1f} ms per frame from the camera.")
    lg.info("Sink excluded: it is PngSink, and the live sink is a window.")
    for in_size in input_sizes:
        in_ms = sum(t.median_ms for t in tracker if (t.width, t.height) == in_size)
        for out_size in output_sizes:
            at_out = [t for t in render if (t.width, t.height) == out_size]
            png_ms = sum(t.median_ms for t in at_out if t.stage == "sink")
            total = loop_budget_ms(timings, in_size, out_size)
            out_ms = total - in_ms
            left = CAMERA_BUDGET_MS - total
            lg.info(
                f"  in {in_size[0]}x{in_size[1]} -> out {out_size[0]}x{out_size[1]}: "
                f"{in_ms:6.2f} + {out_ms:6.2f} = {total:6.2f} ms, "
                f"{1000 / total:5.1f} fps, {left:+6.2f} ms left for the sink "
                f"(PngSink would take {png_ms:.2f})"
            )


def write_csv(path: Path, timings: list[StageTimes], machine: str) -> None:
    """Write the timings.

    Args:
        path: Destination CSV.
        timings: Every timed stage.
        machine: Hostname the timings came from.
    """
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(entry.as_row(machine) for entry in timings)
    lg.info(f"Wrote {path}")


def main() -> None:
    """Parse arguments and run the benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", default=DEFAULT_CLIP, help="clip name or path")
    parser.add_argument("--camera", default=DEFAULT_CAMERA, help="camera registry key")
    parser.add_argument("--screen", default=DEFAULT_SCREEN, help="screen registry key")
    parser.add_argument("--input-sizes", default=DEFAULT_INPUT_SIZES)
    parser.add_argument("--output-sizes", default=DEFAULT_OUTPUT_SIZES)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    args = parser.parse_args()

    paths = get_abyss_paths()
    candidate = Path(args.clip)
    clip = candidate if candidate.exists() else paths.pose_fol / args.clip
    if not clip.exists():
        lg.error(f"Missing clip: {clip}")
        return

    out_fol = paths.cache_fol / "benchmark"
    out_fol.mkdir(parents=True, exist_ok=True)
    screen = get_screen(args.screen)
    machine = platform.node()
    lg.info(f"Benchmarking on {machine}, clip {clip.name}, {args.frames} frames")

    timings: list[StageTimes] = []
    for size in parse_sizes(args.input_sizes):
        timings += time_tracker_path(clip, args.camera, size, args.frames)
    for size in parse_sizes(args.output_sizes):
        timings += time_render_path(screen, size, args.frames, out_fol / "frames")

    report(timings)
    report_budget(timings)
    write_csv(out_fol / f"{machine}_stages.csv", timings, machine)


if __name__ == "__main__":
    main()
