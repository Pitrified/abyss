"""Close the loop: camera to window, or clip to PNGs.

The manual entry point for phase 5, and the only piece that needs a display.

    uv run --no-sync python scripts/live.py camera
    uv run --no-sync python scripts/live.py camera --viewer-ipd-mm 64
    uv run --no-sync python scripts/live.py clip face01.mp4

**Both modes run the same loop**, which is the point rather than a convenience.
`clip` needs no camera and no display and writes PNGs, so the wiring can be
checked on any machine and the live path is not the only way to find out that
something broke. If a change makes `camera` work and `clip` stop, the change is
wrong.

Keys, in the window: ``q`` or escape to quit, ``r`` to re-run the head scale
bootstrap when the wrong person is sitting there.

The manual check this exists for is in
``docs/guides/phase5_live_runbook.md``: sit at a tape-measured distance and see
whether the reported depth agrees.
"""

import argparse
from collections.abc import Iterator
from pathlib import Path

import cv2 as cv
from loguru import logger as lg
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode as VisionRunningMode,
)
import numpy as np
from pose_tools.landmark.face import FaceLandmarkerFrame
from pose_tools.landmark.model_manager import ModelManager

from abyss.config.camera import FrameGeometry
from abyss.config.sink import SinkConfig
from abyss.config.viewer import DEFAULT_IPD_M
from abyss.config.viewer import ViewerConfig
from abyss.loop import run_loop
from abyss.loop import track_with_landmarker
from abyss.params.abyss_devices import get_camera
from abyss.params.abyss_devices import get_screen
from abyss.params.abyss_params import get_abyss_paths
from abyss.render.renderer import WireframeRenderer
from abyss.render.scene import window_box
from abyss.sink.file import PngSink
from abyss.sink.window import WindowSink
from abyss.video.capture import CameraStream
from abyss.video.capture import open_camera

DEFAULT_CAMERA = "g7_webcam"
DEFAULT_CLIP_CAMERA = "unknown_clip"
DEFAULT_SCREEN = "g7_internal"
DEFAULT_DEVICE = 0

CAPTURE_SIZE = (1280, 720)
"""Pinned, not requested.

MJPG 1280x720 is the mode `g7_webcam`'s focal length was measured at, and the
YUYV default silently clamps to 640x480 while reporting success. `open_camera`
verifies what it got rather than trusting the request.
"""

RENDER_SIZE = (1920, 1080)
"""The panel's native resolution.

The render must cover exactly the rectangle `ScreenConfig` describes, so this
should be the desktop resolution of the machine the window opens on. Rendering
native costs 2.5 ms against 1.3 at 720, measured, which is not worth the
softness of letting OpenCV upscale.
"""

LANDMARKER_KWARGS = {
    "running_mode": VisionRunningMode.VIDEO,
    "num_faces": 1,
    "output_facial_transformation_matrixes": True,
}
"""`VIDEO` rather than `LIVE_STREAM`, settled by measurement in Q24.

Inference is 11.6 ms inside a 33 ms budget, so decoupling capture from inference
would buy nothing and would stop the loop being a loop.
"""


def clip_frames(path: Path) -> tuple[Iterator[np.ndarray], tuple[int, int]]:
    """Stream a clip, one frame at a time.

    **It used to read the whole clip into memory, and that was a real cost
    rather than a style point.** 250 frames of 1920x1080 is 1.55 GB, enough to
    push an otherwise idle desktop into swap on a 32 GB machine and leave it
    there - Linux does not page anything back in until it is touched, so the
    lag outlives the run by a long way. Measured on g7 on 2026-09-01: swap
    ended 2032 MB of 2047 used with 25 GB of RAM free.

    Streaming holds one frame instead, and costs nothing: the loop takes an
    iterable and never looks back at a frame it has finished with.

    The first frame is read eagerly, because the caller needs the frame size to
    build the geometry before the loop starts.

    Args:
        path: The clip to read.

    Returns:
        The frames, and the ``(width, height)`` they turned out to be. The size
        is ``(0, 0)`` when the clip has no frames.
    """
    capture = cv.VideoCapture(str(path))
    ok, first = capture.read()
    if not ok:
        capture.release()
        lg.error(f"No frames in {path}")
        return iter(()), (0, 0)

    height, width = first.shape[:2]
    lg.info(f"Streaming {width}x{height} frames from {path.name}")

    def stream() -> Iterator[np.ndarray]:
        frame = first
        try:
            while True:
                yield frame
                ok, frame = capture.read()
                if not ok:
                    return
        finally:
            capture.release()

    return stream(), (width, height)


def camera_frames(stream: CameraStream) -> Iterator[np.ndarray]:
    """Yield frames from a camera until something stops it.

    Unbounded on purpose: the run ends on the quit key, which the loop checks,
    not on the source running out.

    Args:
        stream: The open camera.

    Yields:
        BGR frames.
    """
    while True:
        yield stream.read()


def run_camera(args: argparse.Namespace) -> None:
    """Run live: camera in, fullscreen window out.

    Args:
        args: Parsed command line.
    """
    screen = get_screen(args.screen)
    viewer = ViewerConfig(name="live", ipd_m=args.viewer_ipd_mm / 1000)
    geometry = FrameGeometry(
        camera=get_camera(args.camera),
        width=CAPTURE_SIZE[0],
        height=CAPTURE_SIZE[1],
    )
    renderer = WireframeRenderer(window_box(screen))
    model_path = ModelManager().ensure_model("face_landmarker")

    capture = open_camera(args.device, CAPTURE_SIZE)
    window = WindowSink((args.width, args.height), name="abyss")
    with (
        CameraStream(capture, CAPTURE_SIZE, name=args.camera) as stream,
        FaceLandmarkerFrame(model_path, LANDMARKER_KWARGS) as landmarker,
    ):
        try:
            run_loop(
                camera_frames(stream),
                window,
                screen,
                renderer,
                track_with_landmarker(landmarker, geometry),
                geometry,
                viewer,
                stop=lambda: window.quit_requested,
                reset=window.take_reset_request,
            )
        finally:
            window.close()


def run_clip(args: argparse.Namespace) -> None:
    """Run offline: clip in, numbered PNGs out.

    The same loop, with no camera and no display, which is what makes the live
    wiring checkable anywhere.

    Args:
        args: Parsed command line.
    """
    paths = get_abyss_paths()
    candidate = Path(args.clip)
    clip = candidate if candidate.exists() else paths.pose_fol / args.clip
    if not clip.exists():
        lg.error(f"Missing clip: {clip}")
        return

    frames, size = clip_frames(clip)
    if size == (0, 0):
        return

    screen = get_screen(args.screen)
    viewer = ViewerConfig(name="live", ipd_m=args.viewer_ipd_mm / 1000)
    geometry = FrameGeometry(
        camera=get_camera(args.camera), width=size[0], height=size[1]
    )
    renderer = WireframeRenderer(window_box(screen))
    model_path = ModelManager().ensure_model("face_landmarker")

    out_fol = paths.cache_fol / "live" / clip.stem
    sink = PngSink(
        SinkConfig(
            name=clip.stem,
            out_fol=out_fol,
            width_px=args.width,
            height_px=args.height,
        )
    )
    with FaceLandmarkerFrame(model_path, LANDMARKER_KWARGS) as landmarker:
        run_loop(
            frames,
            sink,
            screen,
            renderer,
            track_with_landmarker(landmarker, geometry),
            geometry,
            viewer,
        )
    sink.close()


def build_parser() -> argparse.ArgumentParser:
    """Build the command line.

    The shared options hang off **each subcommand** rather than off the top
    level, through a parent parser. Defined at the top level they parse only
    *before* the subcommand, so ``live.py camera --viewer-ipd-mm 60`` is
    rejected while ``live.py --viewer-ipd-mm 60 camera`` works - which is the
    opposite of how everyone types it, and the opposite of what this script's
    own docstring and the runbook said.

    Returns:
        The parser.
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--screen", default=DEFAULT_SCREEN, help="screen registry key")
    common.add_argument("--width", type=int, default=RENDER_SIZE[0])
    common.add_argument("--height", type=int, default=RENDER_SIZE[1])
    common.add_argument(
        "--viewer-ipd-mm",
        type=float,
        default=DEFAULT_IPD_M * 1000,
        help="the viewer's interpupillary distance, which sets the depth scale",
    )

    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)

    camera = modes.add_parser(
        "camera", parents=[common], help="live: camera to a fullscreen window"
    )
    camera.add_argument("--device", type=int, default=DEFAULT_DEVICE)
    camera.add_argument("--camera", default=DEFAULT_CAMERA, help="camera registry key")

    clip = modes.add_parser(
        "clip", parents=[common], help="offline: a clip to numbered PNGs"
    )
    clip.add_argument("clip", help="clip name inside the pose folder, or a path")
    clip.add_argument("--camera", default=DEFAULT_CLIP_CAMERA)
    return parser


def main() -> None:
    """Parse arguments and run."""
    args = build_parser().parse_args()
    if args.mode == "camera":
        run_camera(args)
    else:
        run_clip(args)


if __name__ == "__main__":
    main()
