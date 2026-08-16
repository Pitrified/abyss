"""Measure a camera's focal length in pixels, from one object of known size.

This is not camera calibration and needs no checkerboard. One object of known
size, held at a tape-measured distance, gives the focal length directly::

    f_px = apparent_size_px * distance_m / real_size_m

A percent or two of tape-measure error is comfortably good enough: the
per-identity head scale that phase 1 corrects is a 16% effect.

Manual script, because it needs a camera with something in front of it. It
never opens a window, so it works over ssh: frames go to PNG files under
``AbyssPaths.cache_fol / "measure"``.

Three steps, in order.

**1. Capture.** Hold a sheet of A4 (297 mm tall, 210 mm wide) flat, facing the
camera square on, at a tape-measured distance from the *lens*::

    uv run --no-sync python scripts/measure_focal.py capture

This writes a clean frame and a second copy with a labelled pixel grid drawn
over it.

**2. Read the edges.** Open the gridded PNG and read off the row numbers of the
top and bottom edges of the sheet. Their difference is the apparent size in
pixels.

**3. Solve.** Feed the numbers back in to get the config snippet::

    uv run --no-sync python scripts/measure_focal.py solve \
        --pixels 412 --distance-m 0.60 --size-mm 297

Why MJPG is forced: on g7's camera the OpenCV default of YUYV caps at 640x480
and silently ignores a request for anything larger, while MJPG reaches
1280x720. Those two modes are different aspect ratios, so they are not
necessarily the same field of view sampled at two densities. Use
``compare-modes`` to settle that before trusting one measurement at the other
resolution.
"""

import argparse
from pathlib import Path

import cv2 as cv
from loguru import logger as lg
import numpy as np

from abyss.params.abyss_params import get_abyss_paths

DEFAULT_DEVICE = 0
"""``/dev/video0``: on g7 the RGB camera, where video2 is the infrared one."""

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
"""The largest mode this camera offers, and only over MJPG."""

WARMUP_FRAMES = 30
"""Frames to discard so auto-exposure and auto-focus settle before keeping one.

15 was not enough: the same scene came back at mean 209 of 255 after 15 frames
and mean 119 after 30, so the first number was auto-exposure still moving.
"""

GRID_MINOR_PX = 20
"""Spacing of the tick marks along the frame border."""

GRID_MAJOR_PX = 100
"""Spacing of the labelled lines drawn across the whole frame."""

GRID_TICK_PX = 8
"""Length of a minor tick, drawn inward from the border."""

GRID_ALPHA = 0.45
"""Opacity of the major lines, so the image stays readable underneath."""

A4_HEIGHT_MM = 297.0
"""Long edge of a sheet of A4, the suggested target."""

CLIP_LEVEL = 250
"""Grey value at and above which a pixel counts as blown out."""

CLIP_WARN_PCT = 15.0
"""How much of the frame may clip before the capture is worth redoing."""

SUGGESTED_EXPOSURE = 80
"""A starting manual exposure, against auto settling near 312 on a lit wall."""

MM_PER_M = 1000.0


class CameraOpenError(RuntimeError):
    """Raised when the capture device cannot be opened."""

    def __init__(self, device: int) -> None:
        """Initialise with the offending device index.

        Args:
            device: The ``/dev/video*`` index that failed to open.
        """
        super().__init__(
            f"Could not open camera {device}. Check it exists and is not in use "
            f"by another process"
        )


class CameraReadError(RuntimeError):
    """Raised when the capture device opens but yields no frame."""

    def __init__(self, device: int) -> None:
        """Initialise with the offending device index.

        Args:
            device: The ``/dev/video*`` index that yielded nothing.
        """
        super().__init__(f"Camera {device} opened but returned no frame")


def grab_frame(
    device: int,
    width: int,
    height: int,
    *,
    fourcc: str = "MJPG",
    exposure: int | None = None,
) -> np.ndarray:
    """Capture one settled frame at a requested mode.

    The format is set before the size on purpose. OpenCV defaults to YUYV,
    whose USB bandwidth ceiling silently clamps the size back down, so asking
    for 1280x720 without asking for MJPG first returns 640x480.

    Args:
        device: ``/dev/video*`` index to open.
        width: Requested frame width in pixels.
        height: Requested frame height in pixels.
        fourcc: Pixel format to request, as a four character code.
        exposure: Manual exposure in units of 100 microseconds. ``None`` leaves
            the camera on auto, which blows out a white target against a lit
            wall. Lower is darker.

    Returns:
        One BGR frame, at whatever size the camera actually granted.

    Raises:
        CameraOpenError: If the device cannot be opened.
        CameraReadError: If the device yields no frame.
    """
    capture = cv.VideoCapture(device, cv.CAP_V4L2)
    if not capture.isOpened():
        capture.release()
        raise CameraOpenError(device)
    try:
        capture.set(cv.CAP_PROP_FOURCC, cv.VideoWriter.fourcc(*fourcc))
        capture.set(cv.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        if exposure is not None:
            # V4L2 exposure_auto: 1 is manual, 3 is aperture priority.
            capture.set(cv.CAP_PROP_AUTO_EXPOSURE, 1)
            capture.set(cv.CAP_PROP_EXPOSURE, exposure)

        frame = None
        for _ in range(WARMUP_FRAMES):
            ok, latest = capture.read()
            if ok:
                frame = latest
        if frame is None:
            raise CameraReadError(device)

        got_h, got_w = frame.shape[:2]
        if (got_w, got_h) != (width, height):
            lg.warning(f"Asked for {width}x{height}, camera granted {got_w}x{got_h}")
    finally:
        capture.release()
    return frame


def draw_grid(frame: np.ndarray) -> np.ndarray:
    """Draw a measuring scale over a copy of a frame.

    The scale is how the edges get read off without a display or a mouse: open
    the PNG, see which labelled line each edge falls on. A full grid at tick
    spacing buries the image, so only the labelled lines cross the frame and
    they are blended rather than painted on. The fine scale lives as ticks
    along the border, where it costs no visibility.

    Args:
        frame: BGR frame to annotate.

    Returns:
        An annotated copy. The input is not modified.
    """
    height, width = frame.shape[:2]
    major = (0, 255, 255)
    tick = (0, 255, 0)

    # Blend the full-width lines so the scene stays legible through them.
    lines = frame.copy()
    for y in range(0, height, GRID_MAJOR_PX):
        cv.line(lines, (0, y), (width, y), major, 1)
    for x in range(0, width, GRID_MAJOR_PX):
        cv.line(lines, (x, 0), (x, height), major, 1)
    out = cv.addWeighted(lines, GRID_ALPHA, frame, 1 - GRID_ALPHA, 0)

    # Ticks and labels are drawn at full strength: they sit on the border.
    for y in range(0, height, GRID_MINOR_PX):
        cv.line(out, (0, y), (GRID_TICK_PX, y), tick, 1)
        cv.line(out, (width - GRID_TICK_PX, y), (width, y), tick, 1)
    for x in range(0, width, GRID_MINOR_PX):
        cv.line(out, (x, 0), (x, GRID_TICK_PX), tick, 1)
        cv.line(out, (x, height - GRID_TICK_PX), (x, height), tick, 1)

    for y in range(0, height, GRID_MAJOR_PX):
        label(out, f"y{y}", (GRID_TICK_PX + 3, max(y - 4, 12)), major)
    for x in range(0, width, GRID_MAJOR_PX):
        label(out, f"x{x}", (x + 3, height - GRID_TICK_PX - 5), major)

    return out


def label(frame: np.ndarray, text: str, origin: tuple[int, int], colour: tuple) -> None:
    """Draw text with a dark outline, so it survives a blown-out background.

    Args:
        frame: BGR frame to draw on, modified in place.
        text: The string to draw.
        origin: Bottom left corner of the text, in pixels.
        colour: BGR colour of the text itself.
    """
    cv.putText(frame, text, origin, cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
    cv.putText(frame, text, origin, cv.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)


def focal_px(pixels: float, distance_m: float, size_mm: float) -> float:
    """Solve the similar-triangles relation for the focal length.

    Args:
        pixels: Apparent size of the target in pixels.
        distance_m: Distance from the lens to the target in metres.
        size_mm: True size of the target in millimetres, along the same axis.

    Returns:
        Focal length in pixels, valid at the height the frame was captured at.
    """
    return pixels * distance_m / (size_mm / MM_PER_M)


def cmd_capture(args: argparse.Namespace, out_fol: Path) -> None:
    """Capture one frame and write it clean and gridded.

    Args:
        args: Parsed command line arguments.
        out_fol: Folder to write the PNGs into.
    """
    frame = grab_frame(args.device, args.width, args.height, exposure=args.exposure)
    height, width = frame.shape[:2]

    clean = out_fol / f"target_{width}x{height}.png"
    gridded = out_fol / f"target_{width}x{height}_grid.png"
    cv.imwrite(str(clean), frame)
    cv.imwrite(str(gridded), draw_grid(frame))

    lg.info(f"Wrote {clean}")
    lg.info(f"Wrote {gridded}")

    # A clipped frame has no edge to measure: white paper on a blown wall is
    # one flat region, so say so rather than let it reach the ruler step.
    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    clipped = float((grey >= CLIP_LEVEL).mean() * 100)
    if clipped > CLIP_WARN_PCT:
        lg.warning(
            f"{clipped:.0f}% of the frame is clipped white. Re-run with "
            f"--exposure (try {SUGGESTED_EXPOSURE}) so the target keeps an edge"
        )

    print(f"\nCaptured {width}x{height}. Open the gridded PNG:\n  {gridded}")
    print("Read the top and bottom edge rows of the target, then run:\n")
    print(
        f"  uv run --no-sync python scripts/measure_focal.py solve \\\n"
        f"      --pixels <bottom - top> --distance-m <measured> "
        f"--size-mm {A4_HEIGHT_MM:.0f} --at-height {height}"
    )


def cmd_compare_modes(args: argparse.Namespace, out_fol: Path) -> None:
    """Capture the same scene in both modes, to settle how they crop.

    A focal length measured at one resolution only transfers to another if the
    two share a vertical field of view. Point the camera at a fixed scene and
    compare the two PNGs: if the 4:3 frame shows strictly more content left and
    right and the same top to bottom, the modes share a vertical field of view
    and ``focal_px_for_height`` is right to rescale by height alone. If the 4:3
    frame reaches higher and lower than the 16:9 one, they do not, and each
    mode needs its own measurement.

    Args:
        args: Parsed command line arguments.
        out_fol: Folder to write the PNGs into.
    """
    for width, height in ((DEFAULT_WIDTH, DEFAULT_HEIGHT), (640, 480)):
        frame = grab_frame(args.device, width, height)
        got_h, got_w = frame.shape[:2]
        path = out_fol / f"modes_{got_w}x{got_h}.png"
        cv.imwrite(str(path), frame)
        lg.info(f"Wrote {path}")

    print(f"\nWrote both modes to {out_fol}.")
    print("Compare them against a fixed scene: does the 4:3 frame reach higher")
    print("and lower than the 16:9 one, or only wider?")


def cmd_solve(args: argparse.Namespace, out_fol: Path) -> None:
    """Turn the measurement into a focal length and a config snippet.

    Args:
        args: Parsed command line arguments.
        out_fol: Unused, present so every subcommand has one signature.
    """
    del out_fol
    focal = focal_px(args.pixels, args.distance_m, args.size_mm)
    fov_deg = 2 * np.degrees(np.arctan((args.at_height / 2) / focal))

    print(f"\nfocal_px = {focal:.1f} px at {args.at_height} px tall")
    print(f"implied vertical field of view = {fov_deg:.1f} deg")
    print("\nRegistry entry:\n")
    print(f'    "{args.camera}": CameraConfig(')
    print(f'        name="{args.camera}",')
    print(f"        focal_px={focal:.1f},")
    print(f"        focal_measured_at_height={args.at_height},")
    print('        provenance="<target, distance, date>",')
    print("    ),")


def main() -> None:
    """Parse arguments and dispatch to a subcommand."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", type=int, default=DEFAULT_DEVICE, help="/dev/video* index"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    capture = sub.add_parser("capture", help="grab one frame, clean and gridded")
    capture.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    capture.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    capture.add_argument(
        "--exposure",
        type=int,
        default=None,
        help=(
            "manual exposure in 100 us units, lower is darker. Default is the "
            f"camera's auto, which clips a white target. Try {SUGGESTED_EXPOSURE}"
        ),
    )
    capture.set_defaults(func=cmd_capture)

    compare = sub.add_parser(
        "compare-modes", help="capture 16:9 and 4:3 to see how they crop"
    )
    compare.set_defaults(func=cmd_compare_modes)

    solve = sub.add_parser("solve", help="focal length from the measurement")
    solve.add_argument(
        "--pixels", type=float, required=True, help="apparent target size in pixels"
    )
    solve.add_argument(
        "--distance-m", type=float, required=True, help="lens to target, in metres"
    )
    solve.add_argument(
        "--size-mm",
        type=float,
        default=A4_HEIGHT_MM,
        help="true target size in millimetres, same axis as --pixels",
    )
    solve.add_argument(
        "--at-height",
        type=int,
        default=DEFAULT_HEIGHT,
        help="frame height the measurement was made at",
    )
    solve.add_argument("--camera", default="g7_webcam", help="registry key to print")
    solve.set_defaults(func=cmd_solve)

    args = parser.parse_args()

    out_fol = get_abyss_paths().cache_fol / "measure"
    out_fol.mkdir(parents=True, exist_ok=True)
    args.func(args, out_fol)


if __name__ == "__main__":
    main()
