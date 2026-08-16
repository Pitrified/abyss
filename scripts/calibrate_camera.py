"""Calibrate a camera from a ChArUco board shown on a screen.

Better than the one-object-at-a-known-distance method in
``scripts/measure_focal.py``, and it needs no tape measure at all.

A single head-on view of a known-size target cannot give a focal length: ``f``
and ``Z`` only ever appear as ``f / Z``, which is why the A4 method has to
measure the distance. Several views at **different orientations** break that
degeneracy, because each homography constrains the intrinsics through the
orthonormality of the rotation columns. Distance comes out as a result, never
as an input. This is Zhang's method, i.e. :func:`cv2.calibrateCamera`.

Two consequences worth knowing before trusting the numbers.

- The focal length does not depend on the board's physical size. Scaling the
  object points scales the recovered translations and leaves the intrinsics
  alone. A wrong millimetres-per-pixel gives correct intrinsics and wrong
  distances. Verified in ``tests/scripts/test_calibrate_camera.py``.
- Tilt is required, not tolerated. Views that are all fronto-parallel are the
  degenerate case again, however many you take. Roll, pitch and yaw the board
  between shots, and fill different parts of the frame.

Why a screen rather than paper: the physical square size follows from the
panel's pixel pitch exactly, with no ruler, and a screen is rigid and flat
where taped paper bows. E-ink is the better of the two panels here, being matte
where a phone is glossy, and specular glare is the main way this fails.

Three steps.

**1. Emit the board** at the panel's native resolution, so that opening it in
any "fit to screen" viewer is the identity transform::

    uv run --no-sync python scripts/calibrate_camera.py board \
        --device-preset kindle_pw11

Copy the PNG to the device and open it full screen. The JSON written beside it
records the board geometry, and ``solve`` reads it so the two cannot disagree.

**2. Capture views**, moving and tilting the board between each::

    uv run --no-sync python scripts/calibrate_camera.py capture --views 15

**3. Solve**::

    uv run --no-sync python scripts/calibrate_camera.py solve
"""

import argparse
from dataclasses import asdict
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any
from typing import cast

import cv2 as cv
from loguru import logger as lg
import numpy as np
from PIL import Image

from abyss.params.abyss_params import get_abyss_paths

MM_PER_INCH = 25.4

PT_PER_INCH = 72.0
"""PDF points per inch, for reporting the page size."""

GREYSCALE_NDIM = 2
"""Number of axes a single channel image has."""

DICTIONARY_NAME = "DICT_4X4_50"
"""Small dictionary: fewer bits per marker detects more reliably."""

MARKER_RATIO = 0.75
"""Marker side as a fraction of the square side, the usual ChArUco ratio."""

DEFAULT_SQUARES_X = 7
DEFAULT_SQUARES_Y = 9
"""A portrait board. 7x9 squares gives 48 interior corners per full view."""

WARMUP_FRAMES = 30
"""Frames discarded so auto-exposure settles, as in ``measure_focal.py``."""

MIN_VIEWS = 3
"""Below this the problem is underdetermined and calibration is meaningless."""

ADVISED_VIEWS = 8
"""Below this the result is fitted but not trustworthy, so warn."""

MIN_CORNERS_PER_VIEW = 8
"""Views with fewer detected corners than this contribute noise, so drop them."""

RMS_WARN_PX = 1.0
"""Reprojection error above which the capture is worth redoing."""


@dataclass(frozen=True)
class DevicePreset:
    """A panel to display the board on.

    Args:
        width_px: Native panel width in pixels.
        height_px: Native panel height in pixels.
        ppi: Pixel density, which fixes the physical size of a rendered square.
        note: What to watch out for on this device.
    """

    width_px: int
    height_px: int
    ppi: float
    note: str


DEVICE_PRESETS: dict[str, DevicePreset] = {
    "kindle_pw11": DevicePreset(
        width_px=1236,
        height_px=1648,
        ppi=300.0,
        note="e-ink, matte, no glare and no backlight flicker. Preferred",
    ),
    "pixel7pro": DevicePreset(
        width_px=1440,
        height_px=3120,
        ppi=512.0,
        note="glossy OLED, watch for specular glare from room lights",
    ),
}
"""Panels available here. Check the implied diagonal that ``board`` prints."""


class UnknownPresetError(KeyError):
    """Raised when a device preset name is not registered."""

    def __init__(self, name: str, known: list[str]) -> None:
        """Initialise with the offending name.

        Args:
            name: The preset that was asked for.
            known: The presets that exist.
        """
        super().__init__(f"Unknown preset {name!r}, known: {', '.join(sorted(known))}")


class BoardSpecMissingError(FileNotFoundError):
    """Raised when solving without the board description the board step wrote."""

    def __init__(self, path: Path) -> None:
        """Initialise with the path that was expected.

        Args:
            path: Where the spec should have been.
        """
        super().__init__(
            f"No board spec at {path}. Run the 'board' step first: solving needs "
            f"the geometry of the board that was actually displayed"
        )


class TooFewViewsError(ValueError):
    """Raised when there are not enough usable views to calibrate."""

    def __init__(self, usable: int) -> None:
        """Initialise with how many views survived detection.

        Args:
            usable: Number of views with enough detected corners.
        """
        super().__init__(
            f"Only {usable} usable views, need at least {MIN_VIEWS}. Check the "
            f"board fills a good part of the frame and is not glared out"
        )


class CameraOpenError(RuntimeError):
    """Raised when the capture device cannot be opened."""

    def __init__(self, device: int) -> None:
        """Initialise with the offending device index.

        Args:
            device: The ``/dev/video*`` index that failed to open.
        """
        super().__init__(f"Could not open camera {device}")


@dataclass(frozen=True)
class BoardSpec:
    """The board that was displayed, in the units calibration needs.

    Written next to the board image and read back by ``solve``, so the geometry
    used to calibrate is by construction the geometry that was shown.

    Args:
        squares_x: Number of squares across.
        squares_y: Number of squares down.
        square_m: Physical side of one square in metres.
        marker_m: Physical side of one marker in metres.
        dictionary: Name of the predefined aruco dictionary.
        square_px: Side of one square as rendered, in device pixels.
        ppi: Pixel density the physical size was derived from.
        source: Which panel this was generated for.
    """

    squares_x: int
    squares_y: int
    square_m: float
    marker_m: float
    dictionary: str
    square_px: int
    ppi: float
    source: str


def write_pdf(canvas: np.ndarray, path: Path, ppi: float) -> None:
    """Write the board as a single page PDF, for readers that reject images.

    A Kindle will not open a raw PNG, so the board has to travel as a PDF. The
    page is sized ``pixels / ppi`` inches, which makes its aspect ratio exactly
    the panel's. That is the property worth protecting: a reader that scales
    the page uniformly changes only the recovered distances, and leaves the
    focal length alone, but one that stretched the page would corrupt the
    intrinsics themselves.

    Args:
        canvas: The rendered board, as a greyscale array.
        path: Destination PDF path.
        ppi: Pixel density, which sets the page size in inches.
    """
    Image.fromarray(canvas).convert("L").save(path, resolution=ppi)


def build_board(spec: BoardSpec) -> cv.aruco.CharucoBoard:
    """Rebuild the OpenCV board object from a spec.

    Args:
        spec: The board description.

    Returns:
        The board, carrying the metric geometry.
    """
    dictionary = cv.aruco.getPredefinedDictionary(
        getattr(cv.aruco, spec.dictionary),
    )
    return cv.aruco.CharucoBoard(
        (spec.squares_x, spec.squares_y),
        spec.square_m,
        spec.marker_m,
        dictionary,
    )


def detect_corners(
    image: np.ndarray,
    board: cv.aruco.CharucoBoard,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Find the board in one image and pair it with its object points.

    Args:
        image: Grayscale or BGR image to search.
        board: The board being looked for.

    Returns:
        Object points and image points, or ``None`` if too little was found.
    """
    grey = (
        image
        if image.ndim == GREYSCALE_NDIM
        else cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    )
    detector = cv.aruco.CharucoDetector(board)
    corners, ids, _, _ = detector.detectBoard(grey)
    if ids is None or len(ids) < MIN_CORNERS_PER_VIEW:
        return None
    # The stub types detectedCorners as a sequence of matrices, but detectBoard
    # hands back one (N, 2) array and matchImagePoints accepts it. Verified in
    # tests/scripts/test_calibrate_camera.py rather than trusted.
    object_points, image_points = board.matchImagePoints(cast("Any", corners), ids)
    if object_points is None or len(object_points) < MIN_CORNERS_PER_VIEW:
        return None
    return object_points, image_points


def cmd_board(args: argparse.Namespace, out_fol: Path) -> None:
    """Render the board at a panel's native resolution and write its spec.

    The square size is the largest whole number of pixels that fits the panel,
    so every square is an identical integer number of device pixels and the
    physical size follows from the pixel pitch exactly.

    Args:
        args: Parsed command line arguments.
        out_fol: Folder to write the PNG and JSON into.

    Raises:
        UnknownPresetError: If the named preset is not registered.
    """
    if args.device_preset not in DEVICE_PRESETS:
        raise UnknownPresetError(args.device_preset, list(DEVICE_PRESETS))
    preset = DEVICE_PRESETS[args.device_preset]

    square_px = min(
        preset.width_px // args.squares_x,
        preset.height_px // args.squares_y,
    )
    square_m = square_px / preset.ppi * MM_PER_INCH / 1000
    spec = BoardSpec(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_m=square_m,
        marker_m=square_m * MARKER_RATIO,
        dictionary=DICTIONARY_NAME,
        square_px=square_px,
        ppi=preset.ppi,
        source=args.device_preset,
    )

    board = build_board(spec)
    drawn = board.generateImage(
        (square_px * args.squares_x, square_px * args.squares_y)
    )

    # Centre the board on a white canvas of exactly the panel size, so a viewer
    # that fits the image to the screen changes nothing.
    canvas = np.full((preset.height_px, preset.width_px), 255, dtype=np.uint8)
    top = (preset.height_px - drawn.shape[0]) // 2
    left = (preset.width_px - drawn.shape[1]) // 2
    canvas[top : top + drawn.shape[0], left : left + drawn.shape[1]] = drawn

    png = out_fol / f"board_{args.device_preset}.png"
    pdf = out_fol / f"board_{args.device_preset}.pdf"
    spec_path = out_fol / "board_spec.json"
    cv.imwrite(str(png), canvas)
    write_pdf(canvas, pdf, preset.ppi)
    spec_path.write_text(json.dumps(asdict(spec), indent=2) + "\n")
    lg.info(f"Wrote {png}")
    lg.info(f"Wrote {pdf}")
    lg.info(f"Wrote {spec_path}")

    # The advertised diagonal is the one spec that is easy to check by eye, so
    # print it: if this disagrees with the box, the ppi is wrong.
    diagonal_in = float(np.hypot(preset.width_px, preset.height_px)) / preset.ppi
    print(f"\n{args.device_preset}: {preset.note}")
    print(f"  panel      {preset.width_px}x{preset.height_px} px at {preset.ppi:g} ppi")
    print(f"  implies    {diagonal_in:.2f} in diagonal, check against the spec sheet")
    print(f"  square     {square_px} px = {square_m * 1000:.3f} mm")
    print(f"  board      {args.squares_x}x{args.squares_y} squares")
    page_w = preset.width_px / preset.ppi * PT_PER_INCH
    page_h = preset.height_px / preset.ppi * PT_PER_INCH
    print(f"  pdf page   {page_w:.2f} x {page_h:.2f} pt, same aspect as the panel")
    print(f"\nPNG for a phone: {png.name}")
    print(f"PDF for a Kindle, which will not open a raw PNG: {pdf.name}")
    print("Open it full screen. Aspect is what matters, not scale: see")
    print("docs/library/camera_calibration.md if the reader adds margins.")


def cmd_capture(args: argparse.Namespace, out_fol: Path) -> None:
    """Grab several views of the board, reporting what was detected in each.

    Args:
        args: Parsed command line arguments.
        out_fol: Folder holding the board spec, and where views are written.

    Raises:
        CameraOpenError: If the device cannot be opened.
    """
    spec = load_spec(out_fol)
    board = build_board(spec)

    views_fol = out_fol / f"views_{args.width}x{args.height}"
    views_fol.mkdir(parents=True, exist_ok=True)
    for stale in views_fol.glob("view_*.png"):
        stale.unlink()

    capture = cv.VideoCapture(args.device, cv.CAP_V4L2)
    if not capture.isOpened():
        capture.release()
        raise CameraOpenError(args.device)
    try:
        capture.set(cv.CAP_PROP_FOURCC, cv.VideoWriter.fourcc(*"MJPG"))
        capture.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
        if args.exposure is not None:
            capture.set(cv.CAP_PROP_AUTO_EXPOSURE, 1)
            capture.set(cv.CAP_PROP_EXPOSURE, args.exposure)
        for _ in range(WARMUP_FRAMES):
            capture.read()

        print(f"\nTilt and move the board between shots. {args.delay}s per view.")
        print("Vary roll, pitch and yaw: views that are all flat-on cannot solve.\n")
        kept = 0
        for view in range(args.views):
            for remaining in range(args.delay, 0, -1):
                print(f"  view {view + 1}/{args.views} in {remaining}...", end="\r")
                time.sleep(1)
            ok, frame = capture.read()
            if not ok:
                lg.warning(f"view {view + 1}: no frame")
                continue
            found = detect_corners(frame, board)
            count = 0 if found is None else len(found[0])
            if found is None:
                print(f"  view {view + 1}/{args.views}: no board found, skipped ")
                continue
            path = views_fol / f"view_{view:02d}.png"
            cv.imwrite(str(path), frame)
            kept += 1
            print(f"  view {view + 1}/{args.views}: {count} corners, saved       ")
    finally:
        capture.release()

    print(f"\nKept {kept} views in {views_fol}")
    if kept < ADVISED_VIEWS:
        lg.warning(f"{kept} views is thin, {ADVISED_VIEWS} or more is advised")


def load_spec(out_fol: Path) -> BoardSpec:
    """Read the board spec written by the board step.

    Args:
        out_fol: Folder the spec lives in.

    Returns:
        The board description.

    Raises:
        BoardSpecMissingError: If the spec has not been written.
    """
    spec_path = out_fol / "board_spec.json"
    if not spec_path.exists():
        raise BoardSpecMissingError(spec_path)
    return BoardSpec(**json.loads(spec_path.read_text()))


def cmd_solve(args: argparse.Namespace, out_fol: Path) -> None:
    """Calibrate from the saved views and print the registry snippet.

    Args:
        args: Parsed command line arguments.
        out_fol: Folder holding the board spec and the views.

    Raises:
        TooFewViewsError: If too few views survive detection.
    """
    spec = load_spec(out_fol)
    board = build_board(spec)

    views_fol = out_fol / f"views_{args.width}x{args.height}"
    paths = sorted(views_fol.glob("view_*.png"))
    object_points, image_points = [], []
    for path in paths:
        image = cv.imread(str(path))
        if image is None:
            lg.warning(f"{path.name}: unreadable, skipped")
            continue
        found = detect_corners(image, board)
        if found is None:
            lg.warning(f"{path.name}: no board, skipped")
            continue
        object_points.append(found[0])
        image_points.append(found[1])

    if len(object_points) < MIN_VIEWS:
        raise TooFewViewsError(len(object_points))
    if len(object_points) < ADVISED_VIEWS:
        lg.warning(f"Only {len(object_points)} views, {ADVISED_VIEWS} or more advised")

    size = (args.width, args.height)
    rms, matrix, distortion, _, translations = cv.calibrateCamera(
        object_points, image_points, size, None, None
    )
    fx, fy = matrix[0, 0], matrix[1, 1]
    cx, cy = matrix[0, 2], matrix[1, 2]
    depths = [float(np.linalg.norm(t)) for t in translations]

    print(f"\n{len(object_points)} views at {args.width}x{args.height}")
    print(f"  rms reprojection  {rms:.4f} px")
    print(f"  fx, fy            {fx:.2f}, {fy:.2f}   (ratio {fx / fy:.4f})")
    print(f"  cx, cy            {cx:.2f}, {cy:.2f}")
    print(f"  frame centre      {args.width / 2:.2f}, {args.height / 2:.2f}")
    print(f"  distortion        {np.round(distortion.ravel(), 4).tolist()}")
    print(f"  board distance    {min(depths):.3f} to {max(depths):.3f} m")
    if rms > RMS_WARN_PX:
        lg.warning(f"rms {rms:.2f} px is high, consider recapturing")

    # The model is vertical-FOV based and stores one focal, so fy is the one it
    # wants. fx is printed above as a consistency check, not consumed.
    fov_deg = 2 * np.degrees(np.arctan((args.height / 2) / fy))
    print(f"\nimplied vertical field of view {fov_deg:.2f} deg\n")
    print(f'    "{args.camera}": CameraConfig(')
    print(f'        name="{args.camera}",')
    print(f"        focal_px={fy:.1f},")
    print(f"        focal_measured_at_height={args.height},")
    print(
        f'        provenance="ChArUco on {spec.source}, {len(object_points)} views, '
        f'rms {rms:.3f} px, <date>",'
    )
    print("    ),")


def main() -> None:
    """Parse arguments and dispatch to a subcommand."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    board = sub.add_parser("board", help="render the board for a panel")
    board.add_argument("--device-preset", default="kindle_pw11")
    board.add_argument("--squares-x", type=int, default=DEFAULT_SQUARES_X)
    board.add_argument("--squares-y", type=int, default=DEFAULT_SQUARES_Y)
    board.set_defaults(func=cmd_board)

    capture = sub.add_parser("capture", help="grab several views of the board")
    capture.add_argument("--device", type=int, default=0)
    capture.add_argument("--width", type=int, default=1280)
    capture.add_argument("--height", type=int, default=720)
    capture.add_argument("--views", type=int, default=15)
    capture.add_argument("--delay", type=int, default=4, help="seconds between shots")
    capture.add_argument("--exposure", type=int, default=None)
    capture.set_defaults(func=cmd_capture)

    solve = sub.add_parser("solve", help="calibrate from the saved views")
    solve.add_argument("--width", type=int, default=1280)
    solve.add_argument("--height", type=int, default=720)
    solve.add_argument("--camera", default="g7_webcam")
    solve.set_defaults(func=cmd_solve)

    args = parser.parse_args()
    out_fol = get_abyss_paths().cache_fol / "calib"
    out_fol.mkdir(parents=True, exist_ok=True)
    args.func(args, out_fol)


if __name__ == "__main__":
    main()
