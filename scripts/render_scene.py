"""Render the minimal scene through the window, offline.

Two modes. The sweep needs nothing but the repo, so it runs on a box with no
camera and no clips, and it is the artefact to look at first:

    uv run --no-sync python scripts/render_scene.py sweep
    uv run --no-sync python scripts/render_scene.py sweep --frames 120

The track mode drives the same renderer from a real eye track, the CSV phase 1
writes:

    uv run --no-sync python scripts/render_scene.py track cache/viewer/face01_eye.csv
    uv run --no-sync python scripts/render_scene.py track <csv> --raw

Each run writes numbered PNGs, an mp4 and a contact sheet under
``AbyssPaths.cache_fol / "render"``. The sweep also writes the projected
coordinates it produced, which is the regression baseline.

Note the depths in a real track are only as good as the clip's camera, and the
sample clips have none: they fall back to MediaPipe's assumed field of view. The
render is correct for the eye positions it is given, and those positions are on
an unverified scale. The frame says so.
"""

import argparse
import csv
from pathlib import Path

import cv2 as cv
from loguru import logger as lg
import numpy as np

from abyss.config.screen import ScreenConfig
from abyss.config.sink import SinkConfig
from abyss.params.abyss_devices import get_screen
from abyss.params.abyss_params import get_abyss_paths
from abyss.render.frustum import CAMERA_TO_SCREEN
from abyss.render.frustum import EyeBehindScreenError
from abyss.render.frustum import project_points
from abyss.render.frustum import view_projection_matrix
from abyss.render.renderer import WireframeRenderer
from abyss.render.renderer import render_frame
from abyss.render.scene import window_box
from abyss.sink.file import PngSink
from abyss.sink.file import VideoSink

DEFAULT_SCREEN = "g7_internal"
DEFAULT_SIZE = (1280, 720)
DEFAULT_FRAMES = 90
DEFAULT_FPS = 25.0

SWEEP_RIGHT_M = 0.12
"""How far the sweep moves to the viewer's right and left.

Bounded by what keeps the back wall in frame. The wall is the same size as the
mouth and sits 0.6 m behind it, so it leaves the image once the eye is offset by
more than about ``half_width * distance / depth``. Past that the parallax is
still correct but there is nothing left for the cube to slide against.
"""

SWEEP_UP_M = 0.05
"""How far the sweep moves up and down, at twice the horizontal rate."""

SWEEP_DISTANCE_M = 0.55
"""Mean eye distance from the panel."""

SWEEP_DISTANCE_SWING_M = 0.15
"""How much nearer and further the sweep goes."""

CONTACT_TILES = 3
"""Contact sheet is this many tiles on a side."""

TEXT_BGR = (200, 200, 200)
HELD_BGR = (60, 200, 255)
TEXT_ORIGIN = (12, 26)
HELD_ORIGIN = (12, 52)
TEXT_SCALE = 0.6


def sweep_eye_positions(screen: ScreenConfig, frames: int) -> np.ndarray:
    """Build a synthetic eye path, in the camera frame.

    The path is written in viewer terms - so much to the right, so much up, so
    far away - and converted with phase 3's own axis constant rather than by
    restating the signs here. ``CAMERA_TO_SCREEN`` is its own inverse, being a
    180 degree rotation about Z, so multiplying by it converts either way.

    Args:
        screen: The display, which fixes where the camera sits.
        frames: How many positions to produce.

    Returns:
        Eye positions ``(frames, 3)`` in the camera frame, metres.
    """
    turns = np.linspace(0, 2 * np.pi, frames, endpoint=False)
    viewer = np.stack(
        [
            SWEEP_RIGHT_M * np.sin(turns),
            SWEEP_UP_M * np.sin(2 * turns),
            SWEEP_DISTANCE_M + SWEEP_DISTANCE_SWING_M * np.cos(turns),
        ],
        axis=1,
    )
    return np.asarray(screen.camera_to_centre_m) + viewer * CAMERA_TO_SCREEN


def read_track(path: Path, *, raw: bool) -> tuple[np.ndarray, list[bool]]:
    """Read an eye track written by ``scripts/viewer_position.py``.

    The smoothed columns already hold position through a frame with no face:
    the tracker calls ``PositionSmoother.hold()`` there and writes the last
    estimate rather than inventing one. So holding needs no code here, only the
    choice of column. The leading frames before any face has been seen have
    nothing to hold and are skipped.

    Args:
        path: The CSV to read.
        raw: Use the unsmoothed columns instead, to show what the smoother does.

    Returns:
        Eye positions ``(N, 3)`` in the camera frame, and a flag per position
        saying whether it was held rather than measured.
    """
    suffix = "_m" if raw else "_smooth_m"
    positions: list[list[float]] = []
    held: list[bool] = []
    skipped = 0
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            values = [row[f"{axis}{suffix}"] for axis in "xyz"]
            if any(v == "" for v in values):
                skipped += 1
                continue
            positions.append([float(v) for v in values])
            held.append(row["has_face"] != "True")
    if skipped:
        lg.info(f"Skipped {skipped} leading frames with no position to hold")
    return np.array(positions), held


def annotate(frame: np.ndarray, eye_camera_m: np.ndarray, *, held: bool) -> None:
    """Write the eye position onto the frame, in place.

    Ninety frames are unreviewable without it, and marking a held position is
    the same guard phase 5 will need: a capture that dies looks exactly like a
    viewer who left, and only the frame itself can say which.

    Args:
        frame: The image to draw on.
        eye_camera_m: Eye position in the camera frame, metres.
        held: Whether this position was held rather than measured.
    """
    x, y, z = (float(v) for v in eye_camera_m)
    cv.putText(
        frame,
        f"eye {x:+.3f} {y:+.3f} {z:.3f} m (camera frame)",
        TEXT_ORIGIN,
        cv.FONT_HERSHEY_SIMPLEX,
        TEXT_SCALE,
        TEXT_BGR,
        1,
        cv.LINE_AA,
    )
    if held:
        cv.putText(
            frame,
            "HELD: no face this frame",
            HELD_ORIGIN,
            cv.FONT_HERSHEY_SIMPLEX,
            TEXT_SCALE,
            HELD_BGR,
            1,
            cv.LINE_AA,
        )


def contact_sheet(frames: list[np.ndarray]) -> np.ndarray:
    """Tile a few frames into one image.

    Args:
        frames: Frames to tile, at most ``CONTACT_TILES ** 2`` of them.

    Returns:
        A single image of the same size as one frame.
    """
    height, width = frames[0].shape[:2]
    tile = (width // CONTACT_TILES, height // CONTACT_TILES)
    small = [cv.resize(frame, tile) for frame in frames]
    blank = np.zeros((tile[1], tile[0], 3), dtype=np.uint8)
    small += [blank] * (CONTACT_TILES**2 - len(small))
    rows = [
        np.hstack(small[i * CONTACT_TILES : (i + 1) * CONTACT_TILES])
        for i in range(CONTACT_TILES)
    ]
    return np.vstack(rows)


def contact_indices(count: int) -> list[int]:
    """Pick evenly spaced frames for the contact sheet.

    Args:
        count: How many frames there are.

    Returns:
        Indices into the frame sequence.
    """
    tiles = min(CONTACT_TILES**2, count)
    return [round(i * (count - 1) / max(tiles - 1, 1)) for i in range(tiles)]


def render_run(
    screen: ScreenConfig,
    positions: np.ndarray,
    held: list[bool],
    config: SinkConfig,
) -> None:
    """Render every position and write the artefacts.

    Args:
        screen: The display being rendered for.
        positions: Eye positions ``(N, 3)`` in the camera frame.
        held: Per position, whether it was held rather than measured.
        config: Where the frames go and how big they are.
    """
    renderer = WireframeRenderer(window_box(screen))
    png, video = PngSink(config), VideoSink(config)
    wanted = set(contact_indices(len(positions)))
    sheet: list[np.ndarray] = []

    for idx, (eye, was_held) in enumerate(zip(positions, held, strict=True)):
        try:
            frame = render_frame(screen, eye, renderer, config.size)
        except EyeBehindScreenError:
            lg.warning(f"Frame {idx}: eye is not in front of the panel, skipping")
            continue
        annotate(frame, eye, held=was_held)
        png.write(frame)
        video.write(frame)
        if idx in wanted:
            sheet.append(frame)

    png.close()
    video.close()
    sheet_path = config.out_fol / "contact_sheet.png"
    cv.imwrite(str(sheet_path), contact_sheet(sheet))
    lg.info(f"Wrote {sheet_path}")


def sweep_coordinates(
    screen: ScreenConfig,
    positions: np.ndarray,
    size: tuple[int, int],
) -> list[dict]:
    """Project every scene endpoint at a few eye positions.

    This is the regression baseline. It is coordinates rather than images
    because numbers diff readably and pixels do not, and because it isolates
    the projection from however OpenCV happens to draw a line this version.

    Args:
        screen: The display being rendered for.
        positions: The full sweep, of which a few steps are sampled.
        size: Output ``(width_px, height_px)``.

    Returns:
        One row per segment endpoint per sampled step.
    """
    scene = window_box(screen)
    width_px, height_px = size
    rows: list[dict] = []
    for step, idx in enumerate(contact_indices(len(positions))):
        matrix = view_projection_matrix(screen, positions[idx])
        pixels = project_points(
            matrix, scene.segments.reshape(-1, 3), width_px, height_px
        )
        for point, (u, v) in enumerate(pixels):
            rows.append(
                {
                    "step": step,
                    "segment": point // 2,
                    "end": point % 2,
                    "u_px": round(float(u), 4),
                    "v_px": round(float(v), 4),
                }
            )
    return rows


def write_coordinates(path: Path, rows: list[dict]) -> None:
    """Write the projected coordinates.

    Args:
        path: Destination CSV.
        rows: Rows from :func:`sweep_coordinates`.
    """
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lg.info(f"Wrote {path}")


def main() -> None:
    """Parse arguments and render."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", default=DEFAULT_SCREEN, help="screen registry key")
    parser.add_argument("--width", type=int, default=DEFAULT_SIZE[0])
    parser.add_argument("--height", type=int, default=DEFAULT_SIZE[1])
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    modes = parser.add_subparsers(dest="mode", required=True)

    sweep = modes.add_parser("sweep", help="render a synthetic eye path")
    sweep.add_argument("--frames", type=int, default=DEFAULT_FRAMES)

    track = modes.add_parser("track", help="render a phase 1 eye track")
    track.add_argument("csv", help="path to a *_eye.csv")
    track.add_argument(
        "--raw",
        action="store_true",
        help="use the unsmoothed columns, to show what the smoother does",
    )

    args = parser.parse_args()
    screen = get_screen(args.screen)
    size = (args.width, args.height)
    out_root = get_abyss_paths().cache_fol / "render"

    if args.mode == "sweep":
        name = "sweep"
        positions = sweep_eye_positions(screen, args.frames)
        held = [False] * len(positions)
    else:
        csv_path = Path(args.csv)
        name = csv_path.stem + ("_raw" if args.raw else "")
        positions, held = read_track(csv_path, raw=args.raw)
        if not len(positions):
            lg.error(f"No usable positions in {csv_path}")
            return

    config = SinkConfig(
        name=name,
        out_fol=out_root / name,
        width_px=size[0],
        height_px=size[1],
        fps=args.fps,
    )
    lg.info(
        f"Rendering {len(positions)} frames for {screen.name} "
        f"into {config.out_fol}"
    )
    render_run(screen, positions, held, config)

    if args.mode == "sweep":
        write_coordinates(
            config.out_fol / "sweep_coordinates.csv",
            sweep_coordinates(screen, positions, size),
        )


if __name__ == "__main__":
    main()
