"""A synthetic Gaussian splat at known points, for the browser tests.

A handful of small, opaque, single-coloured Gaussians in the screen frame,
written as a 3D Gaussian Splatting PLY file, with a placement file beside it
that leaves them where they are. A test projects :data:`TEST_SPLAT_POINTS` with
Python and looks for each colour at that pixel in the frame Spark drew, which is
the splat path's version of the wireframe readback: it catches a transposed
matrix, a mirrored or flipped frame, or a camera left at the wrong point.

Why each choice:

- **PLY.** The format gsplat and the reference implementation write, so the
  test goes through the loader a trained splat will. No SH bands past degree 0:
  the colour does not depend on the view.
- **Opaque and small.** A Gaussian of a few millimetres at full opacity draws as
  a solid disc whose centre is the projected point.
- **Distinct colours.** Each point is found by its own colour, so a swap of two
  points is a failure rather than a pass.

Run as ``python -m abyss.ui.splat`` to write it by hand.
"""

import json
from pathlib import Path

from loguru import logger as lg
import numpy as np

from abyss.params.abyss_params import get_abyss_paths

SH_C0 = 0.28209479177387814
"""The degree-0 spherical harmonic: a PLY stores colour as ``(rgb - 0.5) / SH_C0``."""

TEST_SPLAT_NAME = "test_points"

TEST_SPLAT_POINTS = np.array(
    [
        [-0.10, 0.05, -0.05],
        [0.10, 0.05, -0.10],
        [0.00, 0.00, -0.15],
        [-0.08, -0.05, -0.20],
        [0.09, -0.04, -0.08],
    ]
)
"""Screen-frame centres, metres: behind ``g7_internal``'s panel and inside it."""

TEST_SPLAT_COLOURS_BGR = np.array(
    [
        [0, 0, 255],
        [0, 255, 0],
        [255, 0, 0],
        [0, 255, 255],
        [255, 0, 255],
    ],
    dtype=np.uint8,
)
"""One colour per point: red, green, blue, yellow, magenta."""

TEST_SPLAT_RADIUS_M = 0.004
"""Every axis of every Gaussian, one standard deviation."""

OPACITY_LOGIT = 8.0
"""A PLY stores opacity before its sigmoid; 8 is 0.9997."""

PLY_FIELDS = (
    "x",
    "y",
    "z",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
)
"""The 3DGS vertex layout, degree 0 only. ``rot_0`` is the quaternion's w."""

IDENTITY_PLACEMENT = {"position": [0, 0, 0], "quaternion": [0, 0, 0, 1], "scale": 1}
"""A placement that leaves the splat in the screen frame as written."""


def write_ply(
    path: Path, points: np.ndarray, colours_bgr: np.ndarray, radius_m: float
) -> None:
    """Write round, opaque Gaussians as a 3DGS PLY file.

    Args:
        path: The file to write.
        points: ``(N, 3)`` centres, metres.
        colours_bgr: ``(N, 3)`` colours, BGR, 0 to 255.
        radius_m: One standard deviation on every axis, metres.
    """
    rgb = colours_bgr[:, ::-1].astype(float) / 255
    data = np.zeros((len(points), len(PLY_FIELDS)), dtype="<f4")
    data[:, 0:3] = points
    data[:, 3:6] = (rgb - 0.5) / SH_C0
    data[:, 6] = OPACITY_LOGIT
    data[:, 7:10] = np.log(radius_m)
    data[:, 10] = 1.0
    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {len(points)}\n"
    header += "".join(f"property float {name}\n" for name in PLY_FIELDS)
    header += "end_header\n"
    path.write_bytes(header.encode() + data.tobytes())


def write_test_splat(folder: Path | None = None) -> Path:
    """Write the test splat and its placement file.

    Args:
        folder: Where to write; defaults to the build folder's ``splats/``.

    Returns:
        The folder, which holds ``<TEST_SPLAT_NAME>.ply`` and ``.json``.
    """
    folder = folder or get_abyss_paths().test_splats_fol
    folder.mkdir(parents=True, exist_ok=True)
    ply = folder / f"{TEST_SPLAT_NAME}.ply"
    write_ply(ply, TEST_SPLAT_POINTS, TEST_SPLAT_COLOURS_BGR, TEST_SPLAT_RADIUS_M)
    placement = {"file": ply.name, **IDENTITY_PLACEMENT}
    (folder / f"{TEST_SPLAT_NAME}.json").write_text(json.dumps(placement) + "\n")
    lg.info(f"Wrote {ply}")
    return folder


if __name__ == "__main__":
    write_test_splat()
