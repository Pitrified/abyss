"""Test the synthetic splat's PLY file and placement file."""

import json
from pathlib import Path

import numpy as np

from abyss.ui.splat import PLY_FIELDS
from abyss.ui.splat import SH_C0
from abyss.ui.splat import TEST_SPLAT_COLOURS_BGR
from abyss.ui.splat import TEST_SPLAT_NAME
from abyss.ui.splat import TEST_SPLAT_POINTS
from abyss.ui.splat import TEST_SPLAT_RADIUS_M
from abyss.ui.splat import write_test_splat


def read_ply(path: Path) -> tuple[list[str], np.ndarray]:
    """Parse the header and the float rows the writer produces."""
    data = path.read_bytes()
    head, body = data.split(b"end_header\n", 1)
    names = [
        line.split()[-1]
        for line in head.decode().splitlines()
        if line.startswith("property")
    ]
    rows = np.frombuffer(body, dtype="<f4").reshape(-1, len(names))
    return names, rows


def test_ply_round_trips_points_colours_and_size(tmp_path: Path) -> None:
    """Centres, colours, scales and rotation come back in the 3DGS layout."""
    folder = write_test_splat(tmp_path)
    names, rows = read_ply(folder / f"{TEST_SPLAT_NAME}.ply")
    assert tuple(names) == PLY_FIELDS
    assert len(rows) == len(TEST_SPLAT_POINTS)
    assert np.allclose(rows[:, 0:3], TEST_SPLAT_POINTS, atol=1e-7)
    rgb = rows[:, 3:6] * SH_C0 + 0.5
    assert np.allclose(rgb * 255, TEST_SPLAT_COLOURS_BGR[:, ::-1], atol=1e-3)
    assert np.allclose(np.exp(rows[:, 7:10]), TEST_SPLAT_RADIUS_M)
    assert np.allclose(rows[:, 10:14], [1, 0, 0, 0])
    assert (1 / (1 + np.exp(-rows[:, 6])) > 0.999).all()


def test_placement_names_the_file_and_leaves_it_in_place(tmp_path: Path) -> None:
    """The placement file points at the PLY with an identity transform."""
    folder = write_test_splat(tmp_path)
    placement = json.loads((folder / f"{TEST_SPLAT_NAME}.json").read_text())
    assert placement == {
        "file": f"{TEST_SPLAT_NAME}.ply",
        "position": [0, 0, 0],
        "quaternion": [0, 0, 0, 1],
        "scale": 1,
    }


def test_colours_are_distinct() -> None:
    """Each point is found by its own colour, so no two may share one."""
    assert len({tuple(c) for c in TEST_SPLAT_COLOURS_BGR}) == len(
        TEST_SPLAT_COLOURS_BGR
    )
