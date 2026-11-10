"""Test the synthetic splat's PLY file and placement file."""

import json
from pathlib import Path

import numpy as np
import pytest

from abyss.ui.splat import PLY_FIELDS
from abyss.ui.splat import SH_C0
from abyss.ui.splat import TEST_SPLAT_COLOURS_BGR
from abyss.ui.splat import TEST_SPLAT_NAME
from abyss.ui.splat import TEST_SPLAT_POINTS
from abyss.ui.splat import TEST_SPLAT_RADIUS_M
from abyss.ui.splat import TYPICAL_EYE_M
from abyss.ui.splat import pick_camera
from abyss.ui.splat import placement_at_camera
from abyss.ui.splat import quaternion_from_matrix
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


def rotate(quaternion: list[float], vector: np.ndarray) -> np.ndarray:
    """Rotate a vector by an ``[x, y, z, w]`` quaternion."""
    q = np.array(quaternion[:3])
    w = quaternion[3]
    return vector + 2 * np.cross(q, np.cross(q, vector) + w * vector)


@pytest.mark.parametrize("heading_deg", [0, 37, 90, 180, 251])
def test_placement_puts_the_camera_at_the_eye_facing_the_screen(
    heading_deg: float,
) -> None:
    """Centre to the eye, heading to -z, up to +y, for any heading."""
    heading = np.radians(heading_deg)
    centre = np.array([0.8, -0.3, 0.6])
    forward = np.array([np.cos(heading), np.sin(heading), -0.4])
    scale = 1.5
    placement = placement_at_camera(centre, forward, TYPICAL_EYE_M, scale)
    q = placement["quaternion"]
    assert np.isclose(np.linalg.norm(q), 1)
    placed = scale * rotate(q, centre) + placement["position"]
    assert np.allclose(placed, TYPICAL_EYE_M, atol=1e-4)
    level = np.array([forward[0], forward[1], 0]) / np.hypot(*forward[:2])
    assert np.allclose(rotate(q, level), [0, 0, -1], atol=1e-5)
    assert np.allclose(rotate(q, np.array([0, 0, 1.0])), [0, 1, 0], atol=1e-5)


def test_quaternion_of_identity() -> None:
    """No rotation is the identity quaternion, not a division by zero."""
    assert quaternion_from_matrix(np.eye(3)) == [0, 0, 0, 1]


def test_pick_camera_prefers_far_and_facing_the_middle() -> None:
    """Of cameras on a ring, the one facing inward wins over one facing out."""
    centres = np.array([[1.0, 0, 0.5], [0, 1.0, 0.5], [0.5, 0, 0.5]])
    forwards = np.array([[1.0, 0, 0], [0, -1.0, -0.3], [-1.0, 0, 0]])
    assert pick_camera(centres, forwards) == 1


def test_colours_are_distinct() -> None:
    """Each point is found by its own colour, so no two may share one."""
    assert len({tuple(c) for c in TEST_SPLAT_COLOURS_BGR}) == len(
        TEST_SPLAT_COLOURS_BGR
    )
