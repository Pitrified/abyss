"""Tests for the ChArUco calibration script.

The script is not importable as a package module, so it is loaded by path.
These tests exist because the claims in its docstring are the kind that sound
right and are worth failing loudly if OpenCV changes underneath them.
"""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import cv2 as cv
import numpy as np
import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "calibrate_camera.py"

TRUE_FOCAL = 900.0
FRAME_W, FRAME_H = 1280, 720
FOCAL_TOLERANCE_PX = 1.0
"""Synthetic data has no noise, so recovery should be near exact."""


def load_script() -> ModuleType:
    """Import the script by path.

    Returns:
        The loaded module.
    """
    spec = importlib.util.spec_from_file_location("calibrate_camera", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script() -> ModuleType:
    """Load the calibration script once for the whole module."""
    return load_script()


@pytest.fixture(scope="module")
def spec(script: ModuleType):  # noqa: ANN201
    """Build a board spec matching what the Kindle preset would produce."""
    return script.BoardSpec(
        squares_x=7,
        squares_y=9,
        square_m=0.0149,
        marker_m=0.0149 * 0.75,
        dictionary="DICT_4X4_50",
        square_px=176,
        ppi=300.0,
        source="test",
    )


def synth_views(
    board: cv.aruco.CharucoBoard,
    focal: float,
    scale: float = 1.0,
) -> tuple[list, list]:
    """Project the board into several tilted views with a known camera.

    Args:
        board: The board to project.
        focal: True focal length in pixels, used for both axes.
        scale: Factor applied to the board's metric size, and to the distances
            so the resulting images are identical.

    Returns:
        Object point arrays and image point arrays, one pair per view.
    """
    matrix = np.array(
        [[focal, 0, FRAME_W / 2], [0, focal, FRAME_H / 2], [0, 0, 1]],
        dtype=float,
    )
    obj = np.asarray(board.getChessboardCorners(), dtype=np.float32)
    rng = np.random.default_rng(0)
    object_points, image_points = [], []
    for _ in range(14):
        rvec = rng.uniform(-0.45, 0.45, 3)
        tvec = np.array(
            [rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), rng.uniform(0.35, 0.6)]
        )
        # Scale the board and the distance together, so the projected image is
        # identical for every scale and the only difference is the metric size
        # the calibration is told the board has.
        projected, _ = cv.projectPoints(obj * scale, rvec, tvec * scale, matrix, None)
        flat = projected.reshape(-1, 2).astype(np.float32)
        inside = (
            flat[:, 0].min() >= 0
            and flat[:, 0].max() < FRAME_W
            and flat[:, 1].min() >= 0
            and flat[:, 1].max() < FRAME_H
        )
        if not inside:
            continue
        object_points.append(obj * scale)
        image_points.append(flat)
    return object_points, image_points


def test_calibration_recovers_a_known_focal_length(
    script: ModuleType,
    spec,
) -> None:
    """The whole point: tilted views of a board give back the true focal."""
    board = script.build_board(spec)
    object_points, image_points = synth_views(board, TRUE_FOCAL)
    assert len(object_points) >= script.MIN_VIEWS

    rms, matrix, _, _, _ = cv.calibrateCamera(
        object_points, image_points, (FRAME_W, FRAME_H), None, None
    )
    assert rms < 0.1
    assert matrix[0, 0] == pytest.approx(TRUE_FOCAL, abs=FOCAL_TOLERANCE_PX)
    assert matrix[1, 1] == pytest.approx(TRUE_FOCAL, abs=FOCAL_TOLERANCE_PX)


def test_intrinsics_do_not_depend_on_the_board_size(
    script: ModuleType,
    spec,
) -> None:
    """Scaling the board scales the distances and leaves the focal alone.

    This is what makes a screen usable as a target without trusting its pixel
    pitch for the focal length, and it is the claim most worth pinning.
    """
    board = script.build_board(spec)
    focals = []
    for scale in (1.0, 2.0):
        object_points, image_points = synth_views(board, TRUE_FOCAL, scale=scale)
        _, matrix, _, _, translations = cv.calibrateCamera(
            object_points, image_points, (FRAME_W, FRAME_H), None, None
        )
        focals.append(matrix[1, 1])
        mean_depth = np.mean([np.linalg.norm(t) for t in translations])
        # The distances must scale, or the two runs were not actually different.
        assert mean_depth == pytest.approx(0.49 * scale, rel=0.25)
    assert focals[0] == pytest.approx(focals[1], abs=FOCAL_TOLERANCE_PX)


def test_detect_corners_finds_the_rendered_board(
    script: ModuleType,
    spec,
) -> None:
    """The detector reads back a board the script itself rendered."""
    board = script.build_board(spec)
    image = board.generateImage(
        (spec.square_px * spec.squares_x, spec.square_px * spec.squares_y)
    )
    found = script.detect_corners(image, board)
    assert found is not None
    object_points, image_points = found
    # A 7x9 board has 6x8 interior corners, all visible in a flat render.
    assert len(object_points) == (spec.squares_x - 1) * (spec.squares_y - 1)
    assert len(image_points) == len(object_points)


def test_detect_corners_rejects_a_blank_image(
    script: ModuleType,
    spec,
) -> None:
    """No board present means None, not an exception or a partial result."""
    board = script.build_board(spec)
    blank = np.full((FRAME_H, FRAME_W), 255, dtype=np.uint8)
    assert script.detect_corners(blank, board) is None


def test_square_size_follows_the_pixel_pitch(script: ModuleType) -> None:
    """300 ppi means a square is its pixel count over 300 inches, exactly."""
    preset = script.DEVICE_PRESETS["kindle_pw11"]
    assert preset.ppi == 300.0
    square_px = min(preset.width_px // 7, preset.height_px // 9)
    square_m = square_px / preset.ppi * script.MM_PER_INCH / 1000
    assert square_m == pytest.approx(square_px * 25.4 / 300 / 1000)


def test_presets_imply_their_advertised_diagonal(script: ModuleType) -> None:
    """A wrong ppi shows up as a diagonal that disagrees with the spec sheet."""
    advertised = {"kindle_pw11": 6.8, "pixel7pro": 6.7}
    for name, inches in advertised.items():
        preset = script.DEVICE_PRESETS[name]
        diagonal = float(np.hypot(preset.width_px, preset.height_px)) / preset.ppi
        assert diagonal == pytest.approx(inches, abs=0.1)
