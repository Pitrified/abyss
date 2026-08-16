"""Test the offline render script.

Loaded by path, the way ``test_calibrate_camera.py`` loads its script: these
are run directly and are not an importable package.

The baseline here is coordinates rather than images. Phase 1's regression
baselines live outside the repo because MediaPipe inference is not guaranteed
bit-identical across machines, and that reason does not apply to this phase:
projecting a fixed scene through a fixed matrix is arithmetic, so it can be a
committed fixture and a real test rather than a manual checksum.
"""

import csv
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest

from abyss.params.abyss_devices import get_screen
from abyss.render.frustum import eye_in_screen_frame

BASELINE = Path(__file__).parent / "data" / "sweep_baseline.csv"


def load_script() -> ModuleType:
    """Load ``scripts/render_scene.py`` by path.

    Returns:
        The loaded module.
    """
    path = Path(__file__).parents[2] / "scripts" / "render_scene.py"
    spec = importlib.util.spec_from_file_location("render_scene", path)
    if spec is None or spec.loader is None:
        pytest.skip(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["render_scene"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script() -> ModuleType:
    """Load the script under test, once for the module."""
    return load_script()


def test_the_sweep_is_written_in_viewer_terms(script) -> None:
    """The path is built with phase 3's axis constant, not by restating signs.

    Converting back must recover exactly the amplitudes the constants name. If
    someone flips a sign here, the sweep would still look plausible and would
    silently be a mirror of itself.
    """
    screen = get_screen("g7_internal")
    positions = script.sweep_eye_positions(screen, 64)
    in_screen = np.array([eye_in_screen_frame(p, screen) for p in positions])

    assert in_screen[:, 0].max() == pytest.approx(script.SWEEP_RIGHT_M, rel=1e-3)
    assert in_screen[:, 1].max() == pytest.approx(script.SWEEP_UP_M, rel=1e-3)
    assert in_screen[:, 2].mean() == pytest.approx(script.SWEEP_DISTANCE_M, abs=1e-3)


def test_the_sweep_stays_in_front_of_the_panel(script) -> None:
    """Or the frustum cannot be built and frames get skipped."""
    screen = get_screen("g7_internal")
    positions = script.sweep_eye_positions(screen, script.DEFAULT_FRAMES)
    in_screen = np.array([eye_in_screen_frame(p, screen) for p in positions])
    assert in_screen[:, 2].min() > 0


def test_the_contact_sheet_samples_the_whole_run(script) -> None:
    """A sheet that missed the extremes would hide the parallax."""
    indices = script.contact_indices(90)
    assert indices[0] == 0
    assert indices[-1] == 89
    assert len(indices) == script.CONTACT_TILES**2


def test_the_contact_sheet_handles_a_short_run(script) -> None:
    """Fewer frames than tiles is not an error, just a sparse sheet."""
    assert script.contact_indices(2) == [0, 1]


def test_a_track_holds_position_through_a_missing_face(script, tmp_path) -> None:
    """The hold is upstream: the smoothed column already carries it."""
    path = tmp_path / "track.csv"
    rows = [
        {"has_face": "False", "x_smooth_m": "", "y_smooth_m": "", "z_smooth_m": ""},
        {
            "has_face": "True",
            "x_smooth_m": "0.1",
            "y_smooth_m": "0.2",
            "z_smooth_m": "0.5",
        },
        {
            "has_face": "False",
            "x_smooth_m": "0.1",
            "y_smooth_m": "0.2",
            "z_smooth_m": "0.5",
        },
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    positions, held = script.read_track(path, raw=False)
    # The leading row had nothing to hold, so it is dropped rather than guessed.
    assert len(positions) == 2
    assert held == [False, True]
    assert positions[1] == pytest.approx([0.1, 0.2, 0.5])


def test_the_sweep_projection_is_unchanged(script) -> None:
    """Regression baseline: the numbers this phase produces, committed.

    A failure means the scene, the screen entry or the projection moved. All
    three are things that should move only on purpose.
    """
    screen = get_screen("g7_internal")
    positions = script.sweep_eye_positions(screen, script.DEFAULT_FRAMES)
    rows = script.sweep_coordinates(screen, positions, script.DEFAULT_SIZE)

    with BASELINE.open(newline="") as handle:
        expected = list(csv.DictReader(handle))

    assert len(rows) == len(expected)
    for row, want in zip(rows, expected, strict=True):
        assert row["step"] == int(want["step"])
        assert row["segment"] == int(want["segment"])
        assert row["u_px"] == pytest.approx(float(want["u_px"]), abs=1e-3)
        assert row["v_px"] == pytest.approx(float(want["v_px"]), abs=1e-3)
