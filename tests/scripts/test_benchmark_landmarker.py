"""Test the parts of the benchmark that are not the measurement.

The timings themselves cannot be asserted - that is the whole point of running
it - so what is testable is the arithmetic that turns them into a decision: the
size parsing, the summary statistics, and which stages the budget counts.

Loaded by path, the way ``test_render_scene.py`` loads its script: these are run
directly and are not an importable package.
"""

from collections.abc import Callable
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest

from abyss.params.abyss_devices import get_screen
from abyss.render.frustum import eye_in_screen_frame


def load_script() -> ModuleType:
    """Load ``scripts/benchmark_landmarker.py`` by path.

    Returns:
        The loaded module.
    """
    path = Path(__file__).parents[2] / "scripts" / "benchmark_landmarker.py"
    spec = importlib.util.spec_from_file_location("benchmark_landmarker", path)
    if spec is None or spec.loader is None:
        pytest.skip(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_landmarker"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script() -> ModuleType:
    """Load the script under test, once for the module."""
    return load_script()


@pytest.fixture
def make_times(script) -> Callable[..., object]:
    """Build a `StageTimes` with a known median.

    A fixture rather than a plain helper because the module is loaded by path,
    so its types cannot be imported here to annotate a return with.

    Args:
        script: The loaded module.

    Returns:
        A callable taking ``(path, stage, size, ms)``.
    """

    def build(path: str, stage: str, size: tuple[int, int], ms: float) -> object:
        return script.StageTimes(path, stage, size[0], size[1], [ms] * 5)

    return build


def test_sizes_parse(script) -> None:
    """The two size axes arrive as strings from the command line."""
    assert script.parse_sizes("1280x720,640x480") == [(1280, 720), (640, 480)]
    assert script.parse_sizes(" 1920X1080 ") == [(1920, 1080)]


def test_the_default_output_sizes_fit_the_panel(script) -> None:
    """The plan's single frame-size axis does not survive this.

    640x480 is a real capture mode and a valid input size, and it is 4:3, so
    rendering into it raises against a 16:9 panel. Splitting the axis is what
    makes both halves measurable, and this pins that the defaults respect it.
    """
    from abyss.render.renderer import AspectMismatchError
    from abyss.render.renderer import check_aspect

    screen = get_screen("g7_internal")
    for width, height in script.parse_sizes(script.DEFAULT_OUTPUT_SIZES):
        check_aspect(screen, width, height)

    with pytest.raises(AspectMismatchError):
        check_aspect(screen, 640, 480)


def test_statistics_come_from_the_samples(script) -> None:
    """Median, p95 and the implied frame rate, on a known distribution."""
    entry = script.StageTimes("tracker", "landmark", 1280, 720, [10.0] * 19 + [50.0])
    assert entry.median_ms == pytest.approx(10.0)
    assert entry.stage_fps == pytest.approx(100.0)
    # numpy interpolates: 0.95 * 19 = 18.05, so 5% of the way from 10 to 50.
    assert entry.p95_ms == pytest.approx(12.0)


def test_the_row_carries_the_machine(script) -> None:
    """Results are compared across machines, so every row names one."""
    entry = script.StageTimes("render", "render", 1920, 1080, [8.0] * 5)
    row = entry.as_row("g7")
    assert row["machine"] == "g7"
    assert row["frames"] == 5
    assert set(row) == set(script.CSV_FIELDS)


def test_the_budget_excludes_the_sink(script, make_times) -> None:
    """The live sink is a window; only `PngSink` exists to time.

    Folding a PNG encode into the total would report a budget for a loop nobody
    is going to run, and it is a large enough stage to change the verdict on
    its own. So a sink of any cost must leave the number alone.
    """
    timings = [
        make_times("tracker", "decode", (1280, 720), 1.0),
        make_times("tracker", "landmark", (1280, 720), 12.0),
        make_times("tracker", "eye_position", (1280, 720), 0.1),
        make_times("render", "projection", (1920, 1080), 0.1),
        make_times("render", "render", (1920, 1080), 9.0),
        make_times("render", "sink", (1920, 1080), 900.0),
    ]
    budget = script.loop_budget_ms(timings, (1280, 720), (1920, 1080))
    assert budget == pytest.approx(22.2)


def test_the_budget_picks_the_requested_sizes(script, make_times) -> None:
    """Both axes are swept, so the wrong size would quietly be summed in."""
    timings = [
        make_times("tracker", "landmark", (1280, 720), 12.0),
        make_times("tracker", "landmark", (640, 480), 13.0),
        make_times("render", "render", (1920, 1080), 9.0),
        make_times("render", "render", (1280, 720), 4.0),
    ]
    cheap = script.loop_budget_ms(timings, (640, 480), (1280, 720))
    dear = script.loop_budget_ms(timings, (1280, 720), (1920, 1080))
    assert cheap == pytest.approx(17.0)
    assert dear == pytest.approx(21.0)


def test_the_bench_path_moves_and_stays_in_front(script) -> None:
    """A fixed eye would let a cache flatter the render stage.

    In front of the panel because the frustum cannot be built otherwise, and
    the benchmark would then be timing an exception path.
    """
    screen = get_screen("g7_internal")
    positions = script.bench_eye_positions(screen, 32)
    in_screen = np.array([eye_in_screen_frame(p, screen) for p in positions])

    assert in_screen[:, 2].min() > 0
    assert np.ptp(in_screen[:, 0]) > 0
    assert np.ptp(in_screen[:, 1]) > 0
