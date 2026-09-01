"""Test the live script's command line.

Only the parsing. Everything below it needs a camera or a display, and is
covered by `tests/test_loop.py` through the loop's injected source and sink.

This file exists because of a real failure rather than for completeness: the
shared options were defined on the top-level parser, so the invocation printed
in the script's own docstring and in the runbook - `live.py camera
--viewer-ipd-mm 60` - was rejected by argparse. The documentation and the code
disagreed and nothing checked, so the test is that the documented commands
parse.
"""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "live.py"
RUNBOOK = Path(__file__).parents[2] / "docs" / "guides" / "phase5_live_runbook.md"


def load_script() -> ModuleType:
    """Load ``scripts/live.py`` by path.

    Returns:
        The loaded module.
    """
    spec = importlib.util.spec_from_file_location("live", SCRIPT)
    if spec is None or spec.loader is None:
        pytest.skip(f"Cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["live"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script() -> ModuleType:
    """Load the script under test, once for the module."""
    return load_script()


@pytest.mark.parametrize(
    "argv",
    [
        ["camera"],
        ["camera", "--viewer-ipd-mm", "60"],
        ["camera", "--viewer-ipd-mm", "60", "--width", "1920", "--height", "1080"],
        ["camera", "--device", "0", "--camera", "g7_webcam"],
        ["clip", "face01.mp4"],
        ["clip", "face01.mp4", "--viewer-ipd-mm", "60"],
        ["clip", "face01.mp4", "--screen", "g4_internal"],
    ],
)
def test_the_documented_invocations_parse(script, argv) -> None:
    """Every form the docstring and the runbook tell someone to type.

    The shared options must follow the subcommand, because that is where a
    person puts them and where both documents say to put them.
    """
    args = script.build_parser().parse_args(argv)
    assert args.mode == argv[0]


def test_the_interpupillary_distance_reaches_the_run(script) -> None:
    """It is the last unmeasured number in the chain and it scales depth.

    A flag that parses into the wrong attribute would silently fall back to the
    63 mm population default, which is a 5% depth error for a 60 mm viewer and
    looks like nothing at all on screen.
    """
    args = script.build_parser().parse_args(["camera", "--viewer-ipd-mm", "60"])
    assert args.viewer_ipd_mm == pytest.approx(60.0)


def test_the_default_is_the_population_mean(script) -> None:
    """Unmeasured, and documented as such, rather than absent."""
    args = script.build_parser().parse_args(["camera"])
    assert args.viewer_ipd_mm == pytest.approx(63.0)


def test_a_subcommand_is_required(script) -> None:
    """There is no sensible default between live and offline."""
    with pytest.raises(SystemExit):
        script.build_parser().parse_args([])


def test_the_runbook_commands_parse(script) -> None:
    """The runbook is what someone follows at the desk, unread by any test.

    It carries the exact command lines, so they are extracted and parsed here.
    That is what would have caught the original defect: the runbook said to
    type something argparse refused.
    """
    parser = script.build_parser()
    found = 0
    for line in RUNBOOK.read_text().splitlines():
        stripped = line.strip()
        if "scripts/live.py" not in stripped or stripped.startswith("|"):
            continue
        argv = stripped.split("scripts/live.py", 1)[1].split()
        # The runbook writes the viewer's own number as a placeholder.
        argv = ["60" if a == "<yours>" else a for a in argv]
        parser.parse_args(argv)
        found += 1
    assert found >= 2
