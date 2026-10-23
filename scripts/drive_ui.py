"""Drive a UI in a real browser and watch everything it does in one stream.

    uv run --no-sync python scripts/drive_ui.py --do "wait 2" --do state --do "shot a"

Serves the UIs, opens one in Chromium with the synthetic clip as the camera,
runs the ``--do`` steps in order, and prints the browser console, page errors,
requests and the server's log as one time-ordered stream. Screenshots go to
``build/ui/``.

Not a test. ``make e2e`` is the check; this is the instrument for when a check
fails and the question is where, between the camera and the canvas, the thing
stopped. The idea is klide's ``scripts/drive.py``, which found a button that did
nothing while every server-side probe said the path worked
(``plans/03_phone_webapp/01_research_dev_harness.md``).

Steps:

- ``wait <seconds>``: let the page run.
- ``state``: print ``window.abyss``.
- ``eval <expression>``: print the value of a JavaScript expression.
- ``shot <name>``: screenshot to ``build/ui/<name>.png``.
- ``until <expression>``: wait until the expression is truthy.

Steps run through DevTools, so they are not subject to the page's
Content-Security-Policy; anything the page itself does still is.
"""

import argparse
import json
from pathlib import Path
import sys
import time

from loguru import logger as lg

from abyss.params.abyss_params import get_abyss_paths
from abyss.ui.browser import launch_chromium
from abyss.ui.browser import wait_until
from abyss.ui.clip import write_clip
from abyss.ui.serve import running_server
from abyss.ui.vendor import ensure_vendor

START = time.monotonic()

PHONE_VIEWPORT = "412x892"
"""The Pixel 7 Pro's viewport in Chrome, in CSS pixels, portrait."""


def out(source: str, line: str) -> None:
    """Print one line of the stream, stamped from the start of the run.

    Args:
        source: Where the line came from.
        line: The line.
    """
    print(f"{time.monotonic() - START:7.2f}  {source:8s} {line}", flush=True)


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Drive a UI in a real browser.")
    parser.add_argument("--ui", default="phone_web", help="UI folder under ui/")
    parser.add_argument("--query", default="", help="query string, without the ?")
    parser.add_argument(
        "--clip", type=Path, help="camera clip; default the synthetic one"
    )
    parser.add_argument(
        "--no-camera", action="store_true", help="no fake camera at all"
    )
    parser.add_argument(
        "--headed", action="store_true", help="open a window (needs a display)"
    )
    parser.add_argument(
        "--viewport",
        default=PHONE_VIEWPORT,
        help="CSS pixels WxH; the default is a Pixel 7 Pro held upright",
    )
    parser.add_argument("--do", action="append", default=[], help="a step, repeatable")
    return parser.parse_args()


def run_step(page: object, step: str, shots: Path) -> None:
    """Run one ``--do`` step.

    Args:
        page: The Playwright page.
        step: The step text.
        shots: Screenshot folder.
    """
    verb, _, arg = step.partition(" ")
    if verb == "wait":
        page.wait_for_timeout(float(arg) * 1000)  # type: ignore[attr-defined]
    elif verb == "state":
        out("drive", json.dumps(page.evaluate("window.abyss")))  # type: ignore[attr-defined]
    elif verb == "eval":
        out("drive", json.dumps(page.evaluate(arg)))  # type: ignore[attr-defined]
    elif verb == "until":
        out("drive", f"until {arg}: {json.dumps(wait_until(page, arg))}")  # type: ignore[arg-type]
    elif verb == "shot":
        path = shots / f"{arg or 'shot'}.png"
        page.screenshot(path=str(path))  # type: ignore[attr-defined]
        out("drive", f"screenshot {path}")
    else:
        out("drive", f"unknown step {step!r}")


def main() -> None:
    """Serve, open, run the steps, close."""
    args = parse_args()
    lg.remove()
    lg.add(lambda message: out("server", message.record["message"]), level="INFO")
    ensure_vendor()
    clip = None
    if not args.no_camera:
        clip = args.clip or write_clip()
    shots = get_abyss_paths().build_fol / "ui"
    shots.mkdir(parents=True, exist_ok=True)

    with (
        running_server() as base,
        launch_chromium(clip, headless=not args.headed) as browser,
    ):
        url = f"{base}/{args.ui}/" + (f"?{args.query}" if args.query else "")
        width, height = (int(v) for v in args.viewport.split("x"))
        page = browser.new_page(viewport={"width": width, "height": height})
        page.on("console", lambda m: out("browser", f"{m.type}: {m.text}"))
        page.on("pageerror", lambda e: out("browser", f"pageerror: {e}"))
        page.on(
            "requestfailed", lambda r: out("browser", f"failed: {r.url} {r.failure}")
        )
        out("drive", f"open {url} camera={clip}")
        page.goto(url)
        for step in args.do:
            run_step(page, step, shots)
    out("drive", "done")


if __name__ == "__main__":
    sys.exit(main())
