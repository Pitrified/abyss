"""What the page needs to run on the phone, rather than in a test.

Stage timings on ``window.abyss``, fullscreen and the portrait lock on a tap,
refitting the picture when the window changes, and drawing at the device's
pixel ratio when no size is forced.
"""

from collections.abc import Iterator

from playwright.sync_api import Browser
from playwright.sync_api import Page
from playwright.sync_api import ViewportSize
import pytest

from abyss.ui.browser import PageLog
from abyss.ui.browser import open_page
from abyss.ui.browser import wait_until

STAGES = ["landmarker", "eye", "matrix", "draw"]
"""Per-frame stages, in pipeline order."""

PIXEL_VIEWPORT: ViewportSize = {"width": 412, "height": 892}
"""A Pixel 7 Pro held upright, in CSS pixels at a device pixel ratio of 3.5."""


@pytest.fixture
def scene_page(browser: Browser, base_url: str) -> Iterator[tuple[Page, PageLog]]:
    """Open the camera page drawing the scene for g7's panel, at no forced size."""
    page, log = open_page(browser, f"{base_url}/phone_web/?screen=g7_internal")
    wait_until(page, "window.abyss.status === 'running'")
    yield page, log
    page.context.close()


def test_rolling_median_of_the_stage_timer(scene_page: tuple[Page, PageLog]) -> None:
    """The median is over the newest frames only, and a skipped stage is absent."""
    page, _ = scene_page
    summary = page.evaluate(
        """import('/phone_web/js/timing.js').then(({ StageTimer }) => {
            const timer = new StageTimer(3);
            [100, 1, 2, 3].forEach((ms, idx) => {
                timer.begin(idx * 10);
                timer.record('landmarker', ms);
                timer.end();
            });
            return timer.summary();
        })"""
    )
    assert summary["median"]["landmarker"] == 2
    assert summary["last"] == {"landmarker": 3, "total": 3}
    assert "matrix" not in summary["median"]
    assert summary["median"]["interval"] == 10
    assert summary["fps"] == 100


def test_stage_timings_are_published(scene_page: tuple[Page, PageLog]) -> None:
    """Every stage has a last and a median frame time, and the stages sum to total."""
    page, log = scene_page
    wait_until(
        page, "window.abyss.timing && window.abyss.timing.last.draw !== undefined"
    )
    first = page.evaluate("window.abyss.frames")
    wait_until(page, f"window.abyss.frames > {first + 10}")
    timing = page.evaluate("window.abyss.timing")
    for stage in STAGES:
        assert timing["median"][stage] >= 0, stage
    assert timing["median"]["landmarker"] > 0
    last = timing["last"]
    assert sum(last[stage] for stage in STAGES) == pytest.approx(last["total"])
    assert timing["median"]["interval"] > 0
    assert timing["fps"] == pytest.approx(1000 / timing["median"]["interval"])
    assert not log.errors


def test_a_tap_goes_fullscreen_and_reports_the_lock(
    scene_page: tuple[Page, PageLog],
) -> None:
    """Fullscreen needs a gesture; the lock's outcome is reported either way.

    Desktop Chromium refuses the portrait lock, so headless the outcome is an
    error name; on the phone it should be "locked".
    """
    page, log = scene_page
    assert page.evaluate("window.abyss.display.fullscreen") is False
    page.mouse.click(10, 10)
    wait_until(page, "window.abyss.display.fullscreen === true")
    wait_until(page, "window.abyss.display.lock !== null")
    assert isinstance(page.evaluate("window.abyss.display.lock"), str)
    assert not log.errors


def test_the_picture_refits_when_the_window_changes(
    scene_page: tuple[Page, PageLog],
) -> None:
    """Going fullscreen changes the window, and the scene must follow it."""
    page, _ = scene_page
    page.set_viewport_size({"width": 1000, "height": 1000})
    wait_until(page, "window.abyss.output.width === 1000")
    page.set_viewport_size({"width": 1000, "height": 400})
    wait_until(page, "window.abyss.output.height === 400")
    output = page.evaluate("window.abyss.output")
    assert output["width"] == round(400 * 0.344 / 0.193)


def test_the_scene_draws_at_the_device_pixel_ratio(
    browser: Browser, base_url: str
) -> None:
    """With no size given, one CSS pixel is devicePixelRatio drawn pixels."""
    url = f"{base_url}/phone_web/?screen=g7_internal&eye=0,0.1,0.6"
    page, log = open_page(
        browser, url, viewport=PIXEL_VIEWPORT, device_scale_factor=3.5
    )
    wait_until(page, "window.abyss.status === 'done'")
    output = page.evaluate("window.abyss.output")
    canvas = page.evaluate(
        "(() => { const c = document.getElementById('scene');"
        " return [c.width, c.height, c.clientWidth, c.clientHeight]; })()"
    )
    page.context.close()
    assert output["pixelRatio"] == 3.5
    assert canvas[:2] == [round(output["width"] * 3.5), round(output["height"] * 3.5)]
    assert canvas[2:] == [output["width"], output["height"]]
    assert not log.errors


def test_a_forced_size_draws_one_pixel_per_pixel(
    browser: Browser, base_url: str
) -> None:
    """size= is for tests reading frames back, so it ignores the pixel ratio."""
    url = f"{base_url}/phone_web/?screen=g7_internal&eye=0,0.1,0.6&size=640x359"
    page, _ = open_page(browser, url, device_scale_factor=2)
    wait_until(page, "window.abyss.status === 'done'")
    size = page.evaluate(
        "[document.getElementById('scene').width,"
        " document.getElementById('scene').height]"
    )
    page.context.close()
    assert size == [640, 359]
