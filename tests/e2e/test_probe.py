"""The probe page: what a device says about itself, ready to copy."""

import json

from playwright.sync_api import Browser
import pytest

from abyss.ui.browser import open_page
from abyss.ui.browser import wait_until
from abyss.ui.clip import FRAME_HEIGHT
from abyss.ui.clip import FRAME_WIDTH


def test_probe_reads_the_device_and_prints_it(browser: Browser, base_url: str) -> None:
    """Screen, camera stream and GPU are read, and the text is the same JSON."""
    page, log = open_page(browser, f"{base_url}/probe/?delay=0", device_scale_factor=2)
    wait_until(page, "window.probe && window.probe.status === 'done'")
    probe = page.evaluate("window.probe")
    assert probe["version"].startswith("ui ")
    assert probe["screen"]["devicePixelRatio"] == 2
    assert probe["screen"]["physical"] == [
        probe["screen"]["width"] * 2,
        probe["screen"]["height"] * 2,
    ]
    settings = probe["camera"]["settings"]
    assert (settings["width"], settings["height"]) == (FRAME_WIDTH, FRAME_HEIGHT)
    assert (probe["camera"]["video"]["width"], probe["camera"]["video"]["height"]) == (
        FRAME_WIDTH,
        FRAME_HEIGHT,
    )
    assert probe["webgl"]["version"].startswith("WebGL 2")
    assert json.loads(page.inner_text("#out")) == probe
    assert log.off_server(base_url) == []
    assert not log.errors
    page.context.close()


def test_probe_tap_records_fullscreen_and_the_lock(
    browser: Browser, base_url: str
) -> None:
    """The tap is the gesture fullscreen needs; what changed is read again."""
    page, log = open_page(browser, f"{base_url}/probe/?delay=0")
    wait_until(page, "window.probe && window.probe.status === 'done'")
    page.click("#fullscreen")
    wait_until(page, "window.probe.afterTap && window.probe.afterTap.lock !== null")
    after = page.evaluate("window.probe.afterTap")
    assert after["fullscreen"] is True
    assert isinstance(after["lock"], str)
    assert "window" in after
    assert not log.errors
    page.context.close()


def test_camera_sweep_measures_each_request(browser: Browser, base_url: str) -> None:
    """Each camera request is opened, tracked, and reported with its coverage.

    The fake camera answers every request from the same clip, so headless this
    checks the plumbing: every variant reports what it got and an iris spacing.
    """
    page, log = open_page(browser, f"{base_url}/probe/?delay=0")
    wait_until(page, "window.probe && window.probe.status === 'done'")
    assert page.evaluate("window.probe.sweep.status") == "done"
    sweep = page.evaluate("window.probe.sweep")
    assert len(sweep["variants"]) >= 8
    for variant in sweep["variants"]:
        assert "error" not in variant, variant
        assert variant["settings"]["width"] > 0
        assert variant["ipdPx"] > 0, variant["name"]
        assert variant["widthPerIpd"] == pytest.approx(
            variant["video"][0] / variant["ipdPx"], rel=1e-3
        )
    assert sweep["widest"] in {v["name"] for v in sweep["variants"]}
    assert json.loads(page.inner_text("#out"))["sweep"] == sweep
    assert not log.errors
    page.context.close()
