"""The version label and the log a person copies off the phone."""

import re

from playwright.sync_api import Browser

from abyss.ui.browser import open_page
from abyss.ui.browser import wait_until


def test_page_shows_the_served_version(browser: Browser, base_url: str) -> None:
    """The label is the version /version.js serves now, so a stale page shows."""
    page, log = open_page(browser, f"{base_url}/phone_web/")
    wait_until(page, "window.abyss.status === 'running'")
    served = page.evaluate("import('/version.js').then(m => m.VERSION)")
    assert served.startswith("ui ")
    assert page.evaluate("window.abyss.version") == served
    assert page.inner_text("#version") == served
    assert not log.errors
    page.context.close()


def test_log_records_events_and_a_summary_a_second(
    browser: Browser, base_url: str
) -> None:
    """Start, config, camera and per-second summaries with fps land in the log."""
    page, log = open_page(browser, f"{base_url}/phone_web/?screen=g7_internal")
    wait_until(page, "window.abyss.log.some(l => l.includes(' fps'))")
    lines = page.evaluate("window.abyss.log")
    text = "\n".join(lines)
    for expected in ("version ui ", "config camera ", "camera ", "output "):
        assert expected in text, expected
    assert not log.errors
    page.context.close()


def test_log_warns_when_the_frame_is_another_crop(
    browser: Browser, base_url: str
) -> None:
    """The fake camera is 720x1280; the Pixel's focal is for 1440x1920."""
    warnings = {}
    for camera in ("pixel7pro_front", "unknown_clip"):
        page, log = open_page(browser, f"{base_url}/phone_web/?camera={camera}")
        wait_until(page, "window.abyss.status === 'running'")
        lines = page.evaluate("window.abyss.log")
        warnings[camera] = [line for line in lines if " warning: " in line]
        assert not log.errors
        page.context.close()
    assert len(warnings["pixel7pro_front"]) == 1
    assert "1440x1920" in warnings["pixel7pro_front"][0]
    assert "720x1280" in warnings["pixel7pro_front"][0]
    assert warnings["unknown_clip"] == []


def test_log_button_copies_header_and_lines(browser: Browser, base_url: str) -> None:
    """The button puts the version, the URL and the log on the clipboard."""
    page, log = open_page(browser, f"{base_url}/phone_web/?screen=g7_internal")
    page.context.grant_permissions(["clipboard-read", "clipboard-write"])
    wait_until(page, "window.abyss.log.some(l => l.includes(' fps'))")
    page.click("#copy-log")
    wait_until(page, "window.abyss.copied !== undefined")
    copied = page.evaluate("navigator.clipboard.readText()")
    version = page.evaluate("window.abyss.version")
    assert copied.startswith(f"abyss {version}")
    assert "/phone_web/?screen=g7_internal" in copied
    assert page.evaluate("window.abyss.log[0]") in copied
    assert page.evaluate("window.abyss.display.fullscreen") is False
    assert not log.errors
    page.context.close()


def test_log_times_each_load_phase_and_large_file(
    browser: Browser, base_url: str
) -> None:
    """Phases and large downloads are logged, timed from navigation."""
    page, log = open_page(browser, f"{base_url}/phone_web/?screen=g7_internal")
    wait_until(page, "window.abyss.status === 'running'")
    wait_until(page, "window.abyssLoading === null")
    lines = page.evaluate("window.abyss.log")
    text = "\n".join(lines)
    for phase in ("config", "scene", "landmarker", "camera"):
        assert f"loading {phase}" in text, phase
    assert "landmarker ready" in text
    fetched = [line for line in lines if " fetched /" in line]
    for name in ("vision_wasm_internal.wasm", "face_landmarker.task", "three.core.js"):
        assert any(name in line for line in fetched), name
    assert not log.errors
    page.context.close()


def test_large_files_are_preloaded_once(browser: Browser, base_url: str) -> None:
    """The preloads start at page load, and tasks-vision reuses them."""
    page, log = open_page(browser, f"{base_url}/phone_web/?screen=g7_internal")
    wait_until(page, "window.abyss.status === 'running'")
    entries = page.evaluate(
        "performance.getEntriesByType('resource')"
        ".map(e => [new URL(e.name).pathname, e.initiatorType])"
    )
    page.context.close()
    for path in (
        "/vendor/@mediapipe/tasks-vision/wasm/vision_wasm_internal.wasm",
        "/models/face_landmarker.task",
    ):
        assert [kind for name, kind in entries if name == path] == ["link"], path
    assert not log.errors


def test_loading_counter_stops_on_error(
    browser_without_camera: Browser, base_url: str
) -> None:
    """An error stops the counter, so the error stays on the status line."""
    page, _ = open_page(browser_without_camera, f"{base_url}/phone_web/")
    wait_until(page, "window.abyss.status === 'error'")
    page.wait_for_timeout(600)
    assert page.evaluate("window.abyssLoading === null") is True
    assert "Error" in page.inner_text("#status")
    page.context.close()


def test_a_reload_revalidates_instead_of_downloading(
    browser: Browser, base_url: str
) -> None:
    """Second load: the large files come back as 304s, almost nothing on the wire."""
    page, log = open_page(browser, f"{base_url}/phone_web/?screen=g7_internal")
    wait_until(page, "window.abyss.status === 'running'")
    first = [
        line
        for line in page.evaluate("window.abyss.log")
        if "wasm_internal.wasm" in line
    ]
    page.reload()
    wait_until(page, "window.abyss && window.abyss.status === 'running'")
    second = [
        line
        for line in page.evaluate("window.abyss.log")
        if "wasm_internal.wasm" in line
    ]
    page.context.close()
    assert wire_kb(first[0]) < 3600  # gzipped from 11757
    assert wire_kb(second[0]) < 2  # a 304: headers only
    assert " 11757 kB," in second[0]
    assert not log.errors


def wire_kb(line: str) -> int:
    """Read the on-the-wire size out of a ``fetched`` log line."""
    match = re.search(r"(\d+) kB on the wire", line)
    assert match, line
    return int(match.group(1))
