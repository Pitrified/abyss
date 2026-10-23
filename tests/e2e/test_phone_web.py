"""The phone UI in Chromium, with the synthetic clip as its camera."""

from pathlib import Path

from playwright.sync_api import Browser
from playwright.sync_api import Page

from abyss.ui.browser import PageLog
from abyss.ui.browser import open_page
from abyss.ui.browser import wait_until
from abyss.ui.clip import FRAME_HEIGHT
from abyss.ui.clip import FRAME_WIDTH


def wait_for_status(page: Page, status: str) -> None:
    """Wait until the page reports a status."""
    wait_until(page, f"window.abyss && window.abyss.status === {status!r}")


def test_fake_camera_reaches_the_page(
    phone_page: tuple[Page, PageLog], clip: Path
) -> None:
    """The page opens the clip as its camera, at the clip's size, and frames arrive."""
    page, log = phone_page
    wait_for_status(page, "running")
    camera = page.evaluate("window.abyss.camera")
    assert camera["label"].endswith(clip.name)
    assert (camera["width"], camera["height"]) == (FRAME_WIDTH, FRAME_HEIGHT)
    first = page.evaluate("window.abyss.frames")
    wait_until(page, f"window.abyss.frames > {first + 10}")
    assert not log.errors


def test_vendored_modules_load_in_the_page(phone_page: tuple[Page, PageLog]) -> None:
    """three.js and tasks-vision import from the server under the page's CSP."""
    page, log = phone_page
    wait_for_status(page, "running")
    revision = page.evaluate(
        "import('/vendor/three/build/three.module.js').then(m => m.REVISION)"
    )
    assert revision == "186"
    exports = page.evaluate(
        "import('/vendor/@mediapipe/tasks-vision/vision_bundle.mjs')"
        ".then(m => Object.keys(m))"
    )
    assert {"FilesetResolver", "FaceLandmarker"} <= set(exports)
    assert not log.errors


def test_page_makes_no_requests_off_the_server(
    phone_page: tuple[Page, PageLog], base_url: str
) -> None:
    """Nothing the page loads comes from anywhere but the server."""
    page, log = phone_page
    wait_for_status(page, "running")
    page.wait_for_timeout(500)
    assert log.requests
    assert log.off_server(base_url) == []


def test_no_camera_is_reported_not_hung(
    browser_without_camera: Browser, base_url: str
) -> None:
    """With no camera the page says so on window.abyss instead of waiting forever."""
    page, _ = open_page(browser_without_camera, f"{base_url}/phone_web/")
    wait_for_status(page, "error")
    error = page.evaluate("window.abyss.error")
    # NotSupportedError is headless Chromium on a machine with a real camera:
    # the permission prompt cannot be shown, so the request is refused as such.
    assert error["name"] in {
        "NotFoundError",
        "NotAllowedError",
        "NotReadableError",
        "NotSupportedError",
    }
    page.context.close()
