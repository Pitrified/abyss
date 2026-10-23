"""The version label and the log a person copies off the phone."""

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
