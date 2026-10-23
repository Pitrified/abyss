"""Fixtures for the browser tests: vendored files, the clip, a server, Chromium.

Everything here is session-scoped and real: the vendor cache is filled from the
network on first use, the clip is regenerated, the server is the one
`make serve` runs, and the browser is Chromium with the clip as its camera.
Every test in this folder carries the ``e2e`` marker, so `make test` skips them.
"""

from collections.abc import Iterator
from pathlib import Path

from playwright.sync_api import Browser
from playwright.sync_api import Page
from playwright.sync_api import Playwright
from playwright.sync_api import sync_playwright
import pytest

from abyss.ui.browser import PageLog
from abyss.ui.browser import open_page
from abyss.ui.browser import start_chromium
from abyss.ui.clip import write_clip
from abyss.ui.serve import ServeConfig
from abyss.ui.serve import running_server
from abyss.ui.vendor import ensure_vendor


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark every test in this folder as e2e."""
    for item in items:
        if Path(str(item.fspath)).parent == Path(__file__).parent:
            item.add_marker(pytest.mark.e2e)


@pytest.fixture(scope="session")
def clip() -> Path:
    """Write the synthetic face clip, after filling the vendor cache."""
    ensure_vendor()
    return write_clip()


@pytest.fixture(scope="session")
def base_url(clip: Path) -> Iterator[str]:
    """Serve the UIs on a free port."""
    with running_server(ServeConfig.default()) as url:
        yield url


@pytest.fixture(scope="session")
def playwright() -> Iterator[Playwright]:
    """One Playwright for the session; it cannot be started twice at once."""
    with sync_playwright() as pw:
        yield pw


@pytest.fixture(scope="session")
def browser(playwright: Playwright, clip: Path) -> Iterator[Browser]:
    """Chromium with the synthetic clip as its camera."""
    chromium = start_chromium(playwright, clip)
    yield chromium
    chromium.close()


@pytest.fixture(scope="session")
def browser_without_camera(playwright: Playwright) -> Iterator[Browser]:
    """Chromium with no fake device, so the camera request fails.

    It fails as no device where the machine has none, and as an unanswerable
    permission prompt where it has a real camera, as g7 does.
    """
    chromium = start_chromium(playwright, None)
    yield chromium
    chromium.close()


@pytest.fixture
def phone_page(browser: Browser, base_url: str) -> Iterator[tuple[Page, PageLog]]:
    """Open the phone UI with the clip as the camera."""
    page, log = open_page(browser, f"{base_url}/phone_web/")
    yield page, log
    page.context.close()
