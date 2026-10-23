"""Launch Chromium for the UI harness, with a clip as the camera.

Chromium has a fake capture device made for this: three flags and
``getUserMedia`` returns the frames of a file, through the same code path a real
webcam takes. ``.mjpeg`` and ``.y4m`` are accepted; the harness uses the MJPEG
written by :mod:`abyss.ui.clip`. The clip loops.

Which Chromium: ``ABYSS_CHROMIUM`` if set, else Playwright's own if it is
installed (``uv run --no-sync playwright install chromium`` on g7), else any
``chromium-*`` under ``PLAYWRIGHT_BROWSERS_PATH``, which is how the cloud
sessions ship one. Playwright's own is tied to its version and the cloud one is
not, so the fallback is what makes the same tests run in both places.

Needs Playwright, from the ``e2e`` dependency group, which ``make sync``
installs. Only the harness imports this module.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
import os
from pathlib import Path
from urllib.parse import urlsplit

from loguru import logger as lg
from playwright.sync_api import Browser
from playwright.sync_api import ConsoleMessage
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page
from playwright.sync_api import Playwright
from playwright.sync_api import Request
from playwright.sync_api import ViewportSize
from playwright.sync_api import sync_playwright

PAGE_TIMEOUT_MS = 30_000
"""Default wait for anything a page is asked to do."""

CHROMIUM_ENV = "ABYSS_CHROMIUM"
BROWSERS_PATH_ENV = "PLAYWRIGHT_BROWSERS_PATH"


class ChromiumNotFoundError(FileNotFoundError):
    """Raised when no Chromium executable can be found."""

    def __init__(self) -> None:
        """Explain the three places that were looked in."""
        super().__init__(
            f"No Chromium found. Set {CHROMIUM_ENV} to an executable, or run "
            "`uv run --no-sync playwright install chromium`."
        )


def fake_camera_args(clip: Path | None) -> list[str]:
    """Chromium flags that replace the real camera.

    Args:
        clip: File to play as the camera. None launches with no fake device,
            so the page sees whatever cameras the machine has, usually none.

    Returns:
        Command-line flags.
    """
    if clip is None:
        return []
    return [
        "--use-fake-ui-for-media-stream",
        "--use-fake-device-for-media-stream",
        f"--use-file-for-fake-video-capture={clip}",
    ]


def find_chromium() -> Path | None:
    """Locate a Chromium executable other than Playwright's own.

    Returns:
        The executable from ``ABYSS_CHROMIUM`` or under
        ``PLAYWRIGHT_BROWSERS_PATH``, newest revision first, or None to let
        Playwright use its own.
    """
    explicit = os.environ.get(CHROMIUM_ENV)
    if explicit:
        return Path(explicit)
    root = os.environ.get(BROWSERS_PATH_ENV)
    if not root:
        return None
    candidates = sorted(
        Path(root).glob("chromium-*/chrome-linux*/chrome"), reverse=True
    )
    return candidates[0] if candidates else None


def start_chromium(
    pw: Playwright, clip: Path | None, *, headless: bool = True
) -> Browser:
    """Start Chromium with a clip as the camera, on a running Playwright.

    Tries Playwright's own browser first and falls back to :func:`find_chromium`
    when it is not installed. Several browsers can share one Playwright, which
    is what the tests do; Playwright itself cannot be started twice at once.

    Args:
        pw: A started Playwright.
        clip: File to play as the camera, or None for no fake device.
        headless: False opens a window, which needs a display.

    Returns:
        The browser. The caller closes it.

    Raises:
        ChromiumNotFoundError: If neither is available.
    """
    args = fake_camera_args(clip)
    if not os.environ.get(CHROMIUM_ENV):
        try:
            return pw.chromium.launch(headless=headless, args=args)
        except PlaywrightError as exc:
            if "Executable doesn't exist" not in str(exc):
                raise
            lg.debug("Playwright's own Chromium is not installed, looking elsewhere")
    path = find_chromium()
    if path is None or not path.is_file():
        raise ChromiumNotFoundError
    lg.debug(f"Using Chromium at {path}")
    return pw.chromium.launch(executable_path=path, headless=headless, args=args)


@contextmanager
def launch_chromium(clip: Path | None, *, headless: bool = True) -> Iterator[Browser]:
    """Start Playwright and one Chromium for a ``with`` block.

    Args:
        clip: File to play as the camera, or None for no fake device.
        headless: False opens a window, which needs a display.

    Yields:
        The browser.
    """
    with sync_playwright() as pw:
        browser = start_chromium(pw, clip, headless=headless)
        try:
            yield browser
        finally:
            browser.close()


@dataclass
class PageLog:
    """What a page did besides render: console lines, errors, requests."""

    console: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    requests: list[str] = field(default_factory=list)

    def off_server(self, base_url: str) -> list[str]:
        """List requests to any host but the server's.

        Args:
            base_url: The server's base URL.

        Returns:
            The offending URLs.
        """
        own = urlsplit(base_url).netloc
        return [u for u in self.requests if urlsplit(u).netloc not in {own, ""}]


def open_page(
    browser: Browser,
    url: str,
    *,
    viewport: ViewportSize | None = None,
    device_scale_factor: float = 1,
) -> tuple[Page, PageLog]:
    """Open a page in a fresh context, recording what it does.

    Args:
        browser: The browser.
        url: Where to go.
        viewport: CSS pixels, or None for Playwright's default.
        device_scale_factor: The page's ``devicePixelRatio``.

    Returns:
        The page and its log.
    """
    log = PageLog()
    context = browser.new_context(
        viewport=viewport, device_scale_factor=device_scale_factor
    )
    page = context.new_page()
    page.set_default_timeout(PAGE_TIMEOUT_MS)

    def on_console(message: ConsoleMessage) -> None:
        log.console.append(f"{message.type}: {message.text}")

    def on_request(request: Request) -> None:
        log.requests.append(request.url)

    page.on("console", on_console)
    page.on("pageerror", lambda exc: log.errors.append(str(exc)))
    page.on("request", on_request)
    page.goto(url)
    return page, log


class PageWaitTimeoutError(TimeoutError):
    """Raised when a page never reaches the state a test waits for."""

    def __init__(self, expression: str, timeout_ms: int, last: object) -> None:
        """Initialise with what was awaited and the last value seen.

        Args:
            expression: The JavaScript expression.
            timeout_ms: How long it was given.
            last: Its last value.
        """
        super().__init__(
            f"{expression!r} not truthy after {timeout_ms} ms; last value {last!r}"
        )


def wait_until(
    page: Page,
    expression: str,
    timeout_ms: int = PAGE_TIMEOUT_MS,
    poll_ms: int = 50,
) -> object:
    """Poll a JavaScript expression from outside the page until it is truthy.

    Playwright's own ``wait_for_function`` evaluates its predicate with
    ``eval`` inside the page, which the server's Content-Security-Policy
    refuses, as it should. ``page.evaluate`` runs through DevTools and is not
    subject to the page's policy, so polling with it keeps the policy strict.

    Args:
        page: The page.
        expression: A JavaScript expression.
        timeout_ms: Give up after this long.
        poll_ms: Wait between evaluations.

    Returns:
        The first truthy value.

    Raises:
        PageWaitTimeoutError: If it never becomes truthy.
    """
    waited = 0
    value: object = None
    while waited <= timeout_ms:
        value = page.evaluate(expression)
        if value:
            return value
        page.wait_for_timeout(poll_ms)
        waited += poll_ms
    raise PageWaitTimeoutError(expression, timeout_ms, value)
