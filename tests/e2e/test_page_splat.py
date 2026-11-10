"""A Gaussian splat in the page, drawn by Spark, held to the Python projection.

The synthetic splat's points are projected with Python and each one's colour is
looked for at that pixel in the frame read back from the canvas. Spark builds
its own splat geometry from the camera, so this is the check that it takes the
camera at the eye and the off-axis projection as given.
"""

import base64

import cv2 as cv
import numpy as np
from playwright.sync_api import Browser
import pytest

from abyss.params.abyss_devices import get_screen
from abyss.render.frustum import project_points
from abyss.render.frustum import view_projection_matrix
from abyss.ui.browser import open_page
from abyss.ui.browser import wait_until
from abyss.ui.splat import TEST_SPLAT_COLOURS_BGR
from abyss.ui.splat import TEST_SPLAT_NAME
from abyss.ui.splat import TEST_SPLAT_POINTS

OUTPUT = (1280, 718)
"""Close to g7_internal's aspect, as in the scene tests."""

EYES = [
    [0.0, 0.1005, 0.6],
    [0.05, -0.03, 0.55],
    [-0.12, 0.2, 0.35],
]
"""Camera-frame eyes: on axis, off to one side, close and off to the other."""

SETTLED_FRAMES = 30
"""Redraws to wait after loading, so Spark's sort for this view has landed."""

SEARCH_PX = 40
"""How far from the projected point a colour's pixels are looked for."""

TOLERANCE_PX = 2.0
"""How far the drawn disc's centre may sit from the projected point."""

SATURATED = 200
"""A channel at least this bright is on."""

DARK = 80
"""A channel at most this bright is off."""

SPLAT_TIMEOUT_MS = 60_000


def render_splat(
    browser: Browser, base_url: str, eye: list[float], keep: float = 1
) -> tuple[np.ndarray, dict]:
    """Draw the test splat for an eye, wait for it to settle, read it back.

    Args:
        browser: The browser.
        base_url: The server.
        eye: Camera-frame eye, metres.
        keep: Share of the splats to keep.

    Returns:
        The frame, BGR, at :data:`OUTPUT` size, and ``window.abyss.splat``.
    """
    eye_text = ",".join(str(v) for v in eye)
    url = f"{base_url}/phone_web/?screen=g7_internal&eye={eye_text}"
    url += f"&size={OUTPUT[0]}x{OUTPUT[1]}&splat={TEST_SPLAT_NAME}&keep={keep}"
    page, log = open_page(browser, url)
    wait_until(
        page,
        "window.abyss.status === 'error'"
        f" || window.abyss.splat?.drawn >= {SETTLED_FRAMES}",
        SPLAT_TIMEOUT_MS,
    )
    assert page.evaluate("window.abyss.status") == "done", page.evaluate(
        "window.abyss.error"
    )
    splat = page.evaluate("window.abyss.splat")
    data_url = page.evaluate("document.getElementById('scene').toDataURL('image/png')")
    page.context.close()
    assert not log.errors, log.errors
    png = base64.b64decode(data_url.split(",", 1)[1])
    frame = cv.imdecode(np.frombuffer(png, np.uint8), cv.IMREAD_COLOR)
    assert frame is not None
    assert frame.shape[:2] == (OUTPUT[1], OUTPUT[0])
    return frame, splat


def colour_mask(frame: np.ndarray, colour_bgr: np.ndarray) -> np.ndarray:
    """Pixels whose channels are on where the colour's are, and off elsewhere."""
    on = colour_bgr > 0
    mask = np.ones(frame.shape[:2], dtype=bool)
    for channel in range(3):
        values = frame[..., channel]
        mask &= values >= SATURATED if on[channel] else values <= DARK
    return mask


def disc_centre(
    frame: np.ndarray, colour_bgr: np.ndarray, near: np.ndarray
) -> np.ndarray:
    """Return the centroid of a colour's pixels around a point, as ``(x, y)``."""
    mask = colour_mask(frame, colour_bgr)
    x, y = (round(v) for v in near)
    rows = slice(max(0, y - SEARCH_PX), y + SEARCH_PX + 1)
    cols = slice(max(0, x - SEARCH_PX), x + SEARCH_PX + 1)
    found_rows, found_cols = np.nonzero(mask[rows, cols])
    assert len(found_rows) > 20, f"no {colour_bgr.tolist()} disc near {near.tolist()}"
    return np.array([found_cols.mean() + cols.start, found_rows.mean() + rows.start])


@pytest.mark.parametrize("eye", EYES)
def test_splat_points_land_where_python_projects_them(
    browser: Browser, base_url: str, eye: list[float]
) -> None:
    """Every Gaussian's disc is centred on its Python-projected pixel."""
    frame, splat = render_splat(browser, base_url, eye)
    assert splat["count"] == len(TEST_SPLAT_POINTS)
    matrix = view_projection_matrix(get_screen("g7_internal"), np.array(eye))
    want = project_points(matrix, TEST_SPLAT_POINTS, *OUTPUT)
    for colour, pixel in zip(TEST_SPLAT_COLOURS_BGR, want, strict=True):
        got = disc_centre(frame, colour, pixel)
        assert np.linalg.norm(got - pixel) <= TOLERANCE_PX, (
            colour.tolist(),
            got,
            pixel,
        )


def test_keep_thins_the_splat_evenly(browser: Browser, base_url: str) -> None:
    """keep=0.6 of five splats keeps three, and drawing still settles."""
    _, splat = render_splat(browser, base_url, EYES[0], keep=0.6)
    assert splat["count"] == 3


SWEEP_TIMEOUT_MS = 90_000


def test_sweep_logs_a_tier_per_count(browser: Browser, base_url: str) -> None:
    """On the fake camera, a sweep logs one row per share, then the table."""
    url = f"{base_url}/phone_web/?screen=g7_internal&splat={TEST_SPLAT_NAME}"
    page, log = open_page(browser, f"{url}&sweep=0.6,1")
    wait_until(
        page, "window.abyss.status === 'error' || window.abyss.sweep", SWEEP_TIMEOUT_MS
    )
    assert page.evaluate("window.abyss.error") is None
    rows = page.evaluate("window.abyss.sweep")
    lines = page.evaluate("window.abyss.log")
    page.context.close()
    assert not log.errors, log.errors
    assert [(r["keep"], r["count"]) for r in rows] == [(0.6, 3), (1, 5)]
    for row in rows:
        assert row["fps"] > 0
        assert row["tier"] in {1, 10, 30}
    done = next(i for i, line in enumerate(lines) if line.endswith(" sweep done"))
    assert len(lines) >= done + 1 + len(rows)
    assert "keep 0.6 3 splats" in lines[done + 1]
