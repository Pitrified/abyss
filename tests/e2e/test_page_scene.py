"""The off-axis projection and the scene, in the page, held to the Python ones.

Four layers, outermost last:

- the frustum port's numbers against ``abyss.render.frustum`` on golden eyes,
  errors included;
- the frame three.js actually draws, read back from the canvas, against points
  Python projects: catches a transposed matrix or a flipped axis inside the
  handoff to three.js, which the numbers alone cannot;
- the mirror hazard: moving the eye to the viewer's right moves the scene the
  way a window does, checked in pixels;
- a clip driving the scene: every frame's matrix is the one Python builds for
  the page's own smoothed eye.
"""

import base64
from collections.abc import Iterator
from pathlib import Path

import cv2 as cv
import numpy as np
from playwright.sync_api import Browser
from playwright.sync_api import Page
import pytest

from abyss.params.abyss_devices import get_screen
from abyss.render.frustum import EyeBehindScreenError
from abyss.render.frustum import frustum_for_eye
from abyss.render.frustum import project_points
from abyss.render.frustum import view_projection_matrix
from abyss.render.renderer import AspectMismatchError
from abyss.render.renderer import check_aspect
from abyss.render.scene import CUBE_BGR
from abyss.render.scene import window_box
from abyss.ui.browser import open_page
from abyss.ui.browser import wait_until
from abyss.ui.page_config import constants

SCREENS = ["g7_internal", "g4_internal"]

EYES = [
    [0.0, 0.1005, 0.6],
    [0.05, -0.03, 0.55],
    [-0.12, 0.2, 0.35],
    [0.3, -0.1, 1.2],
]
"""Camera-frame eyes: on axis for g7, off to each side, close and far."""

RELATIVE = 1e-9
"""Same float64 arithmetic in the same order; only rounding may differ."""

OUTPUT = (1280, 718)
"""Close to g7_internal's 1.7824 aspect, within the 2% the renderer allows."""

LIT_ABOVE_BACKGROUND = 12
"""A pixel this much brighter than the background is a drawn line."""

LIT_FRACTION = 0.97
"""Share of sample points that must land on a lit pixel."""

CLIP_TIMEOUT_MS = 180_000


@pytest.fixture(scope="module")
def page(browser: Browser, base_url: str) -> Iterator[Page]:
    """Open a fixed-eye page, whose modules the cases import."""
    page, _ = open_page(
        browser, f"{base_url}/phone_web/?screen=g7_internal&eye=0,0.1,0.6"
    )
    wait_until(page, "window.abyss.status === 'done'")
    yield page
    page.context.close()


def run_js(page: Page, module: str, body: str, cases: object) -> list:
    """Import a page module and map a body over the cases; errors come back named.

    Args:
        page: The page.
        module: File under ``/phone_web/js/``.
        body: JavaScript taking ``m`` and ``c``.
        cases: JSON-ready cases.

    Returns:
        One result per case, or ``{"error": name}`` where the body threw.
    """
    script = f"async (cases) => {{ const m = await import('/phone_web/js/{module}');"
    script += f" return cases.map((c) => {{ try {{ return {body}; }}"
    script += " catch (e) { return { error: e.name }; } }); }"
    return page.evaluate(script, cases)


def screen_json(name: str) -> dict:
    """Describe a screen as the page receives it."""
    return get_screen(name).model_dump(mode="json")


def test_matrices_match_python(page: Page) -> None:
    """View projection matrices and frustum extents, for every screen and eye."""
    cases = [
        {"screen": screen_json(s), "eye": e, "k": constants()}
        for s in SCREENS
        for e in EYES
    ]
    got = run_js(
        page, "frustum.js", "m.viewProjectionMatrix(c.screen, c.eye, c.k)", cases
    )
    extents = run_js(page, "frustum.js", "m.frustumForEye(c.screen, c.eye, c.k)", cases)
    for case, matrix, frustum in zip(cases, got, extents, strict=True):
        screen = get_screen(case["screen"]["name"])
        want = view_projection_matrix(screen, np.array(case["eye"]))
        assert np.allclose(matrix, want, rtol=RELATIVE, atol=1e-12), case["eye"]
        f = frustum_for_eye(screen, np.array(case["eye"]))
        for side in ("left", "right", "bottom", "top", "near", "far"):
            assert abs(frustum[side] - getattr(f, side)) <= RELATIVE * max(
                1, abs(getattr(f, side))
            )


def test_errors_match_python(page: Page) -> None:
    """An eye behind the panel, and bad clip planes, raise the same named errors."""
    screen = get_screen("g7_internal")
    behind = [0.0, 0.1, -0.2]
    with pytest.raises(EyeBehindScreenError):
        frustum_for_eye(screen, np.array(behind))
    cases = [{"screen": screen_json("g7_internal"), "eye": behind, "k": constants()}]
    (got,) = run_js(page, "frustum.js", "m.frustumForEye(c.screen, c.eye, c.k)", cases)
    assert got == {"error": "EyeBehindScreenError"}
    cases[0]["eye"] = EYES[0]
    (got,) = run_js(
        page, "frustum.js", "m.frustumForEye(c.screen, c.eye, c.k, 1, 0.5)", cases
    )
    assert got == {"error": "InvalidClipPlanesError"}


def test_projected_points_match_python(page: Page) -> None:
    """Every scene vertex lands on the same pixel through either implementation."""
    screen = get_screen("g7_internal")
    points = window_box(screen).segments.reshape(-1, 3)
    cases = []
    for eye in EYES:
        matrix = view_projection_matrix(screen, np.array(eye))
        want = project_points(matrix, points, *OUTPUT)
        cases.append(
            {"m": matrix.tolist(), "p": points.tolist(), "want": want.tolist()}
        )
    got = run_js(
        page,
        "frustum.js",
        f"m.projectPoints(c.m, c.p, {OUTPUT[0]}, {OUTPUT[1]})",
        cases,
    )
    for case, pixels in zip(cases, got, strict=True):
        assert np.allclose(pixels, case["want"], rtol=0, atol=1e-6)
    eye_side = [{"m": cases[0]["m"], "p": [[0.0, 0.1, 0.8]], "want": None}]
    (got,) = run_js(page, "frustum.js", "m.projectPoints(c.m, c.p, 100, 100)", eye_side)
    assert got == {"error": "PointNotInFrontError"}


def test_aspect_check_matches_python(page: Page) -> None:
    """The same sizes pass and fail on both sides."""
    screen = get_screen("g7_internal")
    sizes = [(1280, 720), (1280, 718), (1024, 768), (720, 1280)]
    cases = [
        {"screen": screen_json("g7_internal"), "w": w, "h": h, "k": constants()}
        for w, h in sizes
    ]
    got = run_js(
        page, "frustum.js", "(m.checkAspect(c.screen, c.w, c.h, c.k), 'ok')", cases
    )
    for (w, h), result in zip(sizes, got, strict=True):
        try:
            check_aspect(screen, w, h)
            want = "ok"
        except AspectMismatchError:
            want = {"error": "AspectMismatchError"}
        assert result == want, (w, h)


def render_fixed(browser: Browser, base_url: str, eye: list[float]) -> np.ndarray:
    """Draw one frame for an eye and read it back from the WebGL canvas.

    Args:
        browser: The browser.
        base_url: The server.
        eye: Camera-frame eye, metres.

    Returns:
        The frame, BGR, at :data:`OUTPUT` size.
    """
    eye_text = ",".join(str(v) for v in eye)
    size = f"{OUTPUT[0]}x{OUTPUT[1]}"
    url = f"{base_url}/phone_web/?screen=g7_internal&eye={eye_text}&size={size}"
    page, log = open_page(browser, url)
    wait_until(page, "['done', 'error'].includes(window.abyss.status)")
    assert page.evaluate("window.abyss.status") == "done", page.evaluate(
        "window.abyss.error"
    )
    data_url = page.evaluate("document.getElementById('scene').toDataURL('image/png')")
    page.context.close()
    assert not log.errors
    png = base64.b64decode(data_url.split(",", 1)[1])
    frame = cv.imdecode(np.frombuffer(png, np.uint8), cv.IMREAD_COLOR)
    assert frame is not None
    assert frame.shape[:2] == (OUTPUT[1], OUTPUT[0])
    return frame


@pytest.mark.parametrize("eye", EYES[:3])
def test_drawn_frame_lights_the_projected_segments(
    browser: Browser, base_url: str, eye: list[float]
) -> None:
    """Points Python projects along every segment land on lines three.js drew.

    Samples the inner part of each segment, where neighbouring segments do not
    meet, and allows a pixel of slack for rasterisation.
    """
    frame = render_fixed(browser, base_url, eye)
    brightness = frame.max(axis=2).astype(int)
    background = int(np.median(brightness))
    lit = brightness > background + LIT_ABOVE_BACKGROUND
    lit = cv.dilate(lit.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    screen = get_screen("g7_internal")
    matrix = view_projection_matrix(screen, np.array(eye))
    hits = total = 0
    for segment in window_box(screen).segments:
        for t in np.linspace(0.2, 0.8, 7):
            point = segment[0] * (1 - t) + segment[1] * t
            x, y = project_points(matrix, point, *OUTPUT)[0]
            col, row = round(x), round(y)
            if 2 <= col < OUTPUT[0] - 2 and 2 <= row < OUTPUT[1] - 2:
                total += 1
                hits += bool(lit[row, col])
    assert total > 100
    assert hits / total >= LIT_FRACTION, f"{hits}/{total} sample points on a drawn line"


def cube_centroid_x(frame: np.ndarray) -> float:
    """Return the mean column of the cube's pixels, found by its colour."""
    b, g, r = (frame[..., i].astype(int) for i in range(3))
    cube_b, cube_g, cube_r = CUBE_BGR
    assert cube_r > cube_g > cube_b
    orange = (r > 120) & (r > g + 40) & (g > b + 40)
    assert orange.sum() > 50, "no cube pixels found"
    return float(np.nonzero(orange)[1].mean())


def test_scene_moves_like_a_window(browser: Browser, base_url: str) -> None:
    """The mirror hazard, in pixels: the cube is right of centre, and follows the eye.

    The cube sits at +8 cm in the screen frame, so it is drawn right of centre
    for an eye on axis. Moving the eye to the viewer's right, which is camera
    frame -x for an unmirrored camera, moves something behind the window to
    the right as well. A mirrored handoff fails both.
    """
    centre = [0.0, 0.1005, 0.6]
    viewer_right = [-0.1, 0.1005, 0.6]
    on_axis = cube_centroid_x(render_fixed(browser, base_url, centre))
    shifted = cube_centroid_x(render_fixed(browser, base_url, viewer_right))
    assert on_axis > OUTPUT[0] / 2
    assert shifted > on_axis + 10


@pytest.fixture(scope="module")
def scene_track(browser: Browser, base_url: str, clip: Path) -> list[dict]:
    """Run the clip with the scene drawn for g7's panel."""
    url = f"{base_url}/phone_web/?source=/clips/{clip.name}&camera=unknown_clip"
    url += f"&screen=g7_internal&size={OUTPUT[0]}x{OUTPUT[1]}"
    page, log = open_page(browser, url)
    wait_until(page, "['done', 'error'].includes(window.abyss.status)", CLIP_TIMEOUT_MS)
    assert page.evaluate("window.abyss.status") == "done", page.evaluate(
        "window.abyss.error"
    )
    track = page.evaluate("window.abyss.track")
    page.context.close()
    assert not log.errors
    return track


def test_clip_drives_the_scene_with_python_matrices(scene_track: list[dict]) -> None:
    """Each frame's matrix is Python's for the page's smoothed eye, none before that."""
    screen = get_screen("g7_internal")
    drawn = 0
    for entry in scene_track:
        if entry["smoothed"] is None:
            assert entry["matrix"] is None, entry["idx"]
            continue
        want = view_projection_matrix(screen, np.array(entry["smoothed"]))
        assert np.allclose(entry["matrix"], want, rtol=RELATIVE, atol=1e-12), entry[
            "idx"
        ]
        drawn += 1
    assert drawn >= len(scene_track) - 40
