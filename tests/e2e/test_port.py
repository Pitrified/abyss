"""The page's port, run on golden inputs and compared with the Python it ports.

Inputs are built here, the expected outputs come from calling the real Python
functions on them at test time, and the JavaScript runs in the page on the same
inputs. Nothing is committed, so the cases cannot go stale. These reach the
branches the clip never does: a mirrored camera, a measured focal length, a
viewer turned away, a frame with no face.
"""

from collections.abc import Iterator
from dataclasses import asdict
import math
from pathlib import Path

from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerResult
import numpy as np
from playwright.sync_api import Browser
from playwright.sync_api import Page
import pytest

from abyss.config.camera import CameraConfig
from abyss.config.camera import FrameGeometry
from abyss.params.abyss_devices import get_camera
from abyss.params.abyss_devices import get_viewer
from abyss.ui.browser import open_page
from abyss.ui.browser import wait_until
from abyss.ui.clip import split_mjpeg
from abyss.ui.page_config import constants
from abyss.viewer.eye_position import EyeSample
from abyss.viewer.eye_position import LiveScale
from abyss.viewer.eye_position import estimate_head_scale
from abyss.viewer.eye_position import extract_eye_sample
from abyss.viewer.eye_position import eye_position_m
from abyss.viewer.smoothing import DEFAULT_FILTER_SIZE
from abyss.viewer.smoothing import PositionSmoother

EXACT = 1e-9
"""Float64 on both sides and the same operations, so only rounding differs."""

N_LANDMARKS = 478

CAMERAS = [
    get_camera("g7_webcam"),
    get_camera("unknown_clip"),
    get_camera("pixel7pro_front"),
    CameraConfig(name="fov50", fov_vertical_deg=50.0),
]
"""Measured, fallback, mirrored, and a published field of view."""


@pytest.fixture(scope="module")
def page(browser: Browser, base_url: str) -> Iterator[Page]:
    """Open the phone page, whose modules the cases import."""
    page, _ = open_page(browser, f"{base_url}/phone_web/?camera=unknown_clip")
    wait_until(page, "window.abyss.status !== 'starting'")
    yield page
    page.context.close()


def call(page: Page, module: str, body: str, cases: object) -> list:
    """Import a page module and map a function body over the cases.

    Args:
        page: The page.
        module: File under ``/phone_web/js/``.
        body: JavaScript taking ``m`` (the module) and ``c`` (one case).
        cases: JSON-ready cases.

    Returns:
        One result per case.
    """
    script = f"async (cases) => {{ const m = await import('/phone_web/js/{module}');"
    script += f" return cases.map((c) => {body}); }}"
    return page.evaluate(script, cases)


def geometry_json(geometry: FrameGeometry) -> dict:
    """Describe a geometry the way the page's ``frameGeometry`` builds it."""
    return {
        "width": geometry.width,
        "height": geometry.height,
        "focal": geometry.focal,
        "principalPoint": list(geometry.principal_point),
        "mirrored": geometry.mirrored,
    }


def sample(
    idx: int, u: float, v: float, ipd: float, depth: float, yaw: float
) -> EyeSample:
    """Build a sample with a timestamp derived from its index."""
    return EyeSample(idx, idx * 33.3, u, v, ipd, depth, yaw)


def test_focal_length_matches(page: Page) -> None:
    """Measured, assumed and published intrinsics give Python's focal length."""
    cases = [
        {"camera": cam.model_dump(), "height": h, "want": cam.focal_px_for_height(h)}
        for cam in CAMERAS
        for h in (480, 720, 1080, 1280)
    ]
    got = call(
        page,
        "geometry.js",
        "m.focalPxForHeight(c.camera, c.height, c.constants)",
        [{**c, "constants": constants()} for c in cases],
    )
    for case, value in zip(cases, got, strict=True):
        assert abs(value - case["want"]) < EXACT * case["want"], case


def test_eye_position_matches(page: Page) -> None:
    """Including the mirrored camera, which flips x and nothing else."""
    cases = []
    for cam in CAMERAS:
        geometry = FrameGeometry(cam, 720, 1280)
        for s in (
            sample(0, 100.0, 900.0, 120.0, 0.45, 2.0),
            sample(1, 600.0, 300.0, 90.0, 0.7, -5.0),
        ):
            for scale in (1.0, 0.87):
                want = eye_position_m(s, geometry, scale).tolist()
                cases.append(
                    {
                        "s": asdict(s),
                        "g": geometry_json(geometry),
                        "k": scale,
                        "want": want,
                    }
                )
    got = call(page, "eye_position.js", "m.eyePositionM(c.s, c.g, c.k)", cases)
    for case, value in zip(cases, got, strict=True):
        assert np.allclose(value, case["want"], rtol=0, atol=EXACT), case


def test_head_scale_matches(page: Page) -> None:
    """Turned-away frames are dropped; an even count averages the middle two."""
    geometry = FrameGeometry(get_camera("g7_webcam"), 1280, 720)
    viewer = get_viewer()
    batches = [
        [
            sample(i, 640, 360, 100 + i, 0.5 + 0.01 * i, yaw)
            for i, yaw in enumerate([0, 3, -8, 25, -40, 9])
        ],
        [sample(i, 640, 360, 95 + 2 * i, 0.6, 1.0) for i in range(5)],
        [sample(i, 640, 360, 100, 0.5, 30.0) for i in range(4)],
    ]
    cases = [
        {
            "samples": [asdict(s) for s in batch],
            "g": geometry_json(geometry),
            "viewer": viewer.model_dump(),
            "want": estimate_head_scale(batch, geometry, viewer),
        }
        for batch in batches
    ]
    got = call(
        page,
        "eye_position.js",
        "m.estimateHeadScale(c.samples, c.g, c.viewer, c.constants)",
        [{**c, "constants": constants()} for c in cases],
    )
    for case, value in zip(cases, got, strict=True):
        assert abs(value - case["want"]) < EXACT, case
    assert cases[-1]["want"] == 1.0


def test_live_scale_matches(page: Page) -> None:
    """The bootstrap skips turned-away frames, freezes once, then ignores the rest."""
    geometry = FrameGeometry(get_camera("g7_webcam"), 1280, 720)
    viewer = get_viewer()
    stream = [
        sample(i, 640, 360, 100 + (i % 3), 0.5, 20.0 if i % 4 == 0 else 1.0)
        for i in range(12)
    ]
    python = LiveScale(geometry, viewer, needed=5)
    want = []
    for s in stream:
        python.update(s)
        want.append(
            [
                python.is_ready,
                list(python.progress),
                python.scale if python.is_ready else None,
            ]
        )
    body = """(() => {
        const live = new m.LiveScale(c.g, c.viewer, c.constants, { needed: 5 });
        return c.stream.map((s) => {
            live.update(s);
            return [live.isReady, live.progress, live.isReady ? live.scale : null];
        });
    })()"""
    case = {
        "g": geometry_json(geometry),
        "viewer": viewer.model_dump(),
        "constants": constants(),
        "stream": [asdict(s) for s in stream],
    }
    (got,) = call(page, "eye_position.js", body, [case])
    for step, (g, w) in enumerate(zip(got, want, strict=True)):
        assert g[:2] == w[:2], step
        assert (g[2] is None) == (w[2] is None), step
        if w[2] is not None:
            assert abs(g[2] - w[2]) < EXACT, step


def test_smoother_matches(page: Page) -> None:
    """Primed on the first sample, triangle-weighted, and holding across gaps."""
    rng = np.random.default_rng(7)
    stream: list[list[float] | None] = [
        rng.normal(0, 0.05, 3).tolist() for _ in range(9)
    ]
    stream[4] = None
    stream[5] = None
    smoother = PositionSmoother(DEFAULT_FILTER_SIZE)
    want = []
    for position in stream:
        if position is None:
            held = smoother.hold()
            want.append(None if held is None else held.tolist())
        else:
            want.append(smoother.update(np.array(position)).tolist())
    body = """(() => {
        const sm = new m.PositionSmoother(c.size);
        return c.stream.map((p) => (p === null ? sm.hold() : sm.update(p)));
    })()"""
    (got,) = call(
        page,
        "smoothing.js",
        body,
        [{"size": constants()["filter_size"], "stream": stream}],
    )
    for step, (g, w) in enumerate(zip(got, want, strict=True)):
        assert np.allclose(g, w, rtol=0, atol=EXACT), step


def rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Build a head rotation: yaw about y, then pitch about x."""
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    ry = np.array(
        [[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]]
    )
    rx = np.array(
        [[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]]
    )
    return ry @ rx


def test_extract_matches(page: Page) -> None:
    """A synthetic landmarker result: iris pixels, depth through the rotation, yaw.

    The page receives the matrix column-major, as tasks-vision delivers it;
    Python receives the 4x4 array. A row/column mix-up fails here on yaw and
    depth, with the clip's agreement as the second witness.
    """
    rng = np.random.default_rng(3)
    cases = []
    for yaw, pitch in ((0.0, 0.0), (20.0, 5.0), (-35.0, -10.0)):
        matrix = np.eye(4)
        matrix[:3, :3] = rotation(yaw, pitch)
        matrix[:3, 3] = [1.5, -2.0, -48.0]
        points = rng.uniform(0.2, 0.8, (N_LANDMARKS, 2))
        result = FaceLandmarkerResult(
            face_landmarks=[
                [NormalizedLandmark(x=float(x), y=float(y), z=0.0) for x, y in points]
            ],
            face_blendshapes=[],
            facial_transformation_matrixes=[matrix],
        )
        geometry = FrameGeometry(get_camera("unknown_clip"), 720, 1280)
        want = extract_eye_sample(result, geometry, 4, 133.0)
        assert want is not None
        cases.append(
            {
                "result": {
                    "faceLandmarks": [
                        [{"x": float(x), "y": float(y)} for x, y in points]
                    ],
                    "facialTransformationMatrixes": [
                        {
                            "rows": 4,
                            "columns": 4,
                            "data": matrix.flatten(order="F").tolist(),
                        }
                    ],
                },
                "g": geometry_json(geometry),
                "want": asdict(want),
            }
        )
    cases.append(
        {
            "result": {"faceLandmarks": [], "facialTransformationMatrixes": []},
            "g": cases[0]["g"],
            "want": None,
        }
    )
    got = call(
        page,
        "eye_position.js",
        "m.extractEyeSample(c.result, c.g, 4, 133.0, c.constants)",
        [{**c, "constants": constants()} for c in cases],
    )
    for case, value in zip(cases, got, strict=True):
        if case["want"] is None:
            assert value is None
            continue
        for field, expected in case["want"].items():
            assert abs(value[field] - expected) < 1e-6, (field, value[field], expected)


def test_mjpeg_split_matches(page: Page, clip: Path) -> None:
    """The page splits the clip into the same frames Python does."""
    want = [len(frame) for frame in split_mjpeg(clip.read_bytes())]
    got = page.evaluate(
        f"""async () => {{
            const m = await import('/phone_web/js/clip_source.js');
            return (await m.loadClip('/clips/{clip.name}')).map((f) => f.length);
        }}"""
    )
    assert got == want
