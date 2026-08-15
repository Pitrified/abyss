"""Test the eye position extraction and conversion.

No test here reads a clip or a model file: neither is in git. The MediaPipe
results are faked, and the fixture in ``data/face01_samples.json`` carries 20
real frames so the whole conversion is still exercised on real numbers.
"""

import json
import math
from pathlib import Path
from typing import NamedTuple
from typing import cast

from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerResult
import numpy as np
import pytest

from abyss.config.camera import CameraConfig
from abyss.config.camera import FrameGeometry
from abyss.config.viewer import DEFAULT_IPD_M
from abyss.config.viewer import ViewerConfig
from abyss.viewer.eye_position import EYE_IN_MODEL_CM
from abyss.viewer.eye_position import EyeSample
from abyss.viewer.eye_position import estimate_head_scale
from abyss.viewer.eye_position import extract_eye_sample
from abyss.viewer.eye_position import eye_position_m

FIXTURE = Path(__file__).parent / "data" / "face01_samples.json"


class FakeLandmark(NamedTuple):
    """A normalized landmark, as MediaPipe reports it."""

    x: float
    y: float
    z: float = 0.0


class FakeResult:
    """Stand-in for a FaceLandmarkerResult.

    ``get_landmarks_from_result`` dispatches on the real type, so the fixtures
    below patch the two accessors instead of subclassing MediaPipe's dataclass.
    """

    def __init__(self, landmarks: list | None, matrix: np.ndarray | None) -> None:
        self.landmarks = landmarks
        self.matrix = matrix


@pytest.fixture(autouse=True)
def _patch_accessors(monkeypatch) -> None:
    """Route the pose-tools accessors at the fake result."""
    from abyss.viewer import eye_position

    monkeypatch.setattr(
        eye_position, "get_landmarks_from_result", lambda r, _which: r.landmarks
    )
    monkeypatch.setattr(
        eye_position, "get_facial_transformation_matrix", lambda r: r.matrix
    )


def as_result(
    landmarks: list | None,
    matrix: np.ndarray | None,
) -> FaceLandmarkerResult:
    """Present a fake result as the type the function under test declares."""
    return cast("FaceLandmarkerResult", FakeResult(landmarks, matrix))


def make_landmarks(right_uv: tuple, left_uv: tuple, size: tuple) -> list:
    """Build a landmark list with the two iris centres placed in pixels."""
    width, height = size
    landmarks = [FakeLandmark(0.0, 0.0)] * 478
    landmarks[468] = FakeLandmark(right_uv[0] / width, right_uv[1] / height)
    landmarks[473] = FakeLandmark(left_uv[0] / width, left_uv[1] / height)
    return landmarks


def identity_matrix(depth_cm: float) -> np.ndarray:
    """Build a pose facing the camera, model origin at *depth_cm*."""
    matrix = np.eye(4)
    # MediaPipe looks down -Z, and the eye offset is applied through R.
    matrix[:3, 3] = [0.0, 0.0, -depth_cm - EYE_IN_MODEL_CM[2]]
    return matrix


def geometry(
    width: int = 1920,
    height: int = 1080,
    focal_px: float | None = None,
    *,
    mirrored: bool = False,
) -> FrameGeometry:
    """Bind a camera to a frame size, the way the pipeline does at run time."""
    camera = CameraConfig(
        name="test",
        focal_px=focal_px,
        focal_measured_at_height=height if focal_px is not None else None,
        mirrored=mirrored,
    )
    return FrameGeometry(camera=camera, width=width, height=height)


CAMERA = geometry()
VIEWER = ViewerConfig(name="test")


class TestExtractEyeSample:
    """Pulling measurements out of a result."""

    def test_no_face_yields_nothing(self) -> None:
        assert extract_eye_sample(as_result(None, None), CAMERA, 0, 0.0) is None

    def test_no_matrix_yields_nothing(self) -> None:
        landmarks = make_landmarks((900, 400), (1020, 400), (1920, 1080))
        result = as_result(landmarks, None)
        assert extract_eye_sample(result, CAMERA, 0, 0.0) is None

    def test_iris_midpoint_and_ipd(self) -> None:
        landmarks = make_landmarks((900, 400), (1020, 440), (1920, 1080))
        result = as_result(landmarks, identity_matrix(50.0))
        sample = extract_eye_sample(result, CAMERA, 7, 280.0)
        assert sample is not None
        assert sample.u_px == pytest.approx(960.0)
        assert sample.v_px == pytest.approx(420.0)
        assert sample.ipd_px == pytest.approx(math.hypot(120, 40))
        assert sample.idx == 7
        assert sample.msec == 280.0

    def test_depth_is_metres_from_centimetres(self) -> None:
        landmarks = make_landmarks((900, 400), (1020, 400), (1920, 1080))
        result = as_result(landmarks, identity_matrix(50.0))
        sample = extract_eye_sample(result, CAMERA, 0, 0.0)
        assert sample is not None
        assert sample.depth_m == pytest.approx(0.50)

    def test_depth_uses_the_eye_not_the_model_origin(self) -> None:
        # The matrix places the origin 53 cm away; the eye sits 3 cm in front.
        matrix = np.eye(4)
        matrix[:3, 3] = [0.0, 0.0, -53.0]
        landmarks = make_landmarks((900, 400), (1020, 400), (1920, 1080))
        sample = extract_eye_sample(as_result(landmarks, matrix), CAMERA, 0, 0.0)
        assert sample is not None
        assert sample.depth_m == pytest.approx(0.50)

    def test_yaw_comes_from_the_rotation(self) -> None:
        angle = math.radians(30)
        matrix = np.eye(4)
        matrix[:3, :3] = [
            [math.cos(angle), 0, math.sin(angle)],
            [0, 1, 0],
            [-math.sin(angle), 0, math.cos(angle)],
        ]
        matrix[:3, 3] = [0.0, 0.0, -50.0]
        landmarks = make_landmarks((900, 400), (1020, 400), (1920, 1080))
        sample = extract_eye_sample(as_result(landmarks, matrix), CAMERA, 0, 0.0)
        assert sample is not None
        assert sample.yaw_deg == pytest.approx(30.0)


class TestEyePositionM:
    """Converting a sample into a metric position."""

    def sample(self, u: float = 960.0, v: float = 540.0) -> EyeSample:
        return EyeSample(
            idx=0, msec=0.0, u_px=u, v_px=v, ipd_px=120.0, depth_m=0.5, yaw_deg=0.0
        )

    def test_centred_eye_is_on_the_axis(self) -> None:
        position = eye_position_m(self.sample(), CAMERA)
        assert position[0] == pytest.approx(0.0)
        assert position[1] == pytest.approx(0.0)
        assert position[2] == pytest.approx(0.5)

    def test_right_of_centre_is_positive_x(self) -> None:
        position = eye_position_m(self.sample(u=1160.0), CAMERA)
        assert position[0] > 0

    def test_below_centre_is_positive_y(self) -> None:
        # +Y points down the image, the OpenCV convention.
        position = eye_position_m(self.sample(v=740.0), CAMERA)
        assert position[1] > 0

    def test_pinhole_conversion_by_hand(self) -> None:
        camera = geometry(focal_px=1000.0)
        position = eye_position_m(self.sample(u=1160.0, v=740.0), camera)
        assert position[0] == pytest.approx((1160 - 960) * 0.5 / 1000)
        assert position[1] == pytest.approx((740 - 540) * 0.5 / 1000)

    def test_mirrored_flips_x_only(self) -> None:
        plain = eye_position_m(self.sample(u=1160.0, v=740.0), CAMERA)
        mirrored_camera = geometry(mirrored=True)
        mirrored = eye_position_m(self.sample(u=1160.0, v=740.0), mirrored_camera)
        assert mirrored[0] == pytest.approx(-plain[0])
        assert mirrored[1] == pytest.approx(plain[1])
        assert mirrored[2] == pytest.approx(plain[2])

    def test_focal_scales_depth_only(self) -> None:
        near = geometry(focal_px=1000.0)
        far = geometry(focal_px=2000.0)
        sample = self.sample(u=1160.0, v=740.0)
        # Doubling the true focal doubles the depth the same sample implies,
        # which the caller supplies through head_scale in the real pipeline.
        a = eye_position_m(sample, near)
        b = eye_position_m(sample, far)
        assert b[2] == pytest.approx(a[2])  # depth comes from the sample
        assert b[0] == pytest.approx(a[0] / 2)  # lateral halves with focal

    def test_head_scale_scales_all_three_axes(self) -> None:
        sample = self.sample(u=1160.0, v=740.0)
        plain = eye_position_m(sample, CAMERA, head_scale=1.0)
        scaled = eye_position_m(sample, CAMERA, head_scale=2.0)
        assert np.allclose(scaled, plain * 2)


class TestEstimateHeadScale:
    """The per-viewer identity factor."""

    def make(self, ipd_px: float, depth_m: float, yaw: float) -> EyeSample:
        return EyeSample(
            idx=0,
            msec=0.0,
            u_px=960.0,
            v_px=540.0,
            ipd_px=ipd_px,
            depth_m=depth_m,
            yaw_deg=yaw,
        )

    def test_matches_the_configured_ipd(self) -> None:
        camera = geometry(focal_px=1000.0)
        # An implied IPD of 0.07 m against a configured 0.063 m.
        samples = [self.make(ipd_px=140.0, depth_m=0.5, yaw=0.0) for _ in range(5)]
        scale = estimate_head_scale(samples, camera, VIEWER)
        assert scale == pytest.approx(DEFAULT_IPD_M / 0.07)

    def test_ignores_turned_heads(self) -> None:
        camera = geometry(focal_px=1000.0)
        front = [self.make(ipd_px=140.0, depth_m=0.5, yaw=2.0) for _ in range(3)]
        turned = [self.make(ipd_px=90.0, depth_m=0.5, yaw=40.0) for _ in range(9)]
        assert estimate_head_scale(front + turned, camera, VIEWER) == pytest.approx(
            DEFAULT_IPD_M / 0.07
        )

    def test_no_front_facing_frame_leaves_the_scale_alone(self) -> None:
        camera = geometry(focal_px=1000.0)
        turned = [self.make(ipd_px=90.0, depth_m=0.5, yaw=40.0) for _ in range(4)]
        assert estimate_head_scale(turned, camera, VIEWER) == 1.0


class TestAgainstRealFrames:
    """The fixture: 20 real frames, no clip and no model needed."""

    def load(self) -> tuple[list[EyeSample], FrameGeometry]:
        doc = json.loads(FIXTURE.read_text())
        camera = geometry(width=doc["width"], height=doc["height"])
        samples = []
        for row in doc["frames"]:
            right, left = row["right_iris"], row["left_iris"]
            landmarks = [FakeLandmark(0.0, 0.0)] * 478
            landmarks[468] = FakeLandmark(*right)
            landmarks[473] = FakeLandmark(*left)
            result = as_result(landmarks, np.array(row["matrix"]))
            sample = extract_eye_sample(result, camera, row["idx"], row["msec"])
            assert sample is not None
            samples.append(sample)
        return samples, camera

    def test_every_fixture_frame_converts(self) -> None:
        samples, _ = self.load()
        assert len(samples) == 20

    def test_depth_is_plausible_for_an_interview_shot(self) -> None:
        samples, _ = self.load()
        depths = np.array([s.depth_m for s in samples])
        assert depths.min() > 0.3
        assert depths.max() < 0.8

    def test_head_scale_lands_near_the_measured_value(self) -> None:
        # face01's subject implies ~70 mm of IPD against a 63 mm default, so
        # the correction is a shrink of roughly 10%.
        samples, camera = self.load()
        scale = estimate_head_scale(samples, camera, VIEWER)
        assert 0.85 < scale < 0.95

    def test_positions_stay_in_front_of_the_camera(self) -> None:
        samples, camera = self.load()
        scale = estimate_head_scale(samples, camera, VIEWER)
        for sample in samples:
            position = eye_position_m(sample, camera, scale)
            assert position[2] > 0
            assert abs(position[0]) < 0.3
            assert abs(position[1]) < 0.3
