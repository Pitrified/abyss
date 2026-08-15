"""Eye position in the camera frame, from a face landmarker result.

Frame convention, which every consumer downstream depends on:

- origin at the camera's optical centre
- ``+X`` to the right of the image, ``+Y`` down the image, ``+Z`` away from the
  camera (the OpenCV convention)
- **metres**

MediaPipe works in centimetres and its transformation matrix is y-up, so both
are converted here, once, and never carried further.

The depth comes from the facial transformation matrix rather than from apparent
interpupillary distance: the latter correlates -0.76 with absolute yaw, so it
fails exactly when the viewer turns their head. The matrix locates the model
origin rather than the eye, which is a yaw-coupled centimetre of error, so the
eye offset is applied through the rotation - see
`plans/01_abyss_expansion/01_viewer_position.md`.
"""

from dataclasses import dataclass
import math

from loguru import logger as lg
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerResult
import numpy as np
from pose_tools.utils.mediapipe import FACE_LEFT_IRIS_CENTER
from pose_tools.utils.mediapipe import FACE_RIGHT_IRIS_CENTER
from pose_tools.utils.mediapipe import get_facial_transformation_matrix
from pose_tools.utils.mediapipe import get_landmarks_from_result

from abyss.viewer.camera import CameraAssumptions

CM_TO_M = 0.01

EYE_IN_MODEL_CM = np.array([0.0, 2.5, 3.0])
"""Eye position in MediaPipe's canonical model frame, in centimetres.

Fitted by reprojecting the model pose onto the measured iris pixels over
`face01.mp4`: 2.5 cm above and 3.0 cm in front of the model origin, with a
residual of 2.5 px. Without it the reported depth is that of the head origin,
which sits about 2.7 cm behind the eye and swings by a centimetre with yaw.
"""

FRONT_FACING_YAW_DEG = 10.0
"""Yaw below which a frame is usable for estimating the viewer's head scale."""


@dataclass(frozen=True)
class EyeSample:
    """One frame's worth of raw eye measurements, before scaling.

    Args:
        idx: Frame index.
        msec: Frame timestamp in milliseconds.
        u_px: Iris midpoint, horizontal, in pixels.
        v_px: Iris midpoint, vertical, in pixels.
        ipd_px: Apparent interpupillary distance in pixels.
        depth_m: Eye depth from the camera, in metres, before identity scaling.
        yaw_deg: Head yaw in degrees, used to gate the scale estimate.
    """

    idx: int
    msec: float
    u_px: float
    v_px: float
    ipd_px: float
    depth_m: float
    yaw_deg: float


def extract_eye_sample(
    result: FaceLandmarkerResult,
    camera: CameraAssumptions,
    idx: int,
    msec: float,
) -> EyeSample | None:
    """Pull the raw eye measurements out of a landmarker result.

    Args:
        result: A face landmarker result, from a landmarker built with
            ``output_facial_transformation_matrixes=True``.
        camera: Frame geometry the landmarks are expressed in.
        idx: Frame index, carried through to the sample.
        msec: Frame timestamp in milliseconds, carried through.

    Returns:
        The sample, or ``None`` when the frame has no face or no matrix. A
        frame without a face yields nothing rather than a guess.
    """
    landmarks = get_landmarks_from_result(result, "normalized")
    matrix = get_facial_transformation_matrix(result)
    if landmarks is None or matrix is None:
        return None

    right = landmarks[FACE_RIGHT_IRIS_CENTER]
    left = landmarks[FACE_LEFT_IRIS_CENTER]
    # MediaPipe types landmark coordinates as optional. A landmark without them
    # is not a measurement, so the frame yields nothing.
    if right.x is None or right.y is None or left.x is None or left.y is None:
        lg.warning(f"Frame {idx}: iris landmark without coordinates")
        return None
    right_px = np.array([right.x * camera.width, right.y * camera.height])
    left_px = np.array([left.x * camera.width, left.y * camera.height])
    midpoint = (right_px + left_px) / 2

    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    eye_cm = rotation @ EYE_IN_MODEL_CM + translation

    return EyeSample(
        idx=idx,
        msec=msec,
        u_px=float(midpoint[0]),
        v_px=float(midpoint[1]),
        ipd_px=float(np.linalg.norm(left_px - right_px)),
        # MediaPipe looks down -Z, so the eye's distance is the negated Z.
        depth_m=float(-eye_cm[2] * CM_TO_M),
        yaw_deg=float(math.degrees(math.atan2(-rotation[2, 0], rotation[0, 0]))),
    )


def estimate_head_scale(samples: list[EyeSample], camera: CameraAssumptions) -> float:
    """Estimate the factor correcting MediaPipe's metric scale for this viewer.

    MediaPipe fits an identity-dependent mesh, so the head size implicit in its
    output varies per person: measured at 70.4 mm of interpupillary distance for
    one subject and 62.2 mm for another, 13% apart. Comparing that implied size
    against the viewer's real interpupillary distance gives one constant that
    puts the whole track on the right scale.

    Only near-front-facing frames are used, because apparent interpupillary
    distance shrinks under yaw. The result is a constant, so it does not carry
    that yaw sensitivity into the per-frame position.

    Args:
        samples: Samples from the clip, in any order.
        camera: Camera assumptions, supplying the focal length and the viewer's
            interpupillary distance.

    Returns:
        Multiplicative scale factor, or ``1.0`` when no frame is front-facing
        enough to measure with.
    """
    usable = [s for s in samples if abs(s.yaw_deg) <= FRONT_FACING_YAW_DEG]
    if not usable:
        lg.warning("No front-facing frames, leaving MediaPipe's scale untouched")
        return 1.0

    implied = np.array([s.ipd_px * s.depth_m / camera.focal for s in usable])
    implied_ipd_m = float(np.median(implied))
    scale = camera.ipd_m / implied_ipd_m
    lg.info(
        f"Head scale from {len(usable)} frames: "
        f"implied IPD {implied_ipd_m * 1000:.1f} mm, "
        f"viewer IPD {camera.ipd_m * 1000:.1f} mm, scale {scale:.3f}"
    )
    return scale


def eye_position_m(
    sample: EyeSample,
    camera: CameraAssumptions,
    head_scale: float = 1.0,
) -> np.ndarray:
    """Convert a sample into a 3D eye position in metres.

    Args:
        sample: Raw measurements for one frame.
        camera: Camera assumptions supplying focal length, principal point and
            whether the capture is mirrored.
        head_scale: Factor from :func:`estimate_head_scale`.

    Returns:
        Array ``[x, y, z]`` in metres, in the frame documented at module level.
    """
    cx, cy = camera.principal_point
    depth = sample.depth_m * head_scale
    x = (sample.u_px - cx) * depth / camera.focal
    y = (sample.v_px - cy) * depth / camera.focal
    if camera.mirrored:
        x = -x
    return np.array([x, y, depth])
