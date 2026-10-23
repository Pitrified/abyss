"""The Python pipeline over an MJPEG clip, as the reference the page is held to.

The phone UI carries a second implementation of the eye position maths. That
is accepted only because something compares the two on the same frames
(``plans/03_phone_webapp/tracking.md``), and this is the Python side of that
comparison: the same landmarker settings ``scripts/viewer_position.py`` uses,
the same model file the server hands the page, and the same JPEG bytes the
page decodes.
"""

from dataclasses import asdict
from pathlib import Path

import cv2 as cv
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode as VisionRunningMode,
)
import numpy as np
from pose_tools.landmark.face import FaceLandmarkerFrame
from pose_tools.landmark.model_manager import ModelManager
from pose_tools.video.frame import Frame

from abyss.config.camera import CameraConfig
from abyss.config.camera import FrameGeometry
from abyss.ui.clip import FPS
from abyss.ui.clip import split_mjpeg
from abyss.viewer.eye_position import EyeSample
from abyss.viewer.eye_position import extract_eye_sample

LANDMARKER_OPTIONS = {
    "running_mode": VisionRunningMode.VIDEO,
    "num_faces": 1,
    "output_facial_transformation_matrixes": True,
}
"""The settings the page uses too. VIDEO mode, so tracking carries across frames."""


class UndecodableFrameError(ValueError):
    """Raised when a frame of a clip is not a JPEG OpenCV can decode."""

    def __init__(self, idx: int) -> None:
        """Initialise with the frame index.

        Args:
            idx: The frame that failed.
        """
        super().__init__(f"frame {idx} does not decode")


def frame_msec(idx: int, fps: float = FPS) -> float:
    """Timestamp of a frame, the one both implementations hand MediaPipe.

    Args:
        idx: Frame index.
        fps: Frames per second.

    Returns:
        Milliseconds from the start.
    """
    return idx * 1000.0 / fps


def track_mjpeg(
    path: Path, camera: CameraConfig, fps: float = FPS
) -> list[EyeSample | None]:
    """Run the Python face landmarker over every frame of an MJPEG clip.

    Args:
        path: The clip.
        camera: The device the clip claims to come from.
        fps: Frame rate, for the timestamps VIDEO mode requires.

    Returns:
        One sample per frame, ``None`` where no face was found.

    Raises:
        UndecodableFrameError: If a frame does not decode.
    """
    frames = split_mjpeg(path.read_bytes())
    samples: list[EyeSample | None] = []
    geometry: FrameGeometry | None = None
    model = ModelManager().ensure_model("face_landmarker")
    with FaceLandmarkerFrame(model, LANDMARKER_OPTIONS) as landmarker:
        for idx, jpeg in enumerate(frames):
            bgr = cv.imdecode(np.frombuffer(jpeg, np.uint8), cv.IMREAD_COLOR)
            if bgr is None:
                raise UndecodableFrameError(idx)
            if geometry is None:
                height, width = bgr.shape[:2]
                geometry = FrameGeometry(camera=camera, width=width, height=height)
            msec = frame_msec(idx, fps)
            result = landmarker.detect(Frame.from_opencv(bgr, msec=msec, idx=idx))
            samples.append(extract_eye_sample(result, geometry, idx, msec))
    return samples


def sample_dict(sample: EyeSample | None) -> dict | None:
    """Serialise a sample with the field names the page uses.

    Args:
        sample: A sample or None.

    Returns:
        A plain dict, or None.
    """
    return None if sample is None else asdict(sample)
