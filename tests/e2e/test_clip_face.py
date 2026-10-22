"""The synthetic clip, checked by the Python face landmarker it is meant for.

`tests/ui/test_clip.py` checks the geometry with a drawn stand-in. This checks
the real thing: in the frames where the face is smallest and furthest from
rest, MediaPipe finds it, and the iris midpoint is where the path says.
"""

from pathlib import Path

import cv2 as cv
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode as VisionRunningMode,
)
import numpy as np
from pose_tools.landmark.face import FaceLandmarkerFrame
from pose_tools.landmark.model_manager import ModelManager
from pose_tools.utils.mediapipe import FACE_LEFT_IRIS_CENTER
from pose_tools.utils.mediapipe import FACE_RIGHT_IRIS_CENTER
from pose_tools.video.frame import Frame

from abyss.ui.clip import FRAME_HEIGHT
from abyss.ui.clip import FRAME_WIDTH
from abyss.ui.clip import SEGMENT_FRAMES
from abyss.ui.clip import face_path
from abyss.ui.clip import split_mjpeg

IRIS_TOLERANCE_PX = 2.0
"""How far the tracked iris midpoint may sit from the path.

Measured on 2026-10-22: the worst residual over every fifth frame was 0.81 px.
The path point was itself measured with this landmarker on the still portrait,
so the residual is the landmarker's own jitter after resampling.
"""


def test_landmarker_finds_the_face_where_the_path_puts_it(clip: Path) -> None:
    """At rest, at both sweep extremes and closer, the iris midpoint is on the path."""
    frames = split_mjpeg(clip.read_bytes())
    quarter, three_quarters = SEGMENT_FRAMES // 4, 3 * SEGMENT_FRAMES // 4
    picks = [
        0,
        quarter,
        three_quarters,
        SEGMENT_FRAMES + quarter,
        2 * SEGMENT_FRAMES + quarter,
    ]
    model = ModelManager().ensure_model("face_landmarker")
    with FaceLandmarkerFrame(
        model, {"running_mode": VisionRunningMode.IMAGE}
    ) as landmarker:
        for idx in picks:
            bgr = cv.imdecode(np.frombuffer(frames[idx], np.uint8), cv.IMREAD_COLOR)
            assert bgr is not None, f"frame {idx} does not decode"
            result = landmarker.detect(Frame.from_opencv(bgr, idx=idx))
            assert result.face_landmarks, f"no face in frame {idx}"
            face = result.face_landmarks[0]
            mid_x = (face[FACE_LEFT_IRIS_CENTER].x + face[FACE_RIGHT_IRIS_CENTER].x) / 2
            mid_y = (face[FACE_LEFT_IRIS_CENTER].y + face[FACE_RIGHT_IRIS_CENTER].y) / 2
            point = face_path(idx)
            assert abs(mid_x * FRAME_WIDTH - point.x_px) < IRIS_TOLERANCE_PX, idx
            assert abs(mid_y * FRAME_HEIGHT - point.y_px) < IRIS_TOLERANCE_PX, idx
