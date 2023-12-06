"""Drawing module for landmark detection."""

import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
import mediapipe.python.solutions.drawing_utils as mp_drawing_utils
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult
from loguru import logger as lg
import numpy as np
from abyss.utils.mediapipe import (
    get_default_pose_connections,
    get_landmarks_from_result,
    list_land_to_landlist,
)

from abyss.video.frame import Frame


def draw_landmarks(
    frame: Frame,
    detection_result: PoseLandmarkerResult,
) -> np.ndarray:
    """Draw the landmarks on the image."""
    rgb_image = frame.image.numpy_view()

    # Get the pose landmarks (automatically get only the first).
    pose_landmarks = get_landmarks_from_result(detection_result, "normalized")
    if pose_landmarks is None:
        lg.warning("No pose landmarks detected.")
        return rgb_image

    # Draw the pose landmarks.
    landmark_list = list_land_to_landlist(pose_landmarks)
    mp_drawing_utils.draw_landmarks(
        rgb_image,
        landmark_list,
        get_default_pose_connections(),
        mp_drawing_styles.get_default_pose_landmarks_style(),
        # mp_drawing_styles.get_default_pose_connections_style(),
    )

    return rgb_image
