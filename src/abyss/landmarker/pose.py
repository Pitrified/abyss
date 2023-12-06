"""Detect and extract pose landmarks from images."""

from pathlib import Path
from typing import Generator, Literal

import cv2 as cv

from abyss.video.frame import Frame

from loguru import logger as lg

from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarkerResult,
)
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode as VisionRunningMode,
)


def create_pose_landmarker(
    pose_landmark_model_path: Path,
    **kwargs,
) -> PoseLandmarker:
    """Create the landmarker object.

    Default PoseLandmarkerOptions kwargs are:
        base_options: _BaseOptions,
        running_mode: _RunningMode = _RunningMode.IMAGE,
        num_poses: int = 1,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        output_segmentation_masks: bool = False,
        result_callback: ((PoseLandmarkerResult, Unknown, int) -> None) | None = None
    """
    base_options = BaseOptions(model_asset_path=str(pose_landmark_model_path))
    options = PoseLandmarkerOptions(base_options=base_options, **kwargs)
    detector = PoseLandmarker.create_from_options(options)
    return detector


class PoseLandmarkerFrame:
    """PoseLandmarker that can accept Frame objects as input."""

    def __init__(
        self,
        pose_landmark_model_path: Path,
        pose_landmarker_kwargs: dict = {},
    ) -> None:
        """Initialize the PoseLandmarkerFrame."""
        self.pose_landmarker = create_pose_landmarker(
            pose_landmark_model_path,
            **pose_landmarker_kwargs,
        )
        self.pose_landmarker_kwargs = pose_landmarker_kwargs

    def detect(self, frame: Frame) -> PoseLandmarkerResult:
        """Process a frame, according to the running mode."""
        running_mode = self.pose_landmarker_kwargs.get(
            "running_mode", VisionRunningMode.IMAGE
        )
        if running_mode == VisionRunningMode.IMAGE:
            return self.pose_landmarker.detect(frame.image)
        return self.pose_landmarker.detect_for_video(frame.image, int(frame.msec))
