"""Misc utils for landmark detection."""


from enum import IntEnum
from typing import Literal, Mapping, TypeVar, cast, get_args

import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
import mediapipe.python.solutions.drawing_utils as mp_drawing

# import mediapipe.python.solutions.pose as mp_pose
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.python.solutions.pose_connections import POSE_CONNECTIONS

from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark
import numpy as np
from typing import Literal, overload
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
)
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarkerResult,
)
from mediapipe.tasks.python.components.containers.landmark import (
    Landmark,
    NormalizedLandmark,
)
from mediapipe.framework.formats.landmark_pb2 import (
    NormalizedLandmarkList,
    LandmarkList,
)
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.components.containers.category import Category

T = TypeVar("T")

POSE_LANDMARK_NAMES = PoseLandmark._member_names_
POSE_LANDMARK_MAP = cast(dict[str, IntEnum], PoseLandmark._member_map_)


def get_default_pose_connections() -> list[tuple[int, int]]:
    """Get the default pose connections.

    Cast the connections to a list of tuples for the sake of type checking.
    """
    pose_connections = cast(list[tuple[int, int]], POSE_CONNECTIONS)
    return pose_connections


def get_spec_from_map(
    drawing_spec: mp_drawing.DrawingSpec | Mapping[T, mp_drawing.DrawingSpec],
    key: T,
) -> mp_drawing.DrawingSpec:
    """Extract a DrawingSpec from a Mapping or return the DrawingSpec itself."""
    if isinstance(drawing_spec, Mapping):
        return drawing_spec[key]
    return drawing_spec


@overload
def get_landmarks_from_result(
    result: PoseLandmarkerResult,
    which_info: Literal["world"],
    pose_idx: int = 0,
) -> list[Landmark] | None:
    ...


@overload
def get_landmarks_from_result(
    result: PoseLandmarkerResult,
    which_info: Literal["normalized"],
    pose_idx: int = 0,
) -> list[NormalizedLandmark] | None:
    ...


def get_landmarks_from_result(
    result: PoseLandmarkerResult,
    which_info: Literal["world", "normalized"],
    pose_idx: int = 0,
) -> list[Landmark] | list[NormalizedLandmark] | None:
    """Get the info from the result, for a specific pose."""
    if which_info == "world":
        ll = result.pose_world_landmarks
    elif which_info == "normalized":
        ll = result.pose_landmarks
    if pose_idx >= len(ll):
        return None
    return ll[pose_idx]


@overload
def list_land_to_landlist(
    landmarks: list[NormalizedLandmark],
) -> NormalizedLandmarkList:
    ...


@overload
def list_land_to_landlist(
    landmarks: list[Landmark],
) -> LandmarkList:
    ...


def list_land_to_landlist(
    landmarks: list[NormalizedLandmark] | list[Landmark],
) -> NormalizedLandmarkList | LandmarkList:
    """Convert a list of [Normalized]Landmark to a [Normalized]LandmarkList."""
    # decide which type of landmark list to use
    if isinstance(landmarks[0], NormalizedLandmark):
        landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    elif isinstance(landmarks[0], Landmark):
        landmarks_proto = landmark_pb2.LandmarkList()
    else:
        raise TypeError("landmarks must be a list of NormalizedLandmark or Landmark")

    # add the landmarks to the list
    landmarks_proto.landmark.extend(  # type: ignore # Member "landmark" is unknown
        [landmark.to_pb2() for landmark in landmarks]
    )
    return landmarks_proto
