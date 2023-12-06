"""Utility functions dealing with IO."""

from pathlib import Path
from typing import Literal

from loguru import logger as lg


def get_resource(
    which_res: Literal[
        "pose_landmarker.task",
        "root_fol",
        "sample_fol",
        "3D_model_fol",
        "pose_fol",
    ]
) -> Path:
    """Get the path of the requested resource."""
    # resources
    if which_res == "pose_landmarker.task":
        mp_model_fol = Path.home() / ".mediapipe" / "models"
        pose_landmark_model_path = mp_model_fol / "pose_landmarker.task"
        return pose_landmark_model_path

    # folders that are not in the package
    if which_res == "3D_model_fol":
        return Path.home() / "data" / "3d_models"
    elif which_res == "pose_fol":
        return Path.home() / "data" / "pose"

    # folders that are in the package
    if which_res == "root_fol":
        return Path(__file__).absolute().parents[3]
    elif which_res == "sample_fol":
        return get_resource("root_fol") / "data" / "sample"


def check_create_fol(
    fol: Path,
) -> None:
    """Check if folder exists, if not create it."""
    if not fol.exists():
        lg.debug(f"Creating folder {fol}")
        fol.mkdir(parents=True)
