"""Test the document the page is configured from."""

import json

from abyss.config.camera import MEDIAPIPE_VERTICAL_FOV_DEG
from abyss.params.abyss_devices import get_camera
from abyss.ui.page_config import DEFAULT_CAMERA
from abyss.ui.page_config import page_config
from abyss.viewer.eye_position import EYE_IN_MODEL_CM


def test_default_is_the_phone_camera() -> None:
    """With no query, the page is configured for the Pixel's front camera."""
    config = page_config()
    assert config["camera"]["name"] == DEFAULT_CAMERA
    assert config["camera"] == get_camera(DEFAULT_CAMERA).model_dump(mode="json")
    assert config["screen"] is None


def test_constants_are_the_python_ones() -> None:
    """The page gets the numbers from where they are defined, not a copy."""
    constants = page_config()["constants"]
    assert constants["mediapipe_vertical_fov_deg"] == MEDIAPIPE_VERTICAL_FOV_DEG
    assert constants["eye_in_model_cm"] == EYE_IN_MODEL_CM.tolist()


def test_document_is_json() -> None:
    """Everything in it survives a JSON round trip unchanged."""
    config = page_config("g7_webcam", screen="g7_internal")
    assert json.loads(json.dumps(config)) == config
