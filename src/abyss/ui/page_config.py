"""What the page is told: device configs and the constants of the maths.

The page ports the eye position maths, but not its numbers. Every device
number comes from ``abyss_devices.py`` and every constant from the module that
defines it, served as one JSON document at ``/config.json``. A constant changed
in Python reaches the page on its next load, and nothing in JavaScript can
quietly disagree with it.

The query string picks the devices by registry key:
``/config.json?camera=pixel7pro_front&viewer=default``. ``screen`` is optional
until phase 3 needs it.
"""

from typing import Any

from pose_tools.utils.mediapipe import FACE_LEFT_IRIS_CENTER
from pose_tools.utils.mediapipe import FACE_RIGHT_IRIS_CENTER

from abyss.config.camera import MEDIAPIPE_VERTICAL_FOV_DEG
from abyss.params.abyss_devices import get_camera
from abyss.params.abyss_devices import get_screen
from abyss.params.abyss_devices import get_viewer
from abyss.viewer.eye_position import BOOTSTRAP_SAMPLES
from abyss.viewer.eye_position import CM_TO_M
from abyss.viewer.eye_position import EYE_IN_MODEL_CM
from abyss.viewer.eye_position import FRONT_FACING_YAW_DEG
from abyss.viewer.smoothing import DEFAULT_FILTER_SIZE

DEFAULT_CAMERA = "pixel7pro_front"
DEFAULT_VIEWER = "default"


def constants() -> dict[str, object]:
    """Collect the constants the page's port needs, from where they live.

    Returns:
        Name to value, JSON-ready.
    """
    return {
        "mediapipe_vertical_fov_deg": MEDIAPIPE_VERTICAL_FOV_DEG,
        "eye_in_model_cm": [float(v) for v in EYE_IN_MODEL_CM],
        "cm_to_m": CM_TO_M,
        "front_facing_yaw_deg": FRONT_FACING_YAW_DEG,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "filter_size": DEFAULT_FILTER_SIZE,
        "right_iris_center": FACE_RIGHT_IRIS_CENTER,
        "left_iris_center": FACE_LEFT_IRIS_CENTER,
    }


def page_config(
    camera: str = DEFAULT_CAMERA,
    viewer: str = DEFAULT_VIEWER,
    screen: str | None = None,
) -> dict[str, Any]:
    """Build the document served at ``/config.json``.

    Args:
        camera: Camera registry key.
        viewer: Viewer registry key.
        screen: Screen registry key, or None.

    Returns:
        ``camera``, ``viewer``, ``screen`` (or None) and ``constants``.

    Raises:
        UnknownDeviceError: If a key is not registered.
    """
    return {
        "camera": get_camera(camera).model_dump(mode="json"),
        "viewer": get_viewer(viewer).model_dump(mode="json"),
        "screen": None
        if screen is None
        else get_screen(screen).model_dump(mode="json"),
        "constants": constants(),
    }
