"""What the page is told: device configs and the constants of the maths.

The page ports the eye position maths, but not its numbers. Every device
number comes from ``abyss_devices.py`` and every constant from the module that
defines it, served as one JSON document at ``/config.json``. A constant changed
in Python reaches the page on its next load, and nothing in JavaScript can
quietly disagree with it.

The query string picks the devices by registry key:
``/config.json?camera=pixel7pro_front&viewer=default&screen=g7_internal``.
``screen`` is optional: without it the page tracks and draws no scene.

The scene is served the same way, at ``/scene.json?screen=<key>``: the segments
and colours :func:`abyss.render.scene.window_box` builds for that panel. The
page draws what it is given, so its geometry cannot drift from Python's, and a
different scene is a different document rather than a change to the page.
"""

from typing import Any

from pose_tools.utils.mediapipe import FACE_LEFT_IRIS_CENTER
from pose_tools.utils.mediapipe import FACE_RIGHT_IRIS_CENTER

from abyss.config.camera import MEDIAPIPE_VERTICAL_FOV_DEG
from abyss.params.abyss_devices import get_camera
from abyss.params.abyss_devices import get_screen
from abyss.params.abyss_devices import get_viewer
from abyss.render.frustum import CAMERA_TO_SCREEN
from abyss.render.frustum import DEFAULT_FAR_M
from abyss.render.frustum import DEFAULT_NEAR_M
from abyss.render.renderer import ASPECT_TOLERANCE
from abyss.render.renderer import BACKGROUND_BGR
from abyss.render.scene import window_box
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
        "near_m": DEFAULT_NEAR_M,
        "far_m": DEFAULT_FAR_M,
        "camera_to_screen": CAMERA_TO_SCREEN.tolist(),
        "aspect_tolerance": ASPECT_TOLERANCE,
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


def scene_document(screen: str) -> dict[str, Any]:
    """Build the document served at ``/scene.json``.

    Args:
        screen: Screen registry key; the scene's mouth is that panel.

    Returns:
        ``segments`` as ``(N, 2, 3)`` nested lists in the screen frame, metres;
        ``colours`` as ``(N, 3)`` BGR, 0 to 255; ``background`` BGR.

    Raises:
        UnknownDeviceError: If the key is not registered.
    """
    scene = window_box(get_screen(screen))
    return {
        "segments": scene.segments.tolist(),
        "colours": scene.colours.tolist(),
        "background": list(BACKGROUND_BGR),
    }
