"""The device registry: actual values for the config models.

Entries are keyed by **device**, not by machine. The Pixel will record frames
that another machine processes, so the host running the code says nothing about
the camera the frames came from (Q7).

Values are plain Python literals (Q10). A loader arrives only when something
outside the repo needs to write config, and nothing does.

Camera intrinsics are all unmeasured today, which is deliberate rather than
unfinished: measuring a focal length needs one object of known size at a known
distance in front of the camera, and that is a manual step on the machine
holding it. Until then every camera falls back to MediaPipe's own assumption,
which is exactly what the pipeline did before this registry existed.
"""

from abyss.config.camera import CameraConfig
from abyss.config.screen import ScreenConfig
from abyss.config.stream import StreamConfig
from abyss.config.viewer import ViewerConfig
from abyss.params.abyss_params import get_abyss_paths

KIND_CAMERA = "camera"
KIND_SCREEN = "screen"
KIND_VIEWER = "viewer"


class UnknownDeviceError(KeyError):
    """Raised when a registry lookup names something that is not registered."""

    def __init__(self, kind: str, name: str, known: list[str]) -> None:
        """Initialise with the offending name.

        Args:
            kind: What was being looked up, for the message.
            name: The name that was not found.
            known: The names that are registered.
        """
        super().__init__(f"Unknown {kind} {name!r}, known: {', '.join(sorted(known))}")


CAMERAS: dict[str, CameraConfig] = {
    "unknown_clip": CameraConfig(
        name="unknown_clip",
        provenance="sample clips with no known camera; MediaPipe's assumption applies",
    ),
    "g4_internal": CameraConfig(
        name="g4_internal",
        provenance=(
            "HP HD Camera, USB 04ca:7063. Present but unusable in practice: g4 is "
            "reached over ssh, so nobody is sitting in front of it"
        ),
    ),
    "g7_webcam": CameraConfig(
        name="g7_webcam",
        focal_px=945.0,
        focal_measured_at_height=720,
        provenance=(
            "Chicony 04f2:b6c8, measured 2026-08-16 by ChArUco calibration off a "
            "Kindle Paperwhite 11 at 300 ppi, 15 views at 1280x720, rms 0.263 px. "
            "A second run of 8 views at a different distance gave 940.4, agreeing "
            "to 0.5%. Vertical field of view 41.7 deg, not the 63 deg MediaPipe "
            "assumes, so the fallback underestimates depth here by 38%. "
            "Principal point 646x374 and k1 near zero were also recovered but are "
            "loosely constrained, the board having stayed near the frame centre. "
            "Read from the device with v4l2-ctl on 2026-08-17, correcting an "
            "earlier guess: MJPG offers eight sizes, of which 1280x720, 960x540, "
            "640x360 and 320x180 are exactly 16:9 and share this focal length's "
            "aspect ratio, so rescaling within that family is meaningful. The 4:3 "
            "modes 640x480 and 320x240 are not, and 848x480 is 1.767 rather than "
            "1.778. Every mode caps at 30 fps. Pin capture to MJPG 1280x720, "
            "since the YUYV default silently clamps. Reported by the kernel as "
            "'HP HD Camera', which is the product string on a Chicony module: "
            "g4_internal carries that same product string on different silicon"
        ),
    ),
    "pixel7pro_front": CameraConfig(
        name="pixel7pro_front",
        mirrored=True,
        provenance="unmeasured; front cameras mirror their preview",
    ),
}
"""Capture devices."""

SCREENS: dict[str, ScreenConfig] = {
    "g4_internal": ScreenConfig(
        name="g4_internal",
        width_m=0.309,
        height_m=0.173,
        # Camera above the panel, so the panel centre is below it: +Y is down.
        camera_to_centre_m=(0.0, 0.0965, 0.0),
        provenance=(
            "size from the panel's own EDID, exact. Offset is PROVISIONAL: half the "
            "panel height plus a 10 mm bezel guess, since measuring it needs a ruler "
            "held against a machine reached over ssh"
        ),
    ),
    "g7_internal": ScreenConfig(
        name="g7_internal",
        width_m=0.344,
        height_m=0.193,
        # Camera above the panel, so the panel centre is below it: +Y is down.
        # Half the panel height, 96.5 mm, plus the 4 mm measured bezel gap.
        camera_to_centre_m=(0.0, 0.1005, 0.0),
        provenance=(
            "size from the panel's own EDID on card1-eDP-1, exact. Offset "
            "measured 2026-08-16 with a ruler: 4 mm from the top edge of the "
            "active area to the centre of the lens, plus half the panel height"
        ),
    ),
}
"""Display devices."""

VIEWERS: dict[str, ViewerConfig] = {
    "default": ViewerConfig(name="default"),
}
"""People. One today, which is why nothing selects between them."""

SAMPLE_CLIPS = ("face01.mp4", "face02_portrait.mp4", "face03_zoom.mp4")
"""The clips phase 1 was validated against, in ``AbyssPaths.pose_fol``."""


def get_camera(name: str) -> CameraConfig:
    """Look up a capture device.

    Args:
        name: Registry key.

    Returns:
        The camera config.

    Raises:
        UnknownDeviceError: If the name is not registered.
    """
    if name not in CAMERAS:
        raise UnknownDeviceError(KIND_CAMERA, name, list(CAMERAS))
    return CAMERAS[name]


def get_screen(name: str) -> ScreenConfig:
    """Look up a display device.

    Args:
        name: Registry key.

    Returns:
        The screen config.

    Raises:
        UnknownDeviceError: If the name is not registered.
    """
    if name not in SCREENS:
        raise UnknownDeviceError(KIND_SCREEN, name, list(SCREENS))
    return SCREENS[name]


def get_viewer(name: str = "default") -> ViewerConfig:
    """Look up a viewer.

    Args:
        name: Registry key.

    Returns:
        The viewer config.

    Raises:
        UnknownDeviceError: If the name is not registered.
    """
    if name not in VIEWERS:
        raise UnknownDeviceError(KIND_VIEWER, name, list(VIEWERS))
    return VIEWERS[name]


def sample_stream(clip: str) -> StreamConfig:
    """Build a stream over one of the sample clips.

    The clips live outside the repo and are not on every machine, so this
    builds the config without checking that the file is there.

    Args:
        clip: File name inside ``AbyssPaths.pose_fol``.

    Returns:
        A stream config reading that clip through the ``unknown_clip`` camera.
    """
    return StreamConfig(
        name=clip,
        camera="unknown_clip",
        source=get_abyss_paths().pose_fol / clip,
    )
