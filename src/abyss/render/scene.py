"""The scene: line segments in the screen frame, behind the window.

One primitive, a 3D line segment, so a box, a grid and a floating cube are all
the same thing to project and the same thing to draw. This is an implementation
detail of :class:`~abyss.render.renderer.WireframeRenderer` rather than a shared
contract: an OpenGL renderer would not produce segments at all, which is why the
seam sits at the renderer and not here (Q22).

Everything lives at ``z <= 0``, behind the panel. That is load-bearing rather
than tidy. With the eye in front and the scene behind, no point can ever be at
or behind the eye, so :func:`~abyss.render.frustum.project_points` cannot raise
and this phase needs no clipper at all. :class:`Scene` enforces it, so the
assumption is checked rather than assumed. The named upgrade, wanted the first
time something is meant to poke out through the window, is homogeneous-space
segment clipping against the near plane.

Depth fading is baked in at build time rather than applied per frame: it depends
only on where a segment sits in the scene, which does not change when the eye
moves.
"""

from dataclasses import dataclass

import numpy as np

from abyss.config.screen import ScreenConfig

DEFAULT_DEPTH_M = 0.6
"""How far the box extends behind the panel."""

DEFAULT_CUBE_SIZE_M = 0.06
"""Edge length of the floating cube."""

DEFAULT_CUBE_CENTRE_M = (0.08, -0.03, -0.25)
"""Cube centre in the screen frame: off to one side, so it has room to slide."""

FRAME_MARKER_SCALE = 0.98
"""How far inside the panel rectangle the frame marker sits.

The mouth of the box is the panel itself, and its corners project exactly onto
the image corners: 1280 and 720, not 1279 and 719, since the viewport transform
maps NDC ``+1`` to the outer edge of the last pixel. Its right and bottom edges
therefore fall outside the image and are clipped away by any drawing call, so
drawn naively the mouth appears as two edges out of four.

The marker is the mouth pulled slightly in, so it draws as four edges with a
uniform gap to the border. A gap that stays uniform as the eye moves is the
visible form of phase 3's corner invariant; a gap that opens on one side says
the projection is wrong.
"""

GRID_COLUMNS = 6
"""Cells across the back wall. Parallax needs something to slide against."""

GRID_ROWS = 4
"""Cells down the back wall."""

NEAR_GAIN = 1.0
"""Colour multiplier at the panel plane."""

FAR_GAIN = 0.35
"""Colour multiplier at the back wall, the far end of the depth fade."""

BOX_BGR = (185, 185, 185)
"""The room itself, neutral."""

GRID_BGR = (120, 120, 120)
"""The back wall grid, dimmer than the room so it reads as a backdrop."""

CUBE_BGR = (0, 165, 255)
"""The floating cube, orange, so it is unmistakably a separate object."""

MARKER_BGR = (255, 200, 0)
"""The frame marker, cyan, since it is instrumentation rather than scenery."""

COLOUR_CHANNELS = 3
SEGMENT_ENDS = 2
SPATIAL_DIMS = 3


class MalformedSceneError(ValueError):
    """Raised when segments and colours do not describe a drawable scene."""

    def __init__(self, detail: str) -> None:
        """Initialise with what was wrong.

        Args:
            detail: What did not line up, in words.
        """
        super().__init__(f"Scene is not drawable: {detail}")


class SceneInFrontOfWindowError(ValueError):
    """Raised when a scene reaches out through the window.

    The whole phase relies on the scene sitting behind the panel, since that is
    what makes clipping unnecessary. A scene that breaks it would project fine
    until the eye moved and then fail inside the projection, so it is refused
    at construction instead.
    """

    def __init__(self, max_z_m: float) -> None:
        """Initialise with how far out the scene reaches.

        Args:
            max_z_m: The largest Z found, in metres.
        """
        super().__init__(
            f"Scene must sit behind the panel at z <= 0, found z={max_z_m:.4f} m. "
            f"Drawing in front of the window needs near-plane clipping, which "
            f"this phase deliberately does not have"
        )


@dataclass(frozen=True)
class Scene:
    """A set of coloured line segments in the screen frame.

    Args:
        segments: Array ``(N, 2, 3)``, two endpoints per segment, metres.
        colours: Array ``(N, 3)``, one BGR colour per segment, 0 to 255.
    """

    segments: np.ndarray
    colours: np.ndarray

    def __post_init__(self) -> None:
        """Validate the arrays line up and the scene stays behind the panel.

        Raises:
            MalformedSceneError: If the shapes are wrong or disagree.
            SceneInFrontOfWindowError: If any point sits in front of the panel.
        """
        if self.segments.ndim != SPATIAL_DIMS or self.segments.shape[1:] != (
            SEGMENT_ENDS,
            SPATIAL_DIMS,
        ):
            detail = f"segments must have shape (N, 2, 3), got {self.segments.shape}"
            raise MalformedSceneError(detail)
        if self.colours.shape != (len(self.segments), COLOUR_CHANNELS):
            detail = (
                f"expected {len(self.segments)} colours of 3 channels, got "
                f"{self.colours.shape}"
            )
            raise MalformedSceneError(detail)
        if len(self.segments):
            max_z = float(self.segments[..., 2].max())
            if max_z > 0:
                raise SceneInFrontOfWindowError(max_z)

    def __len__(self) -> int:
        """Count the segments."""
        return len(self.segments)


def _fade(colour: tuple[int, int, int], z_m: float, depth_m: float) -> np.ndarray:
    """Dim a colour according to how far back it sits.

    Args:
        colour: Base BGR colour.
        z_m: Depth in the screen frame, zero at the panel and negative behind.
        depth_m: Depth of the whole scene, the far end of the fade.

    Returns:
        The faded BGR colour, as three values 0 to 255.
    """
    far_fraction = min(abs(z_m) / depth_m, 1.0)
    gain = NEAR_GAIN + (FAR_GAIN - NEAR_GAIN) * far_fraction
    return np.round(np.asarray(colour, dtype=float) * gain)


def _rectangle(half_w: float, half_h: float, z_m: float) -> np.ndarray:
    """Build the four corners of an axis-aligned rectangle.

    Args:
        half_w: Half width in metres.
        half_h: Half height in metres.
        z_m: Depth to place it at.

    Returns:
        Corners ``(4, 3)``, anticlockwise from the top left.
    """
    return np.array(
        [
            [-half_w, half_h, z_m],
            [half_w, half_h, z_m],
            [half_w, -half_h, z_m],
            [-half_w, -half_h, z_m],
        ]
    )


def _loop(corners: np.ndarray) -> list[np.ndarray]:
    """Join a ring of corners into closed segments.

    Args:
        corners: Points ``(N, 3)`` in order.

    Returns:
        One segment per edge, each ``(2, 3)``.
    """
    return [
        np.array([corners[i], corners[(i + 1) % len(corners)]])
        for i in range(len(corners))
    ]


def _cube_segments(centre_m: tuple[float, float, float], size_m: float) -> list:
    """Build the twelve edges of an axis-aligned cube.

    Args:
        centre_m: Cube centre in the screen frame.
        size_m: Edge length in metres.

    Returns:
        Twelve segments, each ``(2, 3)``.
    """
    half = size_m / 2
    cx, cy, cz = centre_m
    front = _rectangle(half, half, cz + half) + np.array([cx, cy, 0.0])
    back = _rectangle(half, half, cz - half) + np.array([cx, cy, 0.0])
    edges = _loop(front) + _loop(back)
    edges += [np.array([front[i], back[i]]) for i in range(len(front))]
    return edges


def window_box(
    screen: ScreenConfig,
    depth_m: float = DEFAULT_DEPTH_M,
    cube_centre_m: tuple[float, float, float] = DEFAULT_CUBE_CENTRE_M,
    cube_size_m: float = DEFAULT_CUBE_SIZE_M,
) -> Scene:
    """Build the minimal scene: a room behind the panel with a cube in it.

    The mouth of the room is the panel rectangle itself, so it maps to the
    image border whatever the eye does. Its edges are not drawn, since they
    clip away exactly on the boundary - see :data:`FRAME_MARKER_SCALE`. The four
    corner connectors are drawn and still converge on the image corners, so the
    room reads as a room.

    Args:
        screen: The display, which sets the size of the mouth.
        depth_m: How far back the room extends.
        cube_centre_m: Where the floating cube sits, in the screen frame.
        cube_size_m: Cube edge length in metres.

    Returns:
        The scene, ready to project.
    """
    half_w, half_h = screen.width_m / 2, screen.height_m / 2
    mouth = _rectangle(half_w, half_h, 0.0)
    back = _rectangle(half_w, half_h, -depth_m)

    segments: list[np.ndarray] = []
    colours: list[tuple[int, int, int]] = []

    def add(new: list[np.ndarray], colour: tuple[int, int, int]) -> None:
        segments.extend(new)
        colours.extend([colour] * len(new))

    # Instrumentation first: the mouth pulled in far enough to be drawable.
    marker = _rectangle(half_w * FRAME_MARKER_SCALE, half_h * FRAME_MARKER_SCALE, 0.0)
    add(_loop(marker), MARKER_BGR)

    # The room: the back rectangle, and the corners running out to the mouth.
    add(_loop(back), BOX_BGR)
    add([np.array([mouth[i], back[i]]) for i in range(len(mouth))], BOX_BGR)

    # One line along the middle of each wall, so they read as receding.
    mids = (mouth + np.roll(mouth, -1, axis=0)) / 2
    back_mids = (back + np.roll(back, -1, axis=0)) / 2
    add([np.array([mids[i], back_mids[i]]) for i in range(len(mids))], BOX_BGR)

    # The back wall grid, which is what the cube slides against.
    grid: list[np.ndarray] = []
    for i in range(1, GRID_COLUMNS):
        x = -half_w + 2 * half_w * i / GRID_COLUMNS
        grid.append(np.array([[x, half_h, -depth_m], [x, -half_h, -depth_m]]))
    for i in range(1, GRID_ROWS):
        y = -half_h + 2 * half_h * i / GRID_ROWS
        grid.append(np.array([[-half_w, y, -depth_m], [half_w, y, -depth_m]]))
    add(grid, GRID_BGR)

    add(_cube_segments(cube_centre_m, cube_size_m), CUBE_BGR)

    stacked = np.array(segments)
    midpoint_z = stacked[..., 2].mean(axis=1)
    faded = np.array(
        [
            _fade(colour, z, depth_m)
            for colour, z in zip(colours, midpoint_z, strict=True)
        ]
    )
    return Scene(segments=stacked, colours=faded)
