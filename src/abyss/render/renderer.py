"""Turning a projection into pixels.

This is the seam of phase 4, and where it sits was the one thing worth getting
right before writing any of it. The obvious place is the scene: an interface
that yields geometry, wireframe today and a loaded model later. That is wrong,
because the named second implementation is an OpenGL renderer
(``plans/02_scene_rendering/``) and it does not produce geometry, it produces
pixels. A geometry-shaped interface would exclude the very implementation it
exists for.

So the seam is one step up, at :class:`Renderer`: a view projection matrix and
a size in, an image out. A numpy wireframe satisfies it today; a GL context that
binds the same 4x4 and reads the framebuffer back satisfies it later, and so
does a splat renderer or a reprojection of captured content.
"""

from typing import Protocol
from typing import runtime_checkable

import cv2 as cv
import numpy as np

from abyss.config.screen import ScreenConfig
from abyss.render.frustum import project_points
from abyss.render.frustum import view_projection_matrix
from abyss.render.scene import Scene

BACKGROUND_BGR = (16, 16, 16)
"""Near black, so a faded far edge still has somewhere to fade to."""

LINE_THICKNESS_PX = 1

ASPECT_TOLERANCE = 0.02
"""How far the output aspect may differ from the panel's, as a fraction.

It cannot be an equality check. ``g7_internal`` is 344 by 193 mm, an aspect of
1.7824, while 1280x720 is 1.7778: a real panel is not exactly 16:9, so an exact
test would reject the actual device. 2% passes g7's 0.26% and still catches a
genuine mix-up such as rendering a 16:9 panel into a 4:3 image.
"""


class AspectMismatchError(ValueError):
    """Raised when the output shape does not match the panel's shape.

    Worth its own error because the failure is otherwise invisible. The
    projection maps the panel corners onto the viewport corners whatever the
    viewport is, so a mismatched output does not crop or letterbox: it fills
    perfectly and silently stretches everything in it.
    """

    def __init__(self, screen_aspect: float, output_aspect: float) -> None:
        """Initialise with both aspects.

        Args:
            screen_aspect: Width over height of the panel.
            output_aspect: Width over height of the output image.
        """
        super().__init__(
            f"Panel aspect {screen_aspect:.4f} does not match output aspect "
            f"{output_aspect:.4f}. The render would fill the frame and stretch "
            f"rather than letterbox, so this is refused"
        )


@runtime_checkable
class Renderer(Protocol):
    """Something that can draw a scene through a projection."""

    def render(
        self,
        view_projection: np.ndarray,
        width_px: int,
        height_px: int,
    ) -> np.ndarray:
        """Draw one frame.

        Args:
            view_projection: 4x4 matrix taking screen frame points to clip
                space, from :func:`~abyss.render.frustum.view_projection_matrix`.
            width_px: Output width in pixels.
            height_px: Output height in pixels.

        Returns:
            A BGR image of that size.
        """
        ...


class WireframeRenderer:
    """Draw a scene as anti-aliased lines with numpy and OpenCV.

    The cheapest thing that makes head-coupled perspective visible, and
    deliberately no more. There is no occlusion and no depth sorting: with
    wireframe the sort would only decide which of two crossing lines is drawn
    on top, which nothing can assert and nobody notices. Depth is carried by
    the colour fade baked into the scene instead.

    Args:
        scene: The segments to draw.
        background: Fill colour for the frame, BGR.
    """

    def __init__(
        self,
        scene: Scene,
        background: tuple[int, int, int] = BACKGROUND_BGR,
    ) -> None:
        """Initialise with the scene to draw.

        Args:
            scene: The segments to draw.
            background: Fill colour for the frame, BGR.
        """
        self.scene = scene
        self.background = background

    def _blank(self, width_px: int, height_px: int) -> np.ndarray:
        """Build a fresh frame filled with the background colour.

        Its own method because the obvious spelling is a performance trap that
        phase 5's benchmark caught. ``np.full(shape, (16, 16, 16), np.uint8)``
        takes a **3-tuple**, which puts numpy on a broadcast assignment rather
        than a memset: 8.24 ms at 1920x1080, against 1.79 ms for the per-channel
        fill below and 0.20 ms for the scalar `np.full(shape, 16)`. A quarter of
        a live frame budget, spent producing an identical array.

        The scalar is not used even though it is cheaper again. It is equivalent
        only while the background is grey, which would make correctness a
        precondition on a constructor argument callers may change, and a
        non-grey background would then render silently in the wrong colour.
        Picking the scalar when all three channels agree is the named upgrade if
        something ever needs the other 1.6 ms.

        The frame is freshly allocated every call on purpose. Reusing one buffer
        would take this to zero and would be wrong at this seam: `render`
        returns the frame and callers keep it, so a reused buffer would rewrite
        frames that had already been handed to a sink.

        Args:
            width_px: Frame width in pixels.
            height_px: Frame height in pixels.

        Returns:
            A new BGR image of that size, filled with :attr:`background`.
        """
        frame = np.empty((height_px, width_px, 3), dtype=np.uint8)
        for channel, value in enumerate(self.background):
            frame[:, :, channel] = value
        return frame

    def render(
        self,
        view_projection: np.ndarray,
        width_px: int,
        height_px: int,
    ) -> np.ndarray:
        """Draw the scene through this projection.

        Args:
            view_projection: 4x4 matrix taking screen frame points to clip
                space.
            width_px: Output width in pixels.
            height_px: Output height in pixels.

        Returns:
            A BGR image of that size.
        """
        frame = self._blank(width_px, height_px)
        if not len(self.scene):
            return frame

        ends = self.scene.segments.reshape(-1, 3)
        pixels = project_points(view_projection, ends, width_px, height_px)
        # The scene is bounded and sits inside the mouth's extrusion, so
        # projections stay within a few frame widths and need no guard before
        # the integer conversion. cv.line clips the rest.
        drawn = np.rint(pixels).astype(int).reshape(-1, 2, 2)

        for (start, end), colour in zip(drawn, self.scene.colours, strict=True):
            cv.line(
                frame,
                (int(start[0]), int(start[1])),
                (int(end[0]), int(end[1])),
                tuple(float(c) for c in colour),
                LINE_THICKNESS_PX,
                cv.LINE_AA,
            )
        return frame


def check_aspect(screen: ScreenConfig, width_px: int, height_px: int) -> None:
    """Refuse an output shape that would silently stretch the scene.

    Args:
        screen: The panel being rendered for.
        width_px: Output width in pixels.
        height_px: Output height in pixels.

    Raises:
        AspectMismatchError: If the two aspects differ by more than
            :data:`ASPECT_TOLERANCE`.
    """
    screen_aspect = screen.width_m / screen.height_m
    output_aspect = width_px / height_px
    if abs(output_aspect - screen_aspect) / screen_aspect > ASPECT_TOLERANCE:
        raise AspectMismatchError(screen_aspect, output_aspect)


def render_frame(
    screen: ScreenConfig,
    eye_camera_m: np.ndarray,
    renderer: Renderer,
    size: tuple[int, int],
) -> np.ndarray:
    """Render one frame for one eye position.

    The entry point both the offline loop and, later, the live one call. It is
    where the screen and the output size meet, which is why the aspect check
    lives here rather than in either config.

    Args:
        screen: The display being looked through.
        eye_camera_m: Eye position in the **camera** frame, metres.
        renderer: What draws the frame.
        size: Output ``(width_px, height_px)``.

    Returns:
        A BGR image of that size.
    """
    width_px, height_px = size
    check_aspect(screen, width_px, height_px)
    matrix = view_projection_matrix(screen, eye_camera_m)
    return renderer.render(matrix, width_px, height_px)
