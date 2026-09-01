"""The sink that puts frames on the panel.

**The only module in the package that needs a display.** It is separate for
exactly that reason: `cv.imshow` and `cv.waitKey` work on g7 and not on g4, so
nothing here is imported by the test suite and nothing here runs over ssh.

Fullscreen is a geometric requirement rather than a presentation choice.
`ScreenConfig` describes the whole panel, 344 by 193 mm with the camera 100.5 mm
above its centre, and the frustum is built from that rectangle. A window
floating on the desktop would be a different, smaller, differently placed
window onto the world, and every number in the config would then describe a
rectangle that is not on screen.

Key handling lives here because this is the object holding the event loop, and
`cv.waitKey` is the only thing that pumps it: without a `waitKey` call after
`imshow`, the window never paints. The loop asks about keys through two
callables rather than by knowing what kind of sink it has.
"""

import cv2 as cv
from loguru import logger as lg
import numpy as np

from abyss.sink.base import check_size

QUIT_KEYS = (ord("q"), 27)
"""Keys that end a live run: ``q`` and escape."""

RESET_KEY = ord("r")
"""Key that re-runs the head scale bootstrap (Q23)."""

MARK_KEY = ord("m")
"""Key that records the current reading to the log.

The tape measure check needs a number read while sitting at a measured
distance, and the run is fullscreen: there is no terminal to look at, the text
is small at a metre, and remembering three readings while holding still is how
a measurement becomes an impression. Pressing a key puts it in the scrollback
instead, where it can be copied afterwards.
"""

WAIT_MS = 1
"""Milliseconds to give the window's event loop per frame.

Not a frame-pacing knob. The loop owns pacing (Q26) and runs as fast as the
source allows; this is the shortest wait that still lets the window paint and
report a keypress.
"""


class WindowSink:
    """Show frames fullscreen on the panel.

    Args:
        size: The ``(width_px, height_px)`` frames will be handed in at.
        name: Window title, also used in logs.
    """

    def __init__(self, size: tuple[int, int], name: str = "abyss") -> None:
        """Open the window and make it fullscreen.

        Args:
            size: The ``(width_px, height_px)`` frames will be handed in at.
            name: Window title, also used in logs.
        """
        self._size = size
        self.name = name
        self._quit = False
        self._reset = False
        self._mark = False
        cv.namedWindow(name, cv.WINDOW_NORMAL)
        cv.setWindowProperty(name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        lg.info(f"Opened fullscreen window {name!r} for {size[0]}x{size[1]} frames")

    @property
    def size(self) -> tuple[int, int]:
        """Frame size this sink accepts, as ``(width_px, height_px)``.

        The size it was given rather than one it discovered. The render has to
        match the panel the config describes, so a window that came up at some
        other size is a setup problem to be seen and fixed, not a resolution to
        silently adopt.
        """
        return self._size

    def write(self, frame: np.ndarray) -> None:
        """Show one frame and pump the window's event loop.

        Args:
            frame: A BGR image of exactly :attr:`size`.
        """
        check_size(self._size, frame)
        cv.imshow(self.name, frame)
        key = cv.waitKey(WAIT_MS) & 0xFF
        if key in QUIT_KEYS:
            self._quit = True
        elif key == RESET_KEY:
            self._reset = True
        elif key == MARK_KEY:
            self._mark = True

    def close(self) -> None:
        """Destroy the window."""
        cv.destroyWindow(self.name)
        lg.info(f"Closed window {self.name!r}")

    @property
    def quit_requested(self) -> bool:
        """Whether a quit key has been pressed. Stays true once set."""
        return self._quit

    def take_mark_request(self) -> bool:
        """Whether a reading was asked for, clearing the request.

        Returns:
            True if the mark key has been pressed since the last call.
        """
        asked, self._mark = self._mark, False
        return asked

    def take_reset_request(self) -> bool:
        """Whether a reset was asked for, clearing the request.

        Consumed rather than latched: a reset is an event that happens once,
        while a quit is a state the run does not come back from.

        Returns:
            True if the reset key has been pressed since the last call.
        """
        asked, self._reset = self._reset, False
        return asked
