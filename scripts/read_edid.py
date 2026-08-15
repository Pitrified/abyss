"""Read the physical size of the connected display, from its own EDID.

Every display stores a 128-byte EDID block describing itself, which the kernel
exposes per connector under ``/sys/class/drm/``. The size it reports is exact,
so this replaces holding a ruler against a panel.

Run it once per machine and paste the numbers into ``ScreenConfig`` in
``abyss.params.abyss_devices``. Nothing in the package reads ``/sys`` at run
time: the answer is a config literal, not a lookup.

    uv run --no-sync python scripts/read_edid.py

Two traps this script exists to have solved already:

- Every ``edid`` node reports a size of 0 bytes to ``stat``, including the live
  one, because sysfs generates the contents on read. Filtering on file size
  discards the panel along with the empty connectors.
- Which connector is real is answered by the neighbouring ``status`` file, not
  by guessing at the name.
"""

from pathlib import Path

DRM_FOL = Path("/sys/class/drm")

EDID_HEADER = bytes([0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00])
"""Fixed header every EDID block starts with."""

DTD_OFFSET = 54
"""Byte offset of the first detailed timing descriptor."""

BASIC_SIZE_OFFSET = 21
"""Byte offset of the basic display size, in whole centimetres."""


class EdidParseError(ValueError):
    """Raised when a block does not look like EDID."""

    def __init__(self, connector: str) -> None:
        """Initialise with the offending connector.

        Args:
            connector: Name of the connector the block came from.
        """
        super().__init__(f"{connector}: not an EDID block (bad header)")


def read_size_mm(block: bytes, connector: str) -> tuple[int, int]:
    """Pull the physical size out of an EDID block.

    Reads the detailed timing descriptor rather than the basic block, because
    the basic block rounds to whole centimetres: 31x17 cm where the descriptor
    says 309x173 mm.

    Args:
        block: The bytes read from the connector's ``edid`` node.
        connector: Name of the connector, for error messages.

    Returns:
        Width and height of the active area in millimetres.

    Raises:
        EdidParseError: If the block does not start with the EDID header.
    """
    if not block.startswith(EDID_HEADER):
        raise EdidParseError(connector)

    dtd = block[DTD_OFFSET : DTD_OFFSET + 18]
    # Bytes 12 and 13 hold the low bits of width and height; byte 14 holds the
    # high nibble of each.
    width_mm = ((dtd[14] >> 4) << 8) | dtd[12]
    height_mm = ((dtd[14] & 0x0F) << 8) | dtd[13]
    if width_mm and height_mm:
        return width_mm, height_mm

    # A descriptor may leave the size undefined, in which case the coarse
    # figures in the basic block are all there is.
    print(f"{connector}: no size in the timing descriptor, falling back to cm")
    return block[BASIC_SIZE_OFFSET] * 10, block[BASIC_SIZE_OFFSET + 1] * 10


def main() -> None:
    """Report every connected display."""
    if not DRM_FOL.is_dir():
        print(f"No {DRM_FOL}: this is not a Linux machine with a DRM driver")
        return

    found = 0
    for connector in sorted(DRM_FOL.glob("card*-*")):
        status_file = connector / "status"
        edid_file = connector / "edid"
        if not status_file.is_file() or not edid_file.is_file():
            continue
        status = status_file.read_text().strip()
        # Not stat(): sysfs reports 0 bytes for all of them, live or not.
        block = edid_file.read_bytes()
        if status != "connected" or not block:
            print(f"{connector.name:<16} {status}")
            continue

        width_mm, height_mm = read_size_mm(block, connector.name)
        diagonal_in = ((width_mm**2 + height_mm**2) ** 0.5) / 25.4
        print(
            f"{connector.name:<16} connected  "
            f"{width_mm}x{height_mm} mm  ({diagonal_in:.1f} in diagonal)"
        )
        print(f"    width_m={width_mm / 1000}, height_m={height_mm / 1000}")
        found += 1

    if not found:
        print("No connected display found")


if __name__ == "__main__":
    main()
