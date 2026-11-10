"""PLY to SPZ through Spark: the converted splat loads with every splat."""

import gzip
from pathlib import Path

from playwright.sync_api import Browser

from abyss.ui.splat import TEST_SPLAT_NAME
from abyss.ui.splat import TEST_SPLAT_POINTS
from abyss.ui.spz import transcode_spz

SPZ_MAGIC = b"NGSP"
"""An SPZ file is gzip; its first bytes once unpacked."""


def test_converted_splat_keeps_its_splats(
    browser: Browser, base_url: str, splats: Path
) -> None:
    """The SPZ of the test splat is a valid SPZ and loads with every splat."""
    spz = splats / f"{TEST_SPLAT_NAME}_converted.spz"
    spz.write_bytes(transcode_spz(browser, base_url, f"{TEST_SPLAT_NAME}.ply"))
    assert gzip.decompress(spz.read_bytes())[:4] == SPZ_MAGIC

    page = browser.new_page()
    page.goto(f"{base_url}/phone_web/?screen=g7_internal&eye=0,0.1,0.6")
    count = page.evaluate(
        """async (name) => {
          const { PackedSplats } = await import("@sparkjsdev/spark");
          const splats = new PackedSplats({ url: `/splats/${name}` });
          await splats.initialized;
          return splats.numSplats;
        }""",
        spz.name,
    )
    page.close()
    assert count == len(TEST_SPLAT_POINTS)
