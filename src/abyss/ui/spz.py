"""Convert a 3D Gaussian Splatting PLY to SPZ with Spark's own encoder.

A trained PLY is about 240 bytes a splat; SPZ is about 18, which is what makes a
room of a million splats a download a phone can wait for. Spark ships the
encoder (``transcodeSpz``, compiled to WebAssembly), so the conversion runs in
headless Chromium on the served page rather than through a second SPZ library:
the file is written by the same code family that reads it.

Part of the harness, like :mod:`abyss.ui.browser`: it needs Playwright.
"""

import base64
from dataclasses import replace
from pathlib import Path

from loguru import logger as lg
from playwright.sync_api import Browser

from abyss.ui.browser import launch_chromium
from abyss.ui.browser import open_page
from abyss.ui.browser import wait_until
from abyss.ui.serve import ServeConfig
from abyss.ui.serve import running_server

TRANSCODE_JS = """async (name) => {
  const { transcodeSpz } = await import("@sparkjsdev/spark");
  const response = await fetch(`/splats/${encodeURIComponent(name)}`);
  if (!response.ok) throw new Error(`${name}: ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  const input = { fileBytes: bytes, pathOrUrl: name };
  const { fileBytes } = await transcodeSpz({ inputs: [input] });
  let text = "";
  for (let i = 0; i < fileBytes.length; i += 0x8000) {
    text += String.fromCharCode(...fileBytes.subarray(i, i + 0x8000));
  }
  return btoa(text);
}"""
"""Fetch the PLY from the server, transcode it, return the SPZ as base64."""

PAGE = "/phone_web/?screen=g7_internal&eye=0,0.1,0.6"
"""A fixed-eye page: it loads the import map Spark is resolved through."""


def transcode_spz(browser: Browser, base_url: str, name: str) -> bytes:
    """Transcode a splat the server holds under ``/splats/`` to SPZ.

    Args:
        browser: The browser.
        base_url: The server.
        name: The splat's file name in the served splats folder.

    Returns:
        The SPZ file's bytes.
    """
    page, _ = open_page(browser, f"{base_url}{PAGE}")
    wait_until(page, "window.abyss.status === 'done'")
    encoded = page.evaluate(TRANSCODE_JS, name)
    page.context.close()
    return base64.b64decode(encoded)


def ply_to_spz(ply: Path, spz: Path) -> None:
    """Convert a PLY splat to SPZ, keeping every splat and SH band.

    Args:
        ply: The trained splat; its folder is served for the conversion.
        spz: Where to write the result.
    """
    config = replace(ServeConfig.default(), splats_dir=ply.parent)
    with running_server(config) as base, launch_chromium(None) as browser:
        spz.write_bytes(transcode_spz(browser, base, ply.name))
    lg.info(
        f"{ply.name} {ply.stat().st_size / 1e6:.1f} MB -> "
        f"{spz.name} {spz.stat().st_size / 1e6:.1f} MB"
    )
