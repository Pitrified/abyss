"""Serve the browser UIs, their vendored files, the face model and test clips.

One static server for every UI under ``ui/``, standard library only. Routes:

- ``/<ui>/...``: the UI's own files, e.g. ``/phone_web/index.html``. ``/``
  redirects to the phone UI.
- ``/vendor/<npm name>/<path>``: a file from a package in ``ui/vendor.toml``,
  at the pinned version, checked against its lock on every request. The URL
  carries no version, so bumping a pin touches the manifest and nothing else.
- ``/models/face_landmarker.task``: the same model file the Python pipeline
  loads, through pose-tools' ``ModelManager``.
- ``/clips/<name>``: generated clips from ``build/clips/``.
- ``/config.json?camera=&viewer=&screen=``: device configs and constants, see
  :mod:`abyss.ui.page_config`.

Every response carries a Content-Security-Policy whose ``connect-src 'self'``
keeps the page from reaching any host but this one. That is not hygiene for its
own sake: ``@mediapipe/tasks-vision`` posts usage logs to
``odml.pa.googleapis.com`` once a minute, and the policy is what stops it.
``'wasm-unsafe-eval'`` is the one relaxation, because MediaPipe is WebAssembly.

Binds ``127.0.0.1`` by default, which is a secure context for the camera API.
Run as ``python -m abyss.ui.serve``, which is what ``make serve`` does.
"""

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import threading
from typing import ClassVar
from urllib.parse import parse_qs
from urllib.parse import unquote
from urllib.parse import urlsplit

from loguru import logger as lg
from pose_tools.landmark.model_manager import ModelManager

from abyss.params.abyss_devices import UnknownDeviceError
from abyss.params.abyss_params import get_abyss_paths
from abyss.ui.page_config import DEFAULT_CAMERA
from abyss.ui.page_config import DEFAULT_VIEWER
from abyss.ui.page_config import page_config
from abyss.ui.vendor import Manifest
from abyss.ui.vendor import VendorFileMissingError
from abyss.ui.vendor import VendorIntegrityError
from abyss.ui.vendor import load_manifest
from abyss.ui.vendor import verified_path

DEFAULT_UI = "phone_web"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

CSP_DIRECTIVES = (
    "default-src 'self'",
    "script-src 'self' 'wasm-unsafe-eval'",
    "connect-src 'self'",
    "img-src 'self' blob: data:",
    "media-src 'self' blob: mediastream:",
    "worker-src 'self' blob:",
    "style-src 'self'",
)
CONTENT_SECURITY_POLICY = "; ".join(CSP_DIRECTIVES)

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".wasm": "application/wasm",
    ".task": "application/octet-stream",
    ".mjpeg": "video/x-motion-jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
}


@dataclass(frozen=True)
class ServeConfig:
    """Where the server reads from. Defaults are the repository's folders."""

    ui_root: Path
    vendor_cache: Path
    clips_dir: Path
    manifest: Manifest
    model_path: Path | None = None

    @classmethod
    def default(cls) -> "ServeConfig":
        """Build the config the repository uses.

        Returns:
            Folders from :func:`get_abyss_paths` and the parsed manifest. The
            model is resolved lazily, on its first request.
        """
        paths = get_abyss_paths()
        return cls(paths.ui_fol, paths.vendor_fol, paths.clips_fol, load_manifest())


def resolve_inside(root: Path, relative: str) -> Path | None:
    """Join a URL path onto a folder without letting it climb out.

    Args:
        root: The folder.
        relative: The decoded URL path below it.

    Returns:
        The resolved path, or None if it would leave the folder.
    """
    root = root.resolve()
    target = (root / relative.lstrip("/")).resolve()
    return target if target.is_relative_to(root) else None


def split_vendor_path(rest: str) -> tuple[str, str] | None:
    """Split ``<npm name>/<member>``, where a scoped name has two segments.

    Args:
        rest: The URL path after ``/vendor/``.

    Returns:
        Package name and member path, or None if there is no member.
    """
    parts = rest.split("/")
    cut = 2 if parts[0].startswith("@") else 1
    if len(parts) <= cut:
        return None
    return "/".join(parts[:cut]), "/".join(parts[cut:])


class UiRequestHandler(BaseHTTPRequestHandler):
    """Routes requests to the folders in :attr:`config`."""

    config: ClassVar[ServeConfig]

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - name fixed by the base class
        """Send the base class's access log to loguru instead of stderr.

        Args:
            format: printf-style format from the base class.
            *args: Its arguments.
        """
        lg.info(f"serve: {format % args}")

    def do_GET(self) -> None:
        """Answer a GET."""
        path = unquote(urlsplit(self.path).path)
        if path in {"", "/"}:
            self.redirect(f"/{DEFAULT_UI}/")
        elif path.startswith("/vendor/"):
            self.send_vendor(path.removeprefix("/vendor/"))
        elif path == "/models/face_landmarker.task":
            self.send_model()
        elif path == "/config.json":
            self.send_config(urlsplit(self.path).query)
        elif path.startswith("/clips/"):
            self.send_inside(self.config.clips_dir, path.removeprefix("/clips/"))
        else:
            self.send_inside(self.config.ui_root, path)

    def redirect(self, location: str) -> None:
        """Send a 302.

        Args:
            location: Where to.
        """
        self.send_response(HTTPStatus.FOUND)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def send_vendor(self, rest: str) -> None:
        """Send a vendored file, refusing it if it fails its integrity check.

        Args:
            rest: ``<npm name>/<member>``.
        """
        split = split_vendor_path(rest)
        if split is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        name, member = split
        try:
            pkg = self.config.manifest.package(name)
            path = verified_path(pkg, member, self.config.vendor_cache)
        except VendorFileMissingError as exc:
            self.send_error(HTTPStatus.NOT_FOUND, str(exc))
            return
        except VendorIntegrityError as exc:
            lg.error(f"serve: refusing {rest}: {exc}")
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "integrity check failed")
            return
        self.send_file(path)

    def send_config(self, query: str) -> None:
        """Send the page config for the devices named in the query string.

        Args:
            query: The raw query string.
        """
        params = {key: values[-1] for key, values in parse_qs(query).items()}
        try:
            config = page_config(
                camera=params.get("camera", DEFAULT_CAMERA),
                viewer=params.get("viewer", DEFAULT_VIEWER),
                screen=params.get("screen"),
            )
        except UnknownDeviceError as exc:
            self.send_error(HTTPStatus.NOT_FOUND, str(exc))
            return
        self.send_bytes(json.dumps(config).encode(), CONTENT_TYPES[".json"])

    def send_model(self) -> None:
        """Send the face landmarker model the Python pipeline uses."""
        path = self.config.model_path or ModelManager().ensure_model("face_landmarker")
        self.send_file(path)

    def send_inside(self, root: Path, relative: str) -> None:
        """Send a file from under a folder, ``index.html`` for a folder.

        Args:
            root: The folder.
            relative: The URL path below it.
        """
        target = resolve_inside(root, relative)
        if target is None:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if target.is_dir():
            if not relative.endswith("/"):
                self.redirect(f"{self.path}/")
                return
            target /= "index.html"
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_file(target)

    def send_file(self, path: Path) -> None:
        """Send a file with its content type and no caching.

        Args:
            path: A file that exists.
        """
        content_type = CONTENT_TYPES.get(path.suffix, "application/octet-stream")
        self.send_bytes(path.read_bytes(), content_type)

    def send_bytes(self, data: bytes, content_type: str) -> None:
        """Send a body with its content type and no caching.

        Args:
            data: The body.
            content_type: Its MIME type.
        """
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def end_headers(self) -> None:
        """Add the CSP to every response, error pages included, then finish."""
        self.send_header("Content-Security-Policy", CONTENT_SECURITY_POLICY)
        super().end_headers()


def make_server(
    config: ServeConfig | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> ThreadingHTTPServer:
    """Build a server bound to an address, not yet serving.

    Args:
        config: Folders to serve. Defaults to :meth:`ServeConfig.default`.
        host: Address to bind.
        port: Port to bind; 0 picks a free one.

    Returns:
        The bound server.
    """
    handler = type("BoundUiRequestHandler", (UiRequestHandler,), {})
    handler.config = config or ServeConfig.default()
    return ThreadingHTTPServer((host, port), handler)


@contextmanager
def running_server(
    config: ServeConfig | None = None,
    host: str = DEFAULT_HOST,
    port: int = 0,
) -> Iterator[str]:
    """Serve on a background thread for the duration of a ``with`` block.

    Args:
        config: Folders to serve.
        host: Address to bind.
        port: Port to bind; 0, the default, picks a free one.

    Yields:
        The base URL, without a trailing slash.
    """
    server = make_server(config, host, port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    bound_host, bound_port = server.server_address[:2]
    try:
        yield f"http://{bound_host}:{bound_port}"
    finally:
        server.shutdown()
        server.server_close()


def main() -> None:
    """Serve until interrupted."""
    parser = argparse.ArgumentParser(description="Serve the browser UIs.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    server = make_server(host=args.host, port=args.port)
    lg.info(f"Serving on http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        lg.info("Stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
