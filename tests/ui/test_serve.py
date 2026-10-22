"""Test the UI server's routes, headers and refusals, against temporary folders."""

from collections.abc import Iterator
import http.client
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from abyss.ui.serve import CONTENT_SECURITY_POLICY
from abyss.ui.serve import ServeConfig
from abyss.ui.serve import running_server
from abyss.ui.serve import split_vendor_path
from abyss.ui.vendor import Manifest
from abyss.ui.vendor import extract_package


@pytest.fixture
def config(tmp_path: Path, fake_package) -> ServeConfig:
    """Lay out a UI, a vendor cache holding the test package, a clip and a model."""
    ui = tmp_path / "ui"
    (ui / "phone_web").mkdir(parents=True)
    (ui / "phone_web" / "index.html").write_text("<!doctype html><p>hi")
    (tmp_path / "secret.txt").write_text("outside")
    cache = tmp_path / "cache"
    pkg = fake_package.pkg
    extract_package(pkg, fake_package.tarball, cache / "npm" / pkg.folder_name)
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "c.mjpeg").write_bytes(b"\xff\xd8\xff")
    model = tmp_path / "face_landmarker.task"
    model.write_bytes(b"model")
    return ServeConfig(ui, cache, clips, Manifest((pkg,), ()), model_path=model)


@pytest.fixture
def base(config: ServeConfig) -> Iterator[str]:
    """Serve the temporary folders on a free port."""
    with running_server(config) as url:
        yield url


def get(base: str, path: str) -> http.client.HTTPResponse:
    """Send a raw GET, so paths like ``..`` reach the server unnormalised."""
    parts = urlsplit(base)
    conn = http.client.HTTPConnection(parts.hostname or "", parts.port)
    conn.request("GET", path)
    response = conn.getresponse()
    response.read()
    return response


def test_root_redirects_to_the_phone_ui(base: str) -> None:
    """The bare host lands on the phone UI."""
    response = get(base, "/")
    assert response.status == 302
    assert response.getheader("Location") == "/phone_web/"


def test_ui_folder_serves_index_with_csp(base: str) -> None:
    """A UI folder serves its index, with the policy and no caching."""
    response = get(base, "/phone_web/")
    assert response.status == 200
    assert response.getheader("Content-Type", "").startswith("text/html")
    assert response.getheader("Content-Security-Policy") == CONTENT_SECURITY_POLICY
    assert response.getheader("Cache-Control") == "no-store"


def test_csp_blocks_other_hosts() -> None:
    """The point of the policy: fetches go to this server only."""
    assert "connect-src 'self'" in CONTENT_SECURITY_POLICY
    assert "wasm-unsafe-eval" in CONTENT_SECURITY_POLICY


def test_errors_carry_the_csp_too(base: str) -> None:
    """An error page is a page too, and gets the same policy."""
    response = get(base, "/phone_web/nope.js")
    assert response.status == 404
    assert response.getheader("Content-Security-Policy") == CONTENT_SECURITY_POLICY


def test_traversal_is_refused(base: str) -> None:
    """A path that climbs out of a served folder is refused."""
    assert get(base, "/../secret.txt").status == 403
    assert get(base, "/clips/../secret.txt").status == 403


def test_scoped_vendor_file_is_served_with_its_type(base: str) -> None:
    """A scoped package's file is found, and wasm goes out as wasm."""
    response = get(base, "/vendor/@scope/thing/lib/b.wasm")
    assert response.status == 200
    assert response.getheader("Content-Type") == "application/wasm"


def test_unlisted_or_unknown_vendor_paths_are_404(base: str) -> None:
    """Only manifest files are served from the vendor cache."""
    assert get(base, "/vendor/@scope/thing/README.md").status == 404
    assert get(base, "/vendor/other/x.js").status == 404
    assert get(base, "/vendor/@scope/thing").status == 404


def test_tampered_vendor_file_is_refused(
    config: ServeConfig, base: str, fake_package
) -> None:
    """A cache edited after extraction answers 500, never the edited bytes."""
    (
        config.vendor_cache / "npm" / fake_package.pkg.folder_name / "lib" / "a.js"
    ).write_text("alert(1)")
    assert get(base, "/vendor/@scope/thing/lib/a.js").status == 500


def test_model_and_clips(base: str) -> None:
    """The model and the clips have their own routes."""
    assert get(base, "/models/face_landmarker.task").status == 200
    response = get(base, "/clips/c.mjpeg")
    assert response.status == 200
    assert response.getheader("Content-Type") == "video/x-motion-jpeg"


def test_split_vendor_path() -> None:
    """Scoped names take two segments, plain names one."""
    assert split_vendor_path("three/build/three.module.js") == (
        "three",
        "build/three.module.js",
    )
    assert split_vendor_path("@a/b/c/d.js") == ("@a/b", "c/d.js")
    assert split_vendor_path("@a/b") is None


def test_config_route_serves_json_and_names_unknown_devices(base: str) -> None:
    """Known devices give the document, an unknown one a 404, never a default."""
    response = get(base, "/config.json?camera=g7_webcam")
    assert response.status == 200
    assert response.getheader("Content-Type") == "application/json"
    assert get(base, "/config.json?camera=nope").status == 404
