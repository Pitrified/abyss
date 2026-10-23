"""Test the UI server's routes, headers and refusals, against temporary folders."""

from collections.abc import Iterator
import gzip
import http.client
from pathlib import Path
import re
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
    """A UI folder serves its index, with the policy, revalidated every load."""
    response = get(base, "/phone_web/")
    assert response.status == 200
    assert response.getheader("Content-Type", "").startswith("text/html")
    assert response.getheader("Content-Security-Policy") == CONTENT_SECURITY_POLICY
    assert response.getheader("Cache-Control") == "no-cache"
    assert response.getheader("ETag")


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


def test_scene_route_needs_a_known_screen(base: str) -> None:
    """The scene is per panel: no screen is a 400, an unknown one a 404."""
    response = get(base, "/scene.json?screen=g7_internal")
    assert response.status == 200
    assert response.getheader("Content-Type") == "application/json"
    assert get(base, "/scene.json").status == 400
    assert get(base, "/scene.json?screen=nope").status == 404


def read_version(base: str) -> str:
    """Fetch /version.js and pull the string out of its one export."""
    parts = urlsplit(base)
    conn = http.client.HTTPConnection(parts.hostname or "", parts.port)
    conn.request("GET", "/version.js")
    response = conn.getresponse()
    assert response.status == 200
    assert response.getheader("Content-Type") == "text/javascript; charset=utf-8"
    assert response.getheader("Cache-Control") == "no-store"
    body = response.read().decode()
    match = re.fullmatch(r'export const VERSION = "ui ([0-9a-f]{8})";\n', body)
    assert match, body
    return match.group(1)


def test_version_follows_the_ui_files(base: str, config: ServeConfig) -> None:
    """Any edit under ui/ changes the version the page shows, with no restart."""
    before = read_version(base)
    assert read_version(base) == before
    (config.ui_root / "phone_web" / "index.html").write_text("<!doctype html><p>bye")
    assert read_version(base) != before


def fetch(
    base: str, path: str, headers: dict[str, str] | None = None
) -> tuple[http.client.HTTPResponse, bytes]:
    """Send a GET with headers and return the response with its body."""
    parts = urlsplit(base)
    conn = http.client.HTTPConnection(parts.hostname or "", parts.port)
    conn.request("GET", path, headers=headers or {})
    response = conn.getresponse()
    return response, response.read()


@pytest.fixture
def big_script(config: ServeConfig) -> bytes:
    """Write a script large enough to be compressed."""
    data = ("export const x = 1;\n" * 500).encode()
    (config.ui_root / "phone_web" / "big.js").write_bytes(data)
    return data


def test_files_revalidate_with_an_etag(base: str, big_script: bytes) -> None:
    """A file carries an ETag and no-cache; asked again with it, the answer is 304."""
    response, body = fetch(base, "/phone_web/big.js")
    assert body == big_script
    assert response.getheader("Cache-Control") == "no-cache"
    etag = response.getheader("ETag")
    assert etag
    again, empty = fetch(base, "/phone_web/big.js", {"If-None-Match": etag})
    assert again.status == 304
    assert empty == b""
    assert again.getheader("ETag") == etag


def test_a_changed_file_is_sent_again(
    base: str, config: ServeConfig, big_script: bytes
) -> None:
    """The ETag is the content: an edit makes the old one stale."""
    etag = fetch(base, "/phone_web/big.js")[0].getheader("ETag") or ""
    (config.ui_root / "phone_web" / "big.js").write_bytes(big_script + b"// edit\n")
    response, body = fetch(base, "/phone_web/big.js", {"If-None-Match": etag})
    assert response.status == 200
    assert body.endswith(b"// edit\n")


def test_text_is_gzipped_when_accepted(base: str, big_script: bytes) -> None:
    """Gzip on request, with its own ETag; plain bytes otherwise."""
    zipped, body = fetch(base, "/phone_web/big.js", {"Accept-Encoding": "gzip, br"})
    assert zipped.getheader("Content-Encoding") == "gzip"
    assert zipped.getheader("Vary") == "Accept-Encoding"
    assert gzip.decompress(body) == big_script
    assert len(body) < len(big_script) / 10
    plain, raw = fetch(base, "/phone_web/big.js")
    assert plain.getheader("Content-Encoding") is None
    assert raw == big_script
    assert zipped.getheader("ETag") != plain.getheader("ETag")
    refused, _ = fetch(base, "/phone_web/big.js", {"Accept-Encoding": "gzip;q=0"})
    assert refused.getheader("Content-Encoding") is None


def test_generated_routes_are_never_cached(base: str) -> None:
    """Config, scene and version are built per request, so no-store, no ETag."""
    for path in ("/config.json", "/scene.json?screen=g7_internal", "/version.js"):
        response, _ = fetch(base, path)
        assert response.getheader("Cache-Control") == "no-store", path
        assert response.getheader("ETag") is None, path
