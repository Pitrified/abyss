"""Test the vendor manifest, the integrity checks and the extraction.

No network: every tarball here is built in memory, so these run in `make test`.
The real manifest is parsed but never fetched.
"""

import json
from pathlib import Path

import pytest

from abyss.ui.vendor import LOCK_NAME
from abyss.ui.vendor import NpmPackage
from abyss.ui.vendor import RemoteFile
from abyss.ui.vendor import UnsafeArchiveMemberError
from abyss.ui.vendor import UnsafeUrlError
from abyss.ui.vendor import VendorFileMissingError
from abyss.ui.vendor import VendorIntegrityError
from abyss.ui.vendor import download
from abyss.ui.vendor import extract_package
from abyss.ui.vendor import load_manifest
from abyss.ui.vendor import package_is_current
from abyss.ui.vendor import sha256_hex
from abyss.ui.vendor import verified_file
from abyss.ui.vendor import verified_path


def cached(tmp_path: Path, fake) -> Path:
    """Fill a cache with the package, laid out the way ensure_vendor lays it out."""
    extract_package(fake.pkg, fake.tarball, tmp_path / "npm" / fake.pkg.folder_name)
    return tmp_path


def test_real_manifest_parses() -> None:
    """The committed manifest pins the two packages the page needs, and the portrait."""
    manifest = load_manifest()
    assert manifest.package("three").version
    tasks = manifest.package("@mediapipe/tasks-vision")
    assert tasks.integrity.startswith("sha512-")
    assert "vision_bundle.mjs" in tasks.files
    assert len(manifest.file("portrait").sha256) == 64


def test_scoped_tarball_url(fake_package) -> None:
    """A scoped package's tarball is named after the part after the slash."""
    pkg = fake_package.pkg
    assert (
        pkg.tarball_url == "https://registry.npmjs.org/@scope/thing/-/thing-1.2.3.tgz"
    )
    assert "/" not in pkg.folder_name


def test_extract_writes_only_listed_files_and_a_lock(
    tmp_path: Path, fake_package
) -> None:
    """Listed files come out with their hashes recorded; the rest stay packed."""
    pkg = fake_package.pkg
    lock = extract_package(pkg, fake_package.tarball, tmp_path)
    assert lock == {name: sha256_hex(fake_package.files[name]) for name in pkg.files}
    assert not (tmp_path / "README.md").exists()
    assert json.loads((tmp_path / LOCK_NAME).read_text()) == lock
    assert package_is_current(pkg, tmp_path)


def test_extract_refuses_a_tarball_that_does_not_match(
    tmp_path: Path, fake_package, tarball_factory
) -> None:
    """One changed byte in the tarball and nothing is extracted."""
    tampered = tarball_factory(
        {**fake_package.files, "lib/a.js": b"export const a = 2;\n"}
    )
    with pytest.raises(VendorIntegrityError, match="expected sha512-"):
        extract_package(fake_package.pkg, tampered, tmp_path)
    assert not any(tmp_path.iterdir())


def test_extract_names_a_missing_member(tmp_path: Path, fake_package) -> None:
    """A manifest listing a file the package does not ship fails by name."""
    real = fake_package.pkg
    pkg = NpmPackage(real.name, real.version, real.integrity, ("lib/missing.js",))
    with pytest.raises(VendorFileMissingError, match=r"lib/missing\.js"):
        extract_package(pkg, fake_package.tarball, tmp_path)


def test_verified_path_serves_an_intact_file(tmp_path: Path, fake_package) -> None:
    """An untouched cached file resolves to its path."""
    path = verified_path(fake_package.pkg, "lib/a.js", cached(tmp_path, fake_package))
    assert path.read_bytes() == fake_package.files["lib/a.js"]


def test_verified_path_refuses_an_edited_cache(tmp_path: Path, fake_package) -> None:
    """A file changed after extraction is refused, not served."""
    cache = cached(tmp_path, fake_package)
    (cache / "npm" / fake_package.pkg.folder_name / "lib" / "a.js").write_bytes(
        b"alert(1)"
    )
    with pytest.raises(VendorIntegrityError):
        verified_path(fake_package.pkg, "lib/a.js", cache)


def test_verified_path_refuses_unlisted_files(tmp_path: Path, fake_package) -> None:
    """Only manifest files are served, even if something else sits in the folder."""
    cache = cached(tmp_path, fake_package)
    (cache / "npm" / fake_package.pkg.folder_name / "extra.js").write_bytes(b"")
    with pytest.raises(VendorFileMissingError):
        verified_path(fake_package.pkg, "extra.js", cache)


def test_verified_file_checks_the_pin(tmp_path: Path) -> None:
    """A plain file is missing, then served, then refused once it changes."""
    data = b"\xff\xd8 not really a jpeg"
    item = RemoteFile("portrait", "https://example.invalid/p.jpg", sha256_hex(data))
    with pytest.raises(VendorFileMissingError):
        verified_file(item, tmp_path)
    (tmp_path / "files").mkdir()
    (tmp_path / "files" / "portrait.jpg").write_bytes(data)
    assert verified_file(item, tmp_path).name == "portrait.jpg"
    (tmp_path / "files" / "portrait.jpg").write_bytes(data + b"!")
    with pytest.raises(VendorIntegrityError):
        verified_file(item, tmp_path)


def test_manifest_refuses_paths_that_leave_the_package(tmp_path: Path) -> None:
    """A manifest cannot name a file outside its package's folder."""
    manifest = tmp_path / "vendor.toml"
    manifest.write_text(
        '[npm.x]\nversion = "1"\nintegrity = "sha512-x"\nfiles = ["../../etc/passwd"]\n'
    )
    with pytest.raises(UnsafeArchiveMemberError):
        load_manifest(manifest)


def test_download_refuses_plain_http() -> None:
    """Nothing is fetched over plain http."""
    with pytest.raises(UnsafeUrlError):
        download("http://registry.npmjs.org/x")
