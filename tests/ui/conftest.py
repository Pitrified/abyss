"""Shared fixtures for the UI tests: an npm-shaped package built in memory."""

from collections.abc import Callable
from dataclasses import dataclass
import io
import tarfile

import pytest

from abyss.ui.vendor import NpmPackage
from abyss.ui.vendor import sri_digest


def make_tarball(files: dict[str, bytes]) -> bytes:
    """Build an npm-shaped tarball, every file under ``package/``.

    Args:
        files: Path inside the package mapped to its bytes.

    Returns:
        The ``.tgz`` bytes.
    """
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(f"package/{name}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


@dataclass(frozen=True)
class FakePackage:
    """A package, its tarball, and the files in it."""

    pkg: NpmPackage
    tarball: bytes
    files: dict[str, bytes]


@pytest.fixture
def tarball_factory() -> Callable[[dict[str, bytes]], bytes]:
    """:func:`make_tarball`, for tests that build their own."""
    return make_tarball


@pytest.fixture
def fake_package() -> FakePackage:
    """Build a scoped package with two listed files and one unlisted."""
    files = {
        "lib/a.js": b"export const a = 1;\n",
        "lib/b.wasm": b"\x00asm",
        "README.md": b"unused",
    }
    tarball = make_tarball(files)
    pkg = NpmPackage(
        "@scope/thing",
        "1.2.3",
        sri_digest(tarball, "sha512"),
        ("lib/a.js", "lib/b.wasm"),
    )
    return FakePackage(pkg, tarball, files)
