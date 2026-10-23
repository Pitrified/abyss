"""Third-party files for the UIs: a manifest in git, a checked cache outside it.

The manifest, ``ui/vendor.toml``, pins every npm package by exact version and by
the integrity hash the npm registry publishes for its tarball, and every plain
file by URL and sha256. :func:`ensure_vendor` downloads what is missing into
``cache/vendor/``, refuses anything whose hash does not match, and extracts only
the files the manifest lists.

npm publishes one hash per tarball, not per file, so a file extracted from a
verified tarball is recorded in a lock beside it with its own sha256. The server
checks every file against that lock before handing it out
(:func:`verified_path`), so a cache edited after the fact is refused rather than
served.

Downloads go through ``urllib`` because it honours ``HTTPS_PROXY``, which the
cloud sessions need, and every URL is checked to be ``https`` before it is
opened.

Run as ``python -m abyss.ui.vendor``, which is what ``make vendor`` does.
"""

import base64
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tomllib
import urllib.request

from loguru import logger as lg

from abyss.params.abyss_params import get_abyss_paths

NPM_REGISTRY = "https://registry.npmjs.org"
LOCK_NAME = "files.lock.json"
DOWNLOAD_TIMEOUT_S = 120.0


class VendorIntegrityError(RuntimeError):
    """Raised when downloaded or cached bytes do not match their pinned hash."""

    def __init__(self, what: str, expected: str, actual: str) -> None:
        """Initialise with the item and both hashes.

        Args:
            what: The package, file or URL that failed.
            expected: The hash the manifest or lock pins.
            actual: The hash of the bytes in hand.
        """
        super().__init__(
            f"Integrity check failed for {what}: expected {expected}, got {actual}. "
            "Delete the cached copy and run `make vendor` again; if it fails "
            "again, the pin in ui/vendor.toml is wrong or the source changed."
        )


class VendorFileMissingError(FileNotFoundError):
    """Raised when a file is requested that the manifest does not provide."""

    def __init__(self, name: str, member: str | None = None) -> None:
        """Initialise with the missing item.

        Args:
            name: Package name or plain file key.
            member: Path inside the package, when one was asked for.
        """
        what = f"{name}/{member}" if member else name
        super().__init__(
            f"{what!r} is not in the manifest or not in the vendor cache. "
            "Check ui/vendor.toml, then run `make vendor`."
        )


class UnsafeUrlError(ValueError):
    """Raised when a manifest URL is not plain https."""

    def __init__(self, url: str) -> None:
        """Initialise with the URL.

        Args:
            url: The offending URL.
        """
        super().__init__(f"Only https URLs are fetched, got {url!r}")


class UnsafeArchiveMemberError(ValueError):
    """Raised when a manifest names a path that escapes its package folder."""

    def __init__(self, name: str) -> None:
        """Initialise with the member name.

        Args:
            name: The path from the manifest.
        """
        super().__init__(
            f"Refusing archive path {name!r}: it leaves the package folder"
        )


@dataclass(frozen=True)
class NpmPackage:
    """One npm package pinned in the manifest."""

    name: str
    version: str
    integrity: str
    files: tuple[str, ...]

    @property
    def tarball_url(self) -> str:
        """Registry URL of the tarball, scoped names included."""
        basename = self.name.rsplit("/", maxsplit=1)[-1]
        return f"{NPM_REGISTRY}/{self.name}/-/{basename}-{self.version}.tgz"

    @property
    def folder_name(self) -> str:
        """Cache folder name, with the version, safe for a filesystem."""
        return f"{self.name.replace('/', '__')}@{self.version}"


@dataclass(frozen=True)
class RemoteFile:
    """One plain file pinned in the manifest."""

    name: str
    url: str
    sha256: str

    @property
    def filename(self) -> str:
        """Cache file name: the manifest key plus the URL's extension."""
        return self.name + Path(self.url).suffix


@dataclass(frozen=True)
class Manifest:
    """The parsed ``ui/vendor.toml``."""

    packages: tuple[NpmPackage, ...]
    files: tuple[RemoteFile, ...]

    def package(self, name: str) -> NpmPackage:
        """Look up a package by npm name.

        Args:
            name: npm package name, scope included.

        Returns:
            The pinned package.

        Raises:
            VendorFileMissingError: If the manifest does not list it.
        """
        for pkg in self.packages:
            if pkg.name == name:
                return pkg
        raise VendorFileMissingError(name)

    def file(self, name: str) -> RemoteFile:
        """Look up a plain file by manifest key.

        Args:
            name: The key under ``[file]``.

        Returns:
            The pinned file.

        Raises:
            VendorFileMissingError: If the manifest does not list it.
        """
        for item in self.files:
            if item.name == name:
                return item
        raise VendorFileMissingError(name)


def load_manifest(path: Path | None = None) -> Manifest:
    """Parse the manifest.

    Args:
        path: Manifest to read. Defaults to ``ui/vendor.toml``.

    Returns:
        The manifest.

    Raises:
        UnsafeArchiveMemberError: If a listed file would leave its package.
    """
    path = path or get_abyss_paths().vendor_manifest
    data = tomllib.loads(path.read_text())
    packages = []
    for name, spec in data.get("npm", {}).items():
        files = tuple(spec["files"])
        for member in files:
            if member.startswith("/") or ".." in Path(member).parts:
                raise UnsafeArchiveMemberError(member)
        packages.append(NpmPackage(name, spec["version"], spec["integrity"], files))
    files = tuple(
        RemoteFile(name, spec["url"], spec["sha256"])
        for name, spec in data.get("file", {}).items()
    )
    return Manifest(tuple(packages), files)


def sri_digest(data: bytes, algorithm: str) -> str:
    """Hash bytes in the Subresource Integrity format npm uses.

    Args:
        data: The bytes.
        algorithm: ``sha512`` or another hashlib name.

    Returns:
        ``"<algorithm>-<base64 digest>"``.
    """
    digest = hashlib.new(algorithm, data).digest()
    return f"{algorithm}-{base64.b64encode(digest).decode()}"


def sha256_hex(data: bytes) -> str:
    """Hex sha256 of bytes.

    Args:
        data: The bytes.

    Returns:
        Lowercase hex digest.
    """
    return hashlib.sha256(data).hexdigest()


def download(url: str) -> bytes:
    """Fetch a URL over https.

    Args:
        url: Must start with ``https://``.

    Returns:
        The response body.

    Raises:
        UnsafeUrlError: If the URL is not https.
    """
    if not url.startswith("https://"):
        raise UnsafeUrlError(url)
    lg.info(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as response:  # noqa: S310 - scheme checked above
        return response.read()


def extract_package(pkg: NpmPackage, tarball: bytes, dest: Path) -> dict[str, str]:
    """Check a tarball against its pin and extract the listed files.

    Args:
        pkg: The pinned package.
        tarball: The ``.tgz`` bytes.
        dest: Folder to extract into; created if missing.

    Returns:
        The lock: each extracted file's path mapped to its sha256.

    Raises:
        VendorIntegrityError: If the tarball does not match the pin.
        VendorFileMissingError: If a listed file is not in the tarball.
    """
    algorithm = pkg.integrity.split("-", maxsplit=1)[0]
    actual = sri_digest(tarball, algorithm)
    if actual != pkg.integrity:
        raise VendorIntegrityError(pkg.tarball_url, pkg.integrity, actual)
    lock = {}
    with tarfile.open(fileobj=io.BytesIO(tarball), mode="r:gz") as archive:
        for member_name in pkg.files:
            try:
                member = archive.getmember(f"package/{member_name}")
            except KeyError:
                raise VendorFileMissingError(pkg.name, member_name) from None
            handle = archive.extractfile(member)
            if handle is None:
                raise VendorFileMissingError(pkg.name, member_name)
            data = handle.read()
            target = dest / member_name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            lock[member_name] = sha256_hex(data)
    (dest / LOCK_NAME).write_text(json.dumps(lock, indent=1, sort_keys=True))
    return lock


def package_is_current(pkg: NpmPackage, folder: Path) -> bool:
    """Whether a package folder holds exactly the files the manifest lists.

    Args:
        pkg: The pinned package.
        folder: Its cache folder.

    Returns:
        True when the lock exists, names the listed files, and they exist.
    """
    lock_path = folder / LOCK_NAME
    if not lock_path.is_file():
        return False
    lock = json.loads(lock_path.read_text())
    return set(lock) == set(pkg.files) and all((folder / f).is_file() for f in lock)


def ensure_vendor(manifest: Manifest | None = None, cache: Path | None = None) -> None:
    """Download and check everything the manifest lists that is not cached yet.

    Args:
        manifest: Defaults to :func:`load_manifest`.
        cache: Defaults to ``cache/vendor/``.

    Raises:
        VendorIntegrityError: If a plain file does not match its pin.
    """
    manifest = manifest or load_manifest()
    cache = cache or get_abyss_paths().vendor_fol
    for pkg in manifest.packages:
        folder = cache / "npm" / pkg.folder_name
        if package_is_current(pkg, folder):
            lg.debug(f"{pkg.name}@{pkg.version} already cached")
            continue
        extract_package(pkg, download(pkg.tarball_url), folder)
        lg.info(f"Cached {pkg.name}@{pkg.version}: {len(pkg.files)} files")
    for item in manifest.files:
        target = cache / "files" / item.filename
        if target.is_file() and sha256_hex(target.read_bytes()) == item.sha256:
            lg.debug(f"{item.name} already cached")
            continue
        data = download(item.url)
        actual = sha256_hex(data)
        if actual != item.sha256:
            raise VendorIntegrityError(item.url, item.sha256, actual)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        lg.info(f"Cached {item.name}")


def verified_path(pkg: NpmPackage, member: str, cache: Path | None = None) -> Path:
    """Resolve a package file in the cache, checking it against the lock.

    Args:
        pkg: The pinned package.
        member: Path inside the package, as listed in the manifest.
        cache: Defaults to ``cache/vendor/``.

    Returns:
        The path of a file whose bytes match the lock.

    Raises:
        VendorFileMissingError: If the file is not listed or not cached.
        VendorIntegrityError: If the cached bytes no longer match.
    """
    cache = cache or get_abyss_paths().vendor_fol
    folder = cache / "npm" / pkg.folder_name
    lock_path = folder / LOCK_NAME
    if member not in pkg.files or not lock_path.is_file():
        raise VendorFileMissingError(pkg.name, member)
    expected = json.loads(lock_path.read_text()).get(member)
    path = folder / member
    if expected is None or not path.is_file():
        raise VendorFileMissingError(pkg.name, member)
    actual = sha256_hex(path.read_bytes())
    if actual != expected:
        raise VendorIntegrityError(str(path), expected, actual)
    return path


def verified_file(item: RemoteFile, cache: Path | None = None) -> Path:
    """Resolve a plain file in the cache, checking it against its pin.

    Args:
        item: The pinned file.
        cache: Defaults to ``cache/vendor/``.

    Returns:
        The path of a file whose bytes match the pin.

    Raises:
        VendorFileMissingError: If it is not cached.
        VendorIntegrityError: If the cached bytes do not match.
    """
    cache = cache or get_abyss_paths().vendor_fol
    path = cache / "files" / item.filename
    if not path.is_file():
        raise VendorFileMissingError(item.name)
    actual = sha256_hex(path.read_bytes())
    if actual != item.sha256:
        raise VendorIntegrityError(str(path), item.sha256, actual)
    return path


if __name__ == "__main__":
    ensure_vendor()
