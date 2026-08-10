"""Test the abyss paths."""

from pathlib import Path

from abyss.params.abyss_paths import AbyssPaths


def test_src_fol_is_the_package_folder() -> None:
    """The src folder is the folder holding the abyss package."""
    paths = AbyssPaths()
    assert paths.src_fol.name == "abyss"
    assert (paths.src_fol / "__init__.py").exists()


def test_root_fol_holds_the_pyproject() -> None:
    """The root folder is the repository root."""
    paths = AbyssPaths()
    assert (paths.root_fol / "pyproject.toml").exists()


def test_repo_folders_are_under_the_root() -> None:
    """Cache and data folders live inside the repository."""
    paths = AbyssPaths()
    assert paths.cache_fol.parent == paths.root_fol
    assert paths.data_fol.parent == paths.root_fol


def test_pose_fol_is_outside_the_repo() -> None:
    """Sample videos live in the home folder, not in the repository."""
    paths = AbyssPaths()
    assert paths.pose_fol == Path.home() / "data" / "pose"


def test_str_lists_every_path() -> None:
    """The string representation mentions each configured folder."""
    text = str(AbyssPaths())
    for name in ("src_fol", "root_fol", "cache_fol", "data_fol", "pose_fol"):
        assert name in text
