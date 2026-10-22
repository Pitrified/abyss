"""Paths and folders for data files."""

from pathlib import Path

import abyss


class AbyssPaths:
    """Paths and folders for data and resources."""

    def __init__(self) -> None:
        """Load the config for data folders."""
        self.load_config()

    def load_config(self) -> None:
        """Load the config for data folders."""
        # src folder of the package
        self.src_fol = Path(abyss.__file__).parent
        # root folder of the project repository
        self.root_fol = self.src_fol.parents[1]
        # cache
        self.cache_fol = self.root_fol / "cache"
        # data
        self.data_fol = self.root_fol / "data"
        # sample videos to track poses in, outside the repository
        self.pose_fol = Path.home() / "data" / "pose"
        # browser UIs, each a folder of static files
        self.ui_fol = self.root_fol / "ui"
        # third-party files the UIs load: the manifest in git, the files outside it
        self.vendor_manifest = self.ui_fol / "vendor.toml"
        self.vendor_fol = self.cache_fol / "vendor"
        # generated, never committed: synthetic clips, screenshots
        self.build_fol = self.root_fol / "build"
        self.clips_fol = self.build_fol / "clips"

    def __str__(self) -> str:
        """Return the string representation of the object."""
        s = "AbyssPaths:\n"
        s += f"      src_fol: {self.src_fol}\n"
        s += f"     root_fol: {self.root_fol}\n"
        s += f"    cache_fol: {self.cache_fol}\n"
        s += f"     data_fol: {self.data_fol}\n"
        s += f"     pose_fol: {self.pose_fol}\n"
        s += f"       ui_fol: {self.ui_fol}\n"
        s += f"   vendor_fol: {self.vendor_fol}\n"
        s += f"    clips_fol: {self.clips_fol}\n"
        return s.rstrip()
