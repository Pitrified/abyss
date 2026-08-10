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

    def __str__(self) -> str:
        """Return the string representation of the object."""
        s = "AbyssPaths:\n"
        s += f"      src_fol: {self.src_fol}\n"
        s += f"     root_fol: {self.root_fol}\n"
        s += f"    cache_fol: {self.cache_fol}\n"
        s += f"     data_fol: {self.data_fol}\n"
        s += f"     pose_fol: {self.pose_fol}\n"
        return s.rstrip()
