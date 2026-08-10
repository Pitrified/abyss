"""Abyss project params.

Parameters are the actual values of the config.

The class is a singleton, so it can be accessed from anywhere in the code.
"""

from loguru import logger as lg

from abyss.metaclasses.singleton import Singleton
from abyss.params.abyss_paths import AbyssPaths


class AbyssParams(metaclass=Singleton):
    """Abyss project parameters."""

    def __init__(self) -> None:
        """Load the Abyss params."""
        lg.info("Loading Abyss params")
        self.load_config()

    def load_config(self) -> None:
        """Load the abyss configuration."""
        self.paths = AbyssPaths()

    def __str__(self) -> str:
        """Return the string representation of the object."""
        s = "AbyssParams:"
        s += f"\n{self.paths}"
        return s

    def __repr__(self) -> str:
        """Return the string representation of the object."""
        return str(self)


def get_abyss_params() -> AbyssParams:
    """Get the abyss params."""
    return AbyssParams()


def get_abyss_paths() -> AbyssPaths:
    """Get the abyss paths."""
    return get_abyss_params().paths
