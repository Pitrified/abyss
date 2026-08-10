"""Test the abyss params."""

from abyss.params.abyss_params import AbyssParams
from abyss.params.abyss_params import get_abyss_params
from abyss.params.abyss_params import get_abyss_paths


def test_params_is_a_singleton() -> None:
    """Two instantiations return the same object."""
    assert AbyssParams() is AbyssParams()
    assert get_abyss_params() is AbyssParams()


def test_params_exposes_paths() -> None:
    """The params carry the paths."""
    assert get_abyss_paths() is get_abyss_params().paths


def test_str_includes_the_paths() -> None:
    """The string representation embeds the paths block."""
    text = str(get_abyss_params())
    assert "AbyssParams:" in text
    assert "AbyssPaths:" in text
