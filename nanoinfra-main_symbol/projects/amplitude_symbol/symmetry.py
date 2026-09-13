"""Thin wrappers around AIAmplitudes official D3 dihedral utilities.

Never redefines the D3 action — always delegates to the upstream
implementation to guarantee correctness.
"""

try:
    from aiamplitudes_common_public.rels_utils import (
        alphabet,
        get_dihedral_images,
    )
except ModuleNotFoundError as exc:
    if exc.name == "aiamplitudes_common_public":
        raise ModuleNotFoundError(
            "The Symbol project requires the external "
            "'aiamplitudes_common_public' package. Add the official "
            "AIAmplitudes_common_public checkout to the remote environment's "
            "Python path."
        ) from exc
    raise

__all__ = ["alphabet", "get_dihedral_images", "get_orbit_id", "compute_orbit"]


def get_orbit_id(word: str) -> str:
    """Return the canonical orbit identifier for a word.

    The orbit id is the lexicographically smallest dihedral image,
    which is unique per D3 orbit.
    """
    return min(get_dihedral_images(word))


def compute_orbit(word: str) -> list[str]:
    """Return all six dihedral images of a word (the full D3 orbit)."""
    return get_dihedral_images(word)
