"""Reproducible train/val/test splits with orbit-grouped and random-row strategies.

The primary split is **orbit-grouped**: all dihedral images of a word stay
in the same split, guaranteeing zero orbit overlap between train/val/test.

A random-row split is also generated as a control (purely diagnostic).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .symmetry import get_orbit_id

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_DIR = Path(__file__).resolve().parent
_MANIFEST_PATH = _PROJECT_DIR / "MANIFEST.json"


# ---------------------------------------------------------------------------
# Split generation
# ---------------------------------------------------------------------------
def generate_splits(
    symb: dict[str, int],
    seed: int = 42,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    data_path: str | None = None,
    aiamplitudes_commit: str | None = None,
) -> dict[str, Any]:
    """Generate both orbit-grouped and random-row splits.

    Returns a manifest dict to be saved as MANIFEST.json.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9, (
        f"Split fractions must sum to 1.0, got {train_frac}+{val_frac}+{test_frac}"
    )

    rng = random.Random(seed)
    words = list(symb.keys())

    # ------------------------------------------------------------------
    # 1. Orbit-grouped split
    # ------------------------------------------------------------------
    orbit_ids = {w: get_orbit_id(w) for w in words}
    orbit_to_words: dict[str, list[str]] = defaultdict(list)
    for w in words:
        orbit_to_words[orbit_ids[w]].append(w)

    orbit_list = sorted(orbit_to_words.keys())
    rng.shuffle(orbit_list)

    n_orbits = len(orbit_list)
    n_test_orbits = max(1, round(n_orbits * test_frac))
    n_val_orbits = max(1, round(n_orbits * val_frac))
    n_train_orbits = n_orbits - n_test_orbits - n_val_orbits

    test_orbits = set(orbit_list[:n_test_orbits])
    val_orbits = set(orbit_list[n_test_orbits:n_test_orbits + n_val_orbits])
    train_orbits = set(orbit_list[n_test_orbits + n_val_orbits:])

    # --- hard assertions ---
    assert train_orbits.isdisjoint(val_orbits), \
        "ORBIT LEAK: train ∩ val ≠ ∅"
    assert train_orbits.isdisjoint(test_orbits), \
        "ORBIT LEAK: train ∩ test ≠ ∅"
    assert val_orbits.isdisjoint(test_orbits), \
        "ORBIT LEAK: val ∩ test ≠ ∅"

    def _words_in(orbit_set: set[str]) -> list[str]:
        result: list[str] = []
        for oid in sorted(orbit_set):
            result.extend(sorted(orbit_to_words[oid]))
        return result

    orbit_train_words = _words_in(train_orbits)
    orbit_val_words = _words_in(val_orbits)
    orbit_test_words = _words_in(test_orbits)

    # ------------------------------------------------------------------
    # 2. Random-row split (control)
    # ------------------------------------------------------------------
    shuffled_words = list(words)
    rng.shuffle(shuffled_words)

    n_total = len(shuffled_words)
    n_rr_test = max(1, round(n_total * test_frac))
    n_rr_val = max(1, round(n_total * val_frac))

    rr_test_words = sorted(shuffled_words[:n_rr_test])
    rr_val_words = sorted(shuffled_words[n_rr_test:n_rr_test + n_rr_val])
    rr_train_words = sorted(shuffled_words[n_rr_test + n_rr_val:])

    # ------------------------------------------------------------------
    # 3. Assemble manifest
    # ------------------------------------------------------------------
    manifest: dict[str, Any] = {
        "seed": seed,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "total_samples": len(words),
        "total_orbits": n_orbits,
        "generated_at": _timestamp(),
        # Store a portable logical identifier, never a machine-specific path.
        "data_file": Path(data_path).name if data_path else "EZ_symb_new_norm",
        "data_md5": _file_md5(data_path) if data_path else None,
        "aiamplitudes_commit": aiamplitudes_commit,
        "orbit_grouped": {
            "train": {
                "n_samples": len(orbit_train_words),
                "n_orbits": len(train_orbits),
                "orbit_ids": sorted(train_orbits),
                "words": orbit_train_words,
            },
            "val": {
                "n_samples": len(orbit_val_words),
                "n_orbits": len(val_orbits),
                "orbit_ids": sorted(val_orbits),
                "words": orbit_val_words,
            },
            "test": {
                "n_samples": len(orbit_test_words),
                "n_orbits": len(test_orbits),
                "orbit_ids": sorted(test_orbits),
                "words": orbit_test_words,
            },
            "assertions": {
                "train_val_disjoint": True,
                "train_test_disjoint": True,
                "val_test_disjoint": True,
            },
        },
        "random_row": {
            "train": {"n_samples": len(rr_train_words), "words": rr_train_words},
            "val": {"n_samples": len(rr_val_words), "words": rr_val_words},
            "test": {"n_samples": len(rr_test_words), "words": rr_test_words},
        },
    }

    return manifest


def load_manifest(path: str | Path | None = None) -> dict[str, Any]:
    """Load the split manifest from disk."""
    if path is None:
        path = _MANIFEST_PATH
    with open(path, "r") as f:
        return json.load(f)


def get_split_words(
    manifest: dict[str, Any],
    split: str = "train",
    strategy: str = "orbit_grouped",
) -> list[str]:
    """Get the word list for a given split and strategy from a manifest."""
    return manifest[strategy][split]["words"]


def get_split_coeffs(
    manifest: dict[str, Any],
    symb: dict[str, int],
    split: str = "train",
    strategy: str = "orbit_grouped",
) -> list[tuple[str, int]]:
    """Get (word, coefficient) pairs for a given split."""
    words = get_split_words(manifest, split, strategy)
    return [(w, symb[w]) for w in words]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _timestamp() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _file_md5(path: str | None) -> str | None:
    if path is None or not os.path.isfile(path):
        return None
    return hashlib.md5(Path(path).read_bytes()).hexdigest()
