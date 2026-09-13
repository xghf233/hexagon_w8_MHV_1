"""Tests for data loading, orbit computation, and assertions.

Run with:
    python -m pytest projects/amplitude_symbol/tests/test_data.py -v
"""

import os
import sys

import pytest

# Ensure the nanoinfra root is on sys.path for imports
_NANOINFRA_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _NANOINFRA_ROOT not in sys.path:
    sys.path.insert(0, _NANOINFRA_ROOT)

from projects.amplitude_symbol.symmetry import (
    alphabet,
    compute_orbit,
    get_dihedral_images,
    get_orbit_id,
)
from projects.amplitude_symbol.prepare_data import (
    EXPECTED_MD5,
    EXPECTED_SAMPLES,
    EXPECTED_SIZE,
    EXPECTED_WORD_LENGTH,
    audit_symbol,
    load_symbol,
)


# ---------------------------------------------------------------------------
# Symmetry tests (no data file needed)
# ---------------------------------------------------------------------------
class TestSymmetry:
    def test_alphabet_size(self):
        assert len(alphabet) == 6
        assert set(alphabet) == {"a", "b", "c", "d", "e", "f"}

    def test_dihedral_images_count(self):
        """Every word must have exactly 6 dihedral images."""
        images = get_dihedral_images("ababababab")
        assert len(images) == 6

    def test_dihedral_images_unique(self):
        """All 6 dihedral images of a generic word should be distinct."""
        images = get_dihedral_images("aaabbbccdd")
        assert len(set(images)) == 6, f"Expected 6 unique images, got {images}"

    def test_orbit_id_is_min(self):
        """orbit_id must be the lexicographically smallest image."""
        oid = get_orbit_id("ddddccccbb")
        images = get_dihedral_images("ddddccccbb")
        assert oid == min(images)

    def test_same_orbit_same_id(self):
        """Two words in the same orbit must have the same orbit_id."""
        word1 = "ababababab"
        images = get_dihedral_images(word1)
        word2 = images[2]  # pick a different image
        assert get_orbit_id(word1) == get_orbit_id(word2)

    def test_different_word_lengths(self):
        """Should work for any word length."""
        for w in ["a", "ab", "abc", "abcd", "abcde", "abcdef"]:
            images = compute_orbit(w)
            assert len(images) == 6, f"Failed for word '{w}'"

    def test_all_words_valid(self):
        """All dihedral images should only contain valid alphabet letters."""
        for w in ["ababababab", "aaabbbccdd", "fedcbaaaaa"]:
            for img in compute_orbit(w):
                assert all(ch in alphabet for ch in img)


# ---------------------------------------------------------------------------
# Data loading and audit tests (needs the symbol file)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def symb():
    """Load the symbol once for all data tests."""
    return load_symbol()


@pytest.fixture(scope="module")
def report(symb):
    """Run the audit once for all data tests."""
    return audit_symbol(symb)


class TestDataLoading:
    def test_sample_count(self, symb):
        assert len(symb) == EXPECTED_SAMPLES, (
            f"Expected {EXPECTED_SAMPLES}, got {len(symb)}"
        )

    def test_all_words_length_10(self, symb):
        bad = [w for w in symb if len(w) != EXPECTED_WORD_LENGTH]
        assert not bad, f"Found {len(bad)} words with length != {EXPECTED_WORD_LENGTH}: {bad[:10]}"

    def test_all_letters_valid(self, symb):
        valid = set(alphabet)
        bad = sorted(set(
            ch for w in symb for ch in w if ch not in valid
        ))
        assert not bad, f"Invalid letters: {bad}"

    def test_no_zero_coefficients(self, symb):
        zeros = [w for w, c in symb.items() if c == 0]
        assert not zeros, f"Found {len(zeros)} zero-coefficient words"

    def test_coefficient_range(self, symb):
        """Coefficients should be reasonable integers."""
        coeffs = list(symb.values())
        assert all(isinstance(c, int) for c in coeffs), "Non-integer coefficient found"
        assert min(coeffs) < 0, "No negative coefficients"
        assert max(coeffs) > 0, "No positive coefficients"


class TestOrbitAudit:
    def test_orbit_count(self, report):
        assert report["n_orbits"] > 0
        # With 263,880 samples and max orbit size 6, expect at least ~43,980 orbits
        assert report["n_orbits"] >= 40000, (
            f"Too few orbits: {report['n_orbits']}"
        )

    def test_orbit_size_range(self, report):
        sizes = set(report["orbit_size_distribution"].keys())
        # Each orbit has 1-6 members (6 dihedral images)
        assert all(1 <= s <= 6 for s in sizes), f"Invalid orbit sizes: {sizes}"

    def test_no_inconsistent_orbits(self, report):
        assert report["n_inconsistent_orbits"] == 0, (
            f"Found {report['n_inconsistent_orbits']} inconsistent orbits!"
        )

    def test_orbit_consistency_flag(self, report):
        assert report["orbits_consistent"] is True


class TestFileChecks:
    def test_file_md5(self, report):
        if report.get("file_md5"):
            assert report["file_md5"] == EXPECTED_MD5, (
                f"MD5 mismatch: expected {EXPECTED_MD5}, got {report['file_md5']}"
            )

    def test_file_size(self, report):
        if report.get("file_size"):
            assert report["file_size"] == EXPECTED_SIZE, (
                f"Size mismatch: expected {EXPECTED_SIZE}, got {report['file_size']}"
            )

    def test_aiamplitudes_commit_present(self, report):
        commit = report.get("aiamplitudes_commit", "")
        assert commit and not commit.startswith("ERROR"), (
            f"AIAmplitudes commit not found: {commit}"
        )
