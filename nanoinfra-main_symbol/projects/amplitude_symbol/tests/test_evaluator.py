"""Tests for strict_decode_coefficient and metrics computation.

Pure CPU tests — no model, no GPU, no generation.
"""

import os
import sys

import pytest

_NANOINFRA_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _NANOINFRA_ROOT not in sys.path:
    sys.path.insert(0, _NANOINFRA_ROOT)

from projects.amplitude_symbol.evaluator import (
    SampleMetrics,
    OrbitMetrics,
    compute_sample_metrics,
    compute_orbit_metrics,
    strict_decode_coefficient,
)
from projects.amplitude_symbol.tokenizer import (
    BOS_ID,
    COEFF_ID,
    EOS_ID,
    MINUS_ID,
    PAD_ID,
    PLUS_ID,
    num_id,
)


# ======================================================================
# strict_decode_coefficient
# ======================================================================
class TestStrictDecode:
    """Tests for the strict coefficient decoder."""

    # --- Valid cases ---
    def test_positive_single_block(self):
        # [PLUS, NUM_5, EOS]
        assert strict_decode_coefficient([PLUS_ID, num_id(5), EOS_ID]) == 5

    def test_negative_single_block(self):
        # [MINUS, NUM_7, EOS]
        assert strict_decode_coefficient([MINUS_ID, num_id(7), EOS_ID]) == -7

    def test_positive_multi_block(self):
        # 12334 = 12*1000 + 334
        assert strict_decode_coefficient(
            [PLUS_ID, num_id(12), num_id(334), EOS_ID]
        ) == 12334

    def test_negative_multi_block(self):
        assert strict_decode_coefficient(
            [MINUS_ID, num_id(12), num_id(334), EOS_ID]
        ) == -12334

    def test_zero(self):
        # 0 → [PLUS, NUM_0, EOS]
        assert strict_decode_coefficient([PLUS_ID, num_id(0), EOS_ID]) == 0

    def test_single_block_max(self):
        # 999 → [PLUS, NUM_999, EOS]
        assert strict_decode_coefficient(
            [PLUS_ID, num_id(999), EOS_ID]
        ) == 999

    def test_eos_padding_after_real_eos(self):
        # After first real EOS, padding EOS tokens should be ignored
        assert strict_decode_coefficient(
            [PLUS_ID, num_id(3), EOS_ID, EOS_ID, EOS_ID, EOS_ID]
        ) == 3

    # --- Invalid: no EOS ---
    def test_invalid_no_eos(self):
        assert strict_decode_coefficient([PLUS_ID, num_id(5)]) is None

    # --- Invalid: bad sign ---
    def test_invalid_bad_sign_bos(self):
        assert strict_decode_coefficient([BOS_ID, num_id(5), EOS_ID]) is None

    def test_invalid_bad_sign_num(self):
        assert strict_decode_coefficient([num_id(5), num_id(5), EOS_ID]) is None

    # --- Invalid: no NUM after sign ---
    def test_invalid_no_num_after_sign(self):
        assert strict_decode_coefficient([PLUS_ID, EOS_ID]) is None

    def test_invalid_empty_before_eos(self):
        assert strict_decode_coefficient([EOS_ID]) is None

    # --- Invalid: illegal tokens before EOS ---
    def test_invalid_word_token(self):
        # word token 'a' = 0 before EOS
        assert strict_decode_coefficient([PLUS_ID, 0, EOS_ID]) is None

    def test_invalid_bos_before_eos(self):
        assert strict_decode_coefficient([PLUS_ID, BOS_ID, num_id(5), EOS_ID]) is None

    def test_invalid_coeff_before_eos(self):
        assert strict_decode_coefficient([PLUS_ID, COEFF_ID, num_id(5), EOS_ID]) is None

    def test_invalid_pad_before_eos(self):
        assert strict_decode_coefficient([PLUS_ID, PAD_ID, EOS_ID]) is None

    # --- Invalid: leading zero ---
    def test_invalid_leading_zero(self):
        # Two blocks, first is NUM_0 → invalid
        assert strict_decode_coefficient(
            [PLUS_ID, num_id(0), num_id(5), EOS_ID]
        ) is None

    def test_leading_zero_three_blocks(self):
        assert strict_decode_coefficient(
            [PLUS_ID, num_id(0), num_id(5), num_id(10), EOS_ID]
        ) is None

    def test_valid_zero_single_block(self):
        # Single block NUM_0 is fine (coefficient = 0)
        assert strict_decode_coefficient([PLUS_ID, num_id(0), EOS_ID]) == 0

    # --- Invalid: negative zero ---
    def test_invalid_negative_zero(self):
        assert strict_decode_coefficient([MINUS_ID, num_id(0), EOS_ID]) is None

    # --- Invalid: sign token in block position ---
    def test_invalid_sign_in_block_position(self):
        # [PLUS, PLUS, NUM_5, EOS] — second PLUS where a NUM should be
        assert strict_decode_coefficient(
            [PLUS_ID, PLUS_ID, num_id(5), EOS_ID]
        ) is None


# ======================================================================
# compute_sample_metrics
# ======================================================================
class TestSampleMetrics:
    def test_all_correct(self):
        preds = [42, -7, 0, 12334]
        truths = [42, -7, 0, 12334]
        m = compute_sample_metrics(preds, truths)
        assert m.n_samples == 4
        assert m.n_valid == 4
        assert m.n_invalid == 0
        assert m.n_exact_correct == 4
        assert m.n_magnitude_correct == 4
        assert m.n_sign_correct == 4
        assert m.exact_accuracy == 1.0
        assert m.invalid_rate == 0.0

    def test_all_invalid(self):
        preds = [None, None, None]
        truths = [5, 10, -3]
        m = compute_sample_metrics(preds, truths)
        assert m.n_samples == 3
        assert m.n_valid == 0
        assert m.n_invalid == 3
        assert m.n_exact_correct == 0
        assert m.exact_accuracy == 0.0
        assert m.invalid_rate == 1.0

    def test_mixed(self):
        preds = [42, None, -7, 10]     # 2nd invalid, 4th wrong (should be 99)
        truths = [42, 5, -7, 99]
        m = compute_sample_metrics(preds, truths)
        assert m.n_samples == 4
        assert m.n_valid == 3
        assert m.n_invalid == 1
        assert m.n_exact_correct == 2   # 42 and -7 correct
        assert m.n_magnitude_correct == 2  # |10| ≠ |99| so only 42, -7
        # sign: 42✓  None✗  -7✓  10(positive) vs 99(positive)✓ → 3
        assert m.n_sign_correct == 3
        assert m.invalid_rate == 0.25

    def test_sign_accuracy_invalid_counts_as_wrong(self):
        preds = [None, None]
        truths = [1, -1]
        m = compute_sample_metrics(preds, truths)
        assert m.sign_accuracy == 0.0

    def test_magnitude_accuracy(self):
        preds = [5, -5, 0, None]
        truths = [5, 5, 0, 10]
        m = compute_sample_metrics(preds, truths)
        # |5|=|5| ✓ | -5|=|5| ✓ |0|=|0| ✓ None ✗
        assert m.n_magnitude_correct == 3
        assert m.magnitude_accuracy == 0.75

    def test_length_mismatch_raises(self):
        with pytest.raises(AssertionError):
            compute_sample_metrics([1], [1, 2])


# ======================================================================
# compute_orbit_metrics
# ======================================================================
class TestOrbitMetrics:
    # Helper: build samples where each word gets a unique 1-char "orbit"
    # (avoids needing real dihedral images).  We use get_orbit_id from
    # symmetry.py which works on any-length strings.

    def test_all_orbits_consistent_and_correct(self):
        # Two different words that happen to be in DIFFERENT orbits
        # 'aaa' and 'aba' — check they don't share an orbit
        from projects.amplitude_symbol.symmetry import get_orbit_id
        w1, w2 = "aaaaaaabbb", "bbbbbbccaa"
        oid1, oid2 = get_orbit_id(w1), get_orbit_id(w2)
        # These are full 10-letter words; they may or may not be same orbit.
        # Use our mock approach: manually set words that we KNOW are in different orbits.
        # Actually, let's just test with two singletons:
        words = ["abababccdd", "abababdcab"]  # random 10-letter words
        oid_a = get_orbit_id(words[0])
        oid_b = get_orbit_id(words[1])
        # If by chance they're the same orbit, skip
        if oid_a == oid_b:
            pytest.skip("Test words accidentally share an orbit")
        preds = [10, 20]
        truths = [10, 20]
        m = compute_orbit_metrics(preds, truths, words)
        # Each orbit has 1 member → both are singletons
        assert m.n_singleton_orbits == 2
        assert m.n_eligible_orbits == 0

    def test_orbit_consistency_success_whole_orbit_failure(self):
        # One orbit of size 2: both predict same WRONG value → consistent but not correct
        words = ["aaaaaa", "aaaaba"]    # check if they share an orbit...
        # Actually, let's use words we KNOW are in the same orbit.
        # These real 10-letter words are dihedral images of each other:
        # We can't guarantee without testing.  Let's use a different approach.
        # We'll use a mock approach: compute orbit_id ourselves.
        from projects.amplitude_symbol.symmetry import get_orbit_id, get_dihedral_images

        # Pick a word and one of its dihedral images
        base = "ababababab"
        images = get_dihedral_images(base)
        w1, w2 = images[0], images[1]

        words = [w1, w2]
        preds = [99, 99]           # same wrong value
        truths = [10, 10]          # both should be 10
        m = compute_orbit_metrics(preds, truths, words)
        assert m.n_eligible_orbits == 1
        assert m.n_consistent_orbits == 1   # both predict 99
        assert m.n_whole_correct_orbits == 0  # 99 ≠ 10
        assert m.orbit_consistency == 1.0
        assert m.whole_orbit_accuracy == 0.0

    def test_orbit_consistency_failure(self):
        from projects.amplitude_symbol.symmetry import get_dihedral_images

        base = "ababababab"
        images = get_dihedral_images(base)
        w1, w2 = images[0], images[1]

        words = [w1, w2]
        preds = [10, 20]           # different values in same orbit
        truths = [10, 10]
        m = compute_orbit_metrics(preds, truths, words)
        assert m.n_eligible_orbits == 1
        assert m.n_consistent_orbits == 0
        assert m.n_whole_correct_orbits == 0
        assert m.orbit_consistency == 0.0

    def test_orbit_with_invalid_output(self):
        from projects.amplitude_symbol.symmetry import get_dihedral_images

        base = "ababababab"
        images = get_dihedral_images(base)
        w1, w2 = images[0], images[1]

        words = [w1, w2]
        preds = [10, None]         # one invalid
        truths = [10, 10]
        m = compute_orbit_metrics(preds, truths, words)
        assert m.n_eligible_orbits == 1
        assert m.n_consistent_orbits == 0  # not consistent due to None
        assert m.n_whole_correct_orbits == 0

    def test_whole_orbit_correct(self):
        from projects.amplitude_symbol.symmetry import get_dihedral_images

        base = "ababababab"
        images = get_dihedral_images(base)
        w1, w2 = images[0], images[1]

        words = [w1, w2]
        preds = [10, 10]
        truths = [10, 10]
        m = compute_orbit_metrics(preds, truths, words)
        assert m.n_eligible_orbits == 1
        assert m.n_consistent_orbits == 1
        assert m.n_whole_correct_orbits == 1
        assert m.orbit_consistency == 1.0
        assert m.whole_orbit_accuracy == 1.0

    def test_singleton_handling(self):
        # Single word with no dihedral sisters in the eval set → singleton orbit
        from projects.amplitude_symbol.symmetry import get_orbit_id
        word = "abababccdd"
        words = [word]
        preds = [42]
        truths = [42]
        m = compute_orbit_metrics(preds, truths, words)
        # Only 1 word total, so it's the only member of its orbit in eval
        assert m.n_singleton_orbits == 1
        assert m.n_eligible_orbits == 0
        assert m.n_orbits == 1

    def test_multiple_orbits_mixed(self):
        from projects.amplitude_symbol.symmetry import get_dihedral_images

        base1 = "ababababab"
        base2 = "aabbccddee"
        img1 = get_dihedral_images(base1)
        img2 = get_dihedral_images(base2)

        words = [img1[0], img1[1], img2[0], img2[1]]
        preds = [10, 10, None, 20]     # orbit1: consistent & correct; orbit2: invalid + wrong
        truths = [10, 10, 30, 30]
        m = compute_orbit_metrics(preds, truths, words)
        assert m.n_eligible_orbits == 2
        assert m.n_consistent_orbits == 1  # only orbit 1 consistent
        assert m.n_whole_correct_orbits == 1  # only orbit 1 all correct
        assert m.orbit_consistency == 0.5
        assert m.whole_orbit_accuracy == 0.5

    def test_truth_conflict_raises(self):
        from projects.amplitude_symbol.symmetry import get_dihedral_images

        base = "ababababab"
        images = get_dihedral_images(base)
        w1, w2 = images[0], images[1]

        words = [w1, w2]
        truths = [10, 20]  # same orbit but different true coefficients!
        preds = [10, 20]
        with pytest.raises(ValueError, match="conflicting truth"):
            compute_orbit_metrics(preds, truths, words)

    def test_empty_samples(self):
        m = compute_orbit_metrics([], [], [])
        assert m.n_orbits == 0
        assert m.n_eligible_orbits == 0


# ======================================================================
# SampleMetrics dataclass
# ======================================================================
class TestSampleMetricsDataclass:
    def test_defaults(self):
        m = SampleMetrics()
        assert m.n_samples == 0
        assert m.exact_accuracy == 0.0
        assert m.invalid_rate == 0.0

    def test_to_dict(self):
        m = SampleMetrics()
        m.n_samples = 10
        m.n_exact_correct = 8
        m.n_invalid = 1
        d = m.to_dict()
        assert d["n_samples"] == 10
        assert d["exact_accuracy"] == 0.8
        assert d["invalid_output_rate"] == 0.1


# ======================================================================
# OrbitMetrics dataclass
# ======================================================================
class TestOrbitMetricsDataclass:
    def test_defaults(self):
        m = OrbitMetrics()
        assert m.n_orbits == 0
        assert m.orbit_consistency == 0.0
        assert m.whole_orbit_accuracy == 0.0

    def test_to_dict(self):
        m = OrbitMetrics()
        m.n_orbits = 100
        m.n_eligible_orbits = 90
        m.n_consistent_orbits = 72
        m.n_whole_correct_orbits = 45
        m.n_singleton_orbits = 10
        d = m.to_dict()
        assert d["n_orbits"] == 100
        assert d["orbit_consistency"] == 0.8
        assert d["whole_orbit_accuracy"] == 0.5
