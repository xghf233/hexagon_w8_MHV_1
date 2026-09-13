"""Tests for split reproducibility, assertions, and dataset pipeline."""

import json
import os
import sys
import tempfile

import pytest
import torch

# Ensure nanoinfra root is importable
_NANOINFRA_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _NANOINFRA_ROOT not in sys.path:
    sys.path.insert(0, _NANOINFRA_ROOT)

from projects.amplitude_symbol.symmetry import get_orbit_id
from projects.amplitude_symbol.split import generate_splits
from projects.amplitude_symbol.tokenizer import IGNORE_INDEX, assemble_sequence
from projects.amplitude_symbol.dataset import AmplitudeDataSource, AmplitudeDataLoader


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def symb():
    from projects.amplitude_symbol.prepare_data import load_symbol
    return load_symbol()


@pytest.fixture(scope="module")
def manifest(symb):
    return generate_splits(symb, seed=42)


# ---------------------------------------------------------------------------
# Split tests
# ---------------------------------------------------------------------------
class TestOrbitGroupedSplit:
    def test_splits_sum_to_total(self, manifest, symb):
        og = manifest["orbit_grouped"]
        total = og["train"]["n_samples"] + og["val"]["n_samples"] + og["test"]["n_samples"]
        assert total == len(symb)

    def test_fractions_approximately_correct(self, manifest):
        og = manifest["orbit_grouped"]
        total = og["train"]["n_samples"] + og["val"]["n_samples"] + og["test"]["n_samples"]
        assert abs(og["train"]["n_samples"] / total - 0.80) < 0.02
        assert abs(og["val"]["n_samples"] / total - 0.10) < 0.02
        assert abs(og["test"]["n_samples"] / total - 0.10) < 0.02

    def test_train_val_disjoint(self, manifest):
        train_orbits = set(manifest["orbit_grouped"]["train"]["orbit_ids"])
        val_orbits = set(manifest["orbit_grouped"]["val"]["orbit_ids"])
        assert train_orbits.isdisjoint(val_orbits)

    def test_train_test_disjoint(self, manifest):
        train_orbits = set(manifest["orbit_grouped"]["train"]["orbit_ids"])
        test_orbits = set(manifest["orbit_grouped"]["test"]["orbit_ids"])
        assert train_orbits.isdisjoint(test_orbits)

    def test_val_test_disjoint(self, manifest):
        val_orbits = set(manifest["orbit_grouped"]["val"]["orbit_ids"])
        test_orbits = set(manifest["orbit_grouped"]["test"]["orbit_ids"])
        assert val_orbits.isdisjoint(test_orbits)

    def test_assertions_recorded(self, manifest):
        a = manifest["orbit_grouped"]["assertions"]
        assert a["train_val_disjoint"]
        assert a["train_test_disjoint"]
        assert a["val_test_disjoint"]


class TestSplitReproducibility:
    def test_same_seed_same_split(self, symb):
        m1 = generate_splits(symb, seed=42)
        m2 = generate_splits(symb, seed=42)
        assert m1["orbit_grouped"]["train"]["words"] == m2["orbit_grouped"]["train"]["words"]
        assert m1["orbit_grouped"]["val"]["words"] == m2["orbit_grouped"]["val"]["words"]
        assert m1["orbit_grouped"]["test"]["words"] == m2["orbit_grouped"]["test"]["words"]

    def test_different_seed_different_split(self, symb):
        m1 = generate_splits(symb, seed=42)
        m2 = generate_splits(symb, seed=123)
        assert m1["orbit_grouped"]["train"]["words"] != m2["orbit_grouped"]["train"]["words"]


class TestOrbitIntegrity:
    """Verify all dihedral images of a word stay in the same split."""
    def test_images_in_same_split(self, symb, manifest):
        og = manifest["orbit_grouped"]
        # Build word → split mapping once (check set membership in O(1))
        word_to_split: dict[str, str] = {}
        for split_name in ("train", "val", "test"):
            for w in og[split_name]["words"]:
                word_to_split[w] = split_name

        # For each orbit, all member words must be in the same split
        from collections import defaultdict
        orbit_words: dict[str, list[str]] = defaultdict(list)
        for w in symb:
            orbit_words[get_orbit_id(w)].append(w)

        leaks = []
        for oid, members in orbit_words.items():
            splits_found = {word_to_split.get(m) for m in members}
            if len(splits_found) != 1:
                leaks.append((oid, splits_found, members))

        assert not leaks, (
            f"Found {len(leaks)} orbits with members in multiple splits! "
            f"First: {leaks[:3]}"
        )

    def test_random_row_may_share_orbits(self, symb, manifest):
        """Random-row split likely has orbit overlap — this is expected."""
        rr = manifest["random_row"]
        # At least verify the split itself is consistent
        total = rr["train"]["n_samples"] + rr["val"]["n_samples"] + rr["test"]["n_samples"]
        assert total == len(symb)


class TestManifestStructure:
    def test_has_required_fields(self, manifest):
        for key in ("seed", "total_samples", "total_orbits", "data_md5",
                     "orbit_grouped", "random_row"):
            assert key in manifest, f"Missing key: {key}"

    def test_each_split_has_n_samples(self, manifest):
        for strategy in ("orbit_grouped", "random_row"):
            for split in ("train", "val", "test"):
                assert "n_samples" in manifest[strategy][split]
                assert manifest[strategy][split]["n_samples"] > 0


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------
class TestAmplitudeDataSource:
    @pytest.fixture
    def samples(self, symb, manifest):
        from projects.amplitude_symbol.split import get_split_coeffs
        # Use only a tiny subset for speed
        all_samples = get_split_coeffs(manifest, symb, "train", "orbit_grouped")
        return all_samples[:100]

    def test_yields_dict_with_required_keys(self, samples):
        source = AmplitudeDataSource(samples, sequence_len=32, seed=42, shuffle=False)
        row = next(iter(source))
        assert "tokens" in row
        assert "token_types" in row
        assert "loss_weights" in row

    def test_all_tensors_same_length(self, samples):
        source = AmplitudeDataSource(samples, sequence_len=32, seed=42, shuffle=False)
        row = next(iter(source))
        L = len(row["tokens"])
        assert L == 32
        assert len(row["token_types"]) == L
        assert len(row["loss_weights"]) == L

    def test_token_type_range(self, samples):
        source = AmplitudeDataSource(samples, sequence_len=32, seed=42, shuffle=False)
        row = next(iter(source))
        types = row["token_types"]
        assert set(types.tolist()).issubset({0, 1, 2})

    def test_infinite_iteration(self, samples):
        source = AmplitudeDataSource(samples, sequence_len=32, seed=42)
        rows = []
        for i, row in enumerate(source):
            rows.append(row)
            if i >= 199:
                break
        assert len(rows) == 200  # 2 epochs over 100 samples


class TestAmplitudeDataLoader:
    @pytest.fixture
    def samples(self, symb, manifest):
        from projects.amplitude_symbol.split import get_split_coeffs
        all_samples = get_split_coeffs(manifest, symb, "train", "orbit_grouped")
        return all_samples[:256]

    def test_batch_shape(self, samples):
        source = AmplitudeDataSource(samples, sequence_len=32, seed=42, shuffle=False)
        loader = AmplitudeDataLoader(source, batch_size=8)
        batch = next(iter(loader))
        # idx: [B, L-1] because of shift
        assert batch["idx"].shape == (8, 31)
        assert batch["targets"].shape == (8, 31)
        assert batch["token_types"].shape == (8, 31)

    def test_targets_contain_ignore_index(self, samples):
        source = AmplitudeDataSource(samples, sequence_len=32, seed=42, shuffle=False)
        loader = AmplitudeDataLoader(source, batch_size=8)
        batch = next(iter(loader))
        # Some positions must be IGNORE_INDEX (unsupervised word positions)
        assert (batch["targets"] == IGNORE_INDEX).any(), \
            "Expected some IGNORE_INDEX targets for unsupervised positions"

    def test_targets_also_contain_real_values(self, samples):
        source = AmplitudeDataSource(samples, sequence_len=32, seed=42, shuffle=False)
        loader = AmplitudeDataLoader(source, batch_size=8)
        batch = next(iter(loader))
        # Some positions must be real values (supervised coefficient positions)
        assert (batch["targets"] != IGNORE_INDEX).any(), \
            "Expected some real (non-IGNORE) targets for supervised positions"

    def test_idx_and_types_aligned(self, samples):
        source = AmplitudeDataSource(samples, sequence_len=32, seed=42, shuffle=False)
        loader = AmplitudeDataLoader(source, batch_size=8)
        batch = next(iter(loader))
        assert batch["idx"].shape == batch["token_types"].shape

    def test_all_token_types_valid(self, samples):
        source = AmplitudeDataSource(samples, sequence_len=32, seed=42, shuffle=False)
        loader = AmplitudeDataLoader(source, batch_size=8)
        batch = next(iter(loader))
        types = batch["token_types"]
        assert set(types.flatten().tolist()).issubset({0, 1, 2})

    def test_state_dict_round_trip(self, samples):
        source = AmplitudeDataSource(samples, sequence_len=32, seed=42, shuffle=False)
        loader = AmplitudeDataLoader(source, batch_size=4)
        # Read 2 batches
        batch1 = next(iter(loader))
        batch2 = next(iter(loader))
        state = loader.state_dict()
        assert state["rows"] == 8
        # New loader, restore state
        source2 = AmplitudeDataSource(samples, sequence_len=32, seed=42, shuffle=False)
        loader2 = AmplitudeDataLoader(source2, batch_size=4)
        loader2.load_state_dict(state)
        batch3 = next(iter(loader2))
        # batch3 should equal batch3 from original if we'd kept going
        batch3_direct = next(iter(loader))
        assert torch.equal(batch3["idx"], batch3_direct["idx"])
