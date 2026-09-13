"""Read-only M2 data with a narrow (conditions, y_types, label) sample interface."""

from copy import copy
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch

from .alphabet import ALPHABET_VERSION, LETTERS, TRAINING_ALIASES
from .build_samples import validate_arrays
from .data_contract import (ARRAY_SCHEMA, DATASET_ID, N_ROWS, PHYSICAL_OBJECT, REPRESENTATION,
                            SOURCE_SHA256, SOURCE_MANIFEST_SHA256, SPLIT_COUNTS, SPLIT_VERSION,
                            SPLIT_SEED, require, sha256)
from .splits import validate_splits
from .tokenizer import AssembledSequence, SEQUENCE_LENGTH, assemble_batch, encoding_metadata, next_token_batch, _integer

SAMPLER_VERSION = "hexagon-m2-adjacent38-pcg64-epoch-v1"
_require = require


class HexagonDataset:
    """Finite split view. Provenance arrays cannot be passed through __getitem__."""

    def __init__(self, data_dir, split="train", *, expected_metadata_sha256=None):
        require(split in SPLIT_COUNTS, "Unknown split")
        self.root = Path(data_dir).expanduser().resolve(strict=True)
        self.metadata_sha256 = sha256(self.root / "metadata.json")
        require(isinstance(expected_metadata_sha256, str) and len(expected_metadata_sha256) == 64
                and all(c in "0123456789abcdef" for c in expected_metadata_sha256),
                "Trusted lowercase metadata SHA-256 is mandatory")
        require(self.metadata_sha256 == expected_metadata_sha256, "Metadata SHA-256 mismatch")
        meta = json.loads((self.root / "metadata.json").read_text(encoding="utf-8"))
        expected = {
            "schema_version": 1, "dataset_id": DATASET_ID, "rows": N_ROWS,
            "validation_status": "passed", "loop_order": 4, "weight": 8,
            "sector": "MHV", "y_count": 2, "adjacent_only": True, "nonzero_only": True,
            "physical_object": PHYSICAL_OBJECT, "coefficient_scale": 32,
            "representation": REPRESENTATION, "encoding": encoding_metadata(),
            "alphabet": {"version": ALPHABET_VERSION, "id_to_letter": list(LETTERS),
                         "training_aliases": list(TRAINING_ALIASES)},
        }
        for key, value in expected.items():
            require(meta.get(key) == value, f"M2 dataset contract mismatch: {key}")
        require(meta["source"]["sha256"] == SOURCE_SHA256
                and meta["source"]["scaling_manifest_sha256"] == SOURCE_MANIFEST_SHA256,
                "Wrong source provenance")
        partition = meta["split"]
        for key, value in {
            "version": SPLIT_VERSION, "split_type": "random_row", "counts": SPLIT_COUNTS,
            "input_grouped": False, "family_grouped": False, "orbit_grouped": False,
            "seed": SPLIT_SEED, "generator": "numpy.random.Generator(PCG64)",
            "file": "splits.npz", "dtype": "<i4", "slice_order": list(SPLIT_COUNTS),
            "within_split_order": "ascending source-filtered row ID",
        }.items():
            require(partition.get(key) == value, f"Split contract mismatch: {key}")
        names = [f"{name}.npy" for name in ARRAY_SCHEMA] + ["splits.npz", "audit.json"]
        require(set(meta["files"]) == set(names) and set(meta["arrays"]) == set(ARRAY_SCHEMA),
                "Wrong array/file inventory")
        for name in names:
            info, path = meta["files"][name], self.root / name
            require(path.stat().st_size == info["bytes"] and sha256(path) == info["sha256"],
                    f"Dataset file changed: {name}")
        arrays = {}
        for name, (dtype, shape) in ARRAY_SCHEMA.items():
            require(meta["arrays"][name] == {"file": f"{name}.npy", "dtype": dtype, "shape": list(shape)},
                    f"Array schema mismatch: {name}")
            arrays[name] = np.load(self.root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            setattr(self, name, arrays[name])
        summary = validate_arrays(arrays)
        audit = json.loads((self.root / "audit.json").read_text(encoding="utf-8"))
        require(audit["status"] == "passed" and all(audit.get(k) == summary[k] for k in (
            "n_samples", "unique_words", "unique_inputs", "conflicting_inputs")), "Audit summary mismatch")
        with np.load(self.root / "splits.npz", allow_pickle=False) as archive:
            indices = {key: archive[key] for key in archive.files}
        validate_splits(indices)
        for ids in indices.values():
            ids.flags.writeable = False
        self.metadata, self.split_indices = meta, indices
        self.split, self.indices = split, indices[split]
        self.subset_protocol = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        index = _integer(index, "sample index")
        if not 0 <= index < len(self):
            raise IndexError(index)
        row = self.indices[index]
        return self.conditions[row].copy(), self.y_types[row].copy(), int(self.coefficients[row])

    def preencode(self, sequence_len=SEQUENCE_LENGTH):
        return assemble_batch(self.conditions[self.indices], self.y_types[self.indices],
                              self.coefficients[self.indices], sequence_len)

    def identity(self):
        return {"metadata_sha256": self.metadata_sha256, "dataset_id": DATASET_ID,
                "split": self.split, "split_sha256": self.metadata["files"]["splits.npz"]["sha256"],
                "selected_rows_sha256": hashlib.sha256(self.indices.astype("<i4").tobytes()).hexdigest(),
                "n_samples": len(self), "subset_protocol": self.subset_protocol}

    def training_subset(self, count, seed=42):
        """Train-only tiny fixture: distinct inputs; deterministic metadata coverage."""
        count = _integer(count, "subset count")
        require(self.split == "train" and 0 < count <= len(set(self.input_group_ids[self.indices])),
                "Invalid distinct-input tiny subset")
        order = np.random.Generator(np.random.PCG64(seed)).permutation(self.indices)
        unique = {}
        for row in order:
            unique.setdefault(int(self.input_group_ids[row]), int(row))
        candidates = list(unique.values())

        def features(row):
            return {("r", int(self.y_positions[row, 0])),
                    ("y", *map(int, self.y_types[row])),
                    ("sign", int(self.coefficients[row] > 0))}

        selected, covered = [], set()
        while len(selected) < count:
            # max() keeps the first candidate in the seeded order on a tie.
            row = max(candidates, key=lambda r: len(features(r) - covered))
            selected.append(row)
            covered |= features(row)
            candidates.remove(row)
        subset = copy(self)
        subset.indices = np.array(sorted(selected), dtype="<i4")
        subset.indices.flags.writeable = False
        subset.subset_protocol = {"version": "distinct-input-greedy-metadata-coverage-v1", "seed": seed,
                                  "count": count, "covered": [list(x) for x in sorted(covered)]}
        return subset


class HexagonDataLoader:
    """Single-process, single-device infinite TRAIN iterator for NanoInfra.

    CPU tokens are preencoded once. Full batches may straddle epochs: no tail
    rows are dropped. Epoch e uses PCG64(SeedSequence([seed,e])) to shuffle only
    the stored train members. Resume stores rows consumed and regenerates only
    the current epoch's permutation, without replaying earlier batches.

    This is not a distributed sampler; multi-rank use is rejected.
    """

    def __init__(self, dataset: HexagonDataset, batch_size: int,
                 sequence_len: int = SEQUENCE_LENGTH, seed: int = 42,
                 shuffle: bool = True, device: str | torch.device = "cuda"):
        _require(dataset.split == "train", "Infinite loader is for train only")
        if (int(os.environ.get("WORLD_SIZE", "1")) != 1 or
                (torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1)):
            raise ValueError("HexagonDataLoader does not support distributed training")
        self.batch_size = _integer(batch_size, "batch_size")
        self.seed = _integer(seed, "training shuffle seed")
        _require(self.batch_size > 0 and self.seed >= 0, "Invalid batch size or seed")
        _require(isinstance(shuffle, bool), "shuffle must be bool")
        self.dataset = dataset
        self.sequence_len = _integer(sequence_len, "sequence_len")
        self.shuffle = shuffle
        self.device = torch.device(device)
        _require(self.device.type == "cuda", "No CPU model-batch execution is allowed")
        self.encoded = dataset.preencode(self.sequence_len)
        self._rows = 0
        self._cached_epoch = None
        self._order = None

    def _epoch_order(self, epoch: int) -> np.ndarray:
        if self._cached_epoch != epoch:
            self._order = (np.random.Generator(np.random.PCG64(
                np.random.SeedSequence([self.seed, epoch]))).permutation(len(self.dataset))
                if self.shuffle else np.arange(len(self.dataset)))
            self._cached_epoch = epoch
        return self._order

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        parts = []
        cursor = self._rows
        remaining = self.batch_size
        while remaining:
            epoch, position = divmod(cursor, len(self.dataset))
            count = min(remaining, len(self.dataset) - position)
            parts.append(self._epoch_order(epoch)[position:position + count])
            cursor += count
            remaining -= count
        ids = torch.from_numpy(np.concatenate(parts).astype(np.int64, copy=False))
        sequence = AssembledSequence(
            self.encoded.tokens[ids], self.encoded.token_types[ids],
            self.encoded.loss_weights[ids],
        )
        batch = {key: value.to(self.device) for key, value in next_token_batch(sequence).items()}
        # State describes the NEXT unread row, and is committed only after batching.
        self._rows = cursor
        batch["state_dict"] = self.state_dict()
        return batch

    def _contract(self) -> dict:
        return {"sampler_version": SAMPLER_VERSION, "numpy_version": np.__version__,
                "dataset": self.dataset.identity(), "seed": self.seed,
                "shuffle": self.shuffle, "batch_size": self.batch_size,
                "encoding": encoding_metadata(self.sequence_len)}

    def state_dict(self) -> dict:
        return {"contract": self._contract(), "rows_consumed": self._rows}

    def load_state_dict(self, state: dict) -> None:
        _require(state.get("contract") == self._contract(),
                 "Loader resume contract mismatch (data/split/encoding/seed/batch/environment)")
        rows = _integer(state["rows_consumed"], "rows_consumed")
        _require(rows >= 0 and rows % self.batch_size == 0, "Invalid batch-boundary position")
        self._rows = rows
        self._cached_epoch = None
        self._order = None

    # NanoInfra's checkpoint manager uses set_state; keep standard aliases too.
    get_state = state_dict
    set_state = load_state_dict
