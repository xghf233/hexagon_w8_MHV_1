"""Read-only audited arrays, finite split views and a resumable training iterator.

Paths are supplied explicitly. The source Mac path in metadata is provenance,
not a runtime dependency. No Wolfram, orbit code, or source WXF is needed.
"""

import hashlib
from copy import copy
import json
import os
from pathlib import Path

import numpy as np
import torch

from .alphabet import ALPHABET_VERSION, LETTERS, WORD_LENGTH
from .tokenizer import (
    AssembledSequence, SEQUENCE_LENGTH, assemble_batch, encoding_metadata,
    next_token_batch, _integer,
)

SPLITS = ("train", "val", "test")
SAMPLER_VERSION = "heptagon-pcg64-epoch-v1"


def _require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


class HeptagonDataset:
    """Finite view of a stored split; indices always refer to original WXF rows.

    Construction verifies file hashes, shapes, alphabet, nonzero integer labels,
    global word uniqueness and the full partition. It NEVER resplits data.
    Optional expected_metadata_sha256 anchors the manifest to a trusted copy.
    __getitem__ returns (six IDs, coefficient), not a tokenized character string.
    """

    def __init__(self, data_dir: str | Path, split: str = "train", *,
                 expected_metadata_sha256: str | None = None):
        _require(split in SPLITS, f"Unknown split: {split!r}")
        self.root = Path(data_dir).expanduser().resolve(strict=True)
        manifest = self.root / "metadata.json"
        self.metadata_sha256 = _sha256(manifest)
        if expected_metadata_sha256 is not None:
            _require(self.metadata_sha256 == expected_metadata_sha256,
                     "Metadata SHA-256 mismatch")
        with manifest.open(encoding="utf-8") as stream:
            meta = json.load(stream)
        _require(meta.get("schema_version") == 1, "Unsupported dataset schema")
        _require(meta.get("validation_status") == "passed", "Dataset audit was not passed")
        _require(meta.get("weight") == 6 and meta.get("loop_order") == 3
                 and meta.get("sector") == "MHV" and meta.get("nonzero_only") is True,
                 "Expected nonzero weight-6 MHV data")
        alphabet = meta["alphabet"]
        _require(alphabet["version"] == ALPHABET_VERSION
                 and alphabet["id_to_letter"] == list(LETTERS), "Alphabet mismatch")
        partition = meta["split"]
        _require(partition["split_type"] == "random_row"
                 and partition["orbit_grouped"] is False, "Expected random-row split")
        _require(partition["file"] == "splits.npz" and partition["dtype"] == "<i4",
                 "Unsupported split storage")
        _require(partition["slice_order"] == list(SPLITS)
                 and partition["generator"] == "numpy.random.Generator(PCG64)",
                 "Unexpected split protocol")
        split_seed = _integer(partition["seed"], "split seed")
        _require(split_seed >= 0, "Negative split seed")
        _require(set(partition["counts"]) == set(SPLITS), "Unexpected split counts")

        # Use fixed local basenames, not arbitrary paths read from the manifest.
        for name in ("words.npy", "coefficients.npy", "splits.npz", "audit.json"):
            info = meta["files"][name]
            path = self.root / name
            _require(path.stat().st_size == info["bytes"], f"File size mismatch: {name}")
            _require(_sha256(path) == info["sha256"], f"File hash mismatch: {name}")
        with (self.root / "audit.json").open(encoding="utf-8") as stream:
            audit = json.load(stream)
        _require(audit.get("status") == "passed", "Audit status is not passed")

        self.words = np.load(self.root / "words.npy", mmap_mode="r", allow_pickle=False)
        self.coefficients = np.load(self.root / "coefficients.npy", mmap_mode="r",
                                    allow_pickle=False)
        n = self.words.shape[0] if self.words.ndim else 0
        _require(n > 0 and self.words.shape == (n, WORD_LENGTH)
                 and self.words.dtype == np.dtype("uint8"), "Invalid words shape/dtype")
        _require(self.coefficients.shape == (n,)
                 and self.coefficients.dtype == np.dtype("<i2"), "Invalid coefficients shape/dtype")
        for key, dtype, shape in (("words", "uint8", [n, WORD_LENGTH]),
                                  ("coefficients", "<i2", [n])):
            _require(meta["arrays"][key] == {
                "file": f"{key}.npy", "dtype": dtype, "shape": shape,
            }, f"Array metadata mismatch: {key}")
        _require(np.all(self.words < len(LETTERS)), "Illegal letter ID")
        _require(np.all((self.coefficients != 0) & (self.coefficients >= -48)
                        & (self.coefficients <= 48)), "Invalid weight-6 coefficient")
        _require(len(np.unique(self.words.view("V6").reshape(-1))) == n,
                 "Duplicate word in dataset")
        _require(audit["n_samples"] == n and audit["unique_words"] == n,
                 "Audit row counts mismatch")

        with np.load(self.root / "splits.npz", allow_pickle=False) as archive:
            _require(set(archive.files) == set(SPLITS), "Invalid split keys")
            indices = {key: archive[key] for key in SPLITS}
        for key, ids in indices.items():
            count = _integer(partition["counts"][key], f"{key} count")
            _require(count > 0 and ids.shape == (count,) and ids.dtype == np.dtype("<i4"),
                     f"Invalid split indices: {key}")
        combined = np.concatenate(list(indices.values()))
        _require(np.array_equal(np.sort(combined), np.arange(n)),
                 "Splits overlap, omit rows or contain out-of-range indices")

        proposal = meta.get("training_encoding_proposal", {})
        for key, expected in encoding_metadata().items():
            if key in proposal:
                _require(proposal[key] == expected, f"Encoding metadata mismatch: {key}")
        self.metadata = meta
        self.split = split
        self.indices = indices[split]
        self.indices.flags.writeable = False

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        index = _integer(index, "sample index")
        if not 0 <= index < len(self):
            raise IndexError(index)
        row = int(self.indices[index])
        return self.words[row].copy(), int(self.coefficients[row])

    def preencode(self, sequence_len: int = SEQUENCE_LENGTH) -> AssembledSequence:
        """Encode this split once on CPU; no GPU allocation or disk writes."""
        return assemble_batch(self.words[self.indices], self.coefficients[self.indices],
                              sequence_len)

    def identity(self) -> dict:
        """Path-independent identity suitable for checkpoint metadata."""
        return {"metadata_sha256": self.metadata_sha256,
                "dataset_id": self.metadata["dataset_id"], "split": self.split,
                "split_sha256": self.metadata["files"]["splits.npz"]["sha256"],
                "selected_rows_sha256": hashlib.sha256(self.indices.astype("<i4").tobytes()).hexdigest(),
                "n_samples": len(self)}

    def training_subset(self, count: int, seed: int = 42):
        """In-memory train-only diagnostic subset; never rewrites saved splits."""
        count = _integer(count, "subset count")
        _require(self.split == "train" and 0 < count <= len(self), "Invalid training subset")
        positions = np.random.Generator(np.random.PCG64(seed)).choice(len(self), count, replace=False)
        subset = copy(self)
        subset.indices = self.indices[positions].copy()
        subset.indices.flags.writeable = False
        return subset


class HeptagonDataLoader:
    """Single-process, single-device infinite TRAIN iterator for NanoInfra.

    CPU tokens are preencoded once. Full batches may straddle epochs: no tail
    rows are dropped. Epoch e uses PCG64(SeedSequence([seed,e])) to shuffle only
    the stored train members. Resume stores rows consumed and regenerates only
    the current epoch's permutation, without replaying earlier batches.

    This is not a distributed sampler; multi-rank use is rejected.
    """

    def __init__(self, dataset: HeptagonDataset, batch_size: int,
                 sequence_len: int = SEQUENCE_LENGTH, seed: int = 42,
                 shuffle: bool = True, device: str | torch.device = "cpu"):
        _require(dataset.split == "train", "Infinite loader is for train only")
        if (int(os.environ.get("WORLD_SIZE", "1")) != 1 or
                (torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1)):
            raise ValueError("HeptagonDataLoader does not support distributed training")
        self.batch_size = _integer(batch_size, "batch_size")
        self.seed = _integer(seed, "training shuffle seed")
        _require(self.batch_size > 0 and self.seed >= 0, "Invalid batch size or seed")
        _require(isinstance(shuffle, bool), "shuffle must be bool")
        self.dataset = dataset
        self.sequence_len = _integer(sequence_len, "sequence_len")
        self.shuffle = shuffle
        self.device = torch.device(device)
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
