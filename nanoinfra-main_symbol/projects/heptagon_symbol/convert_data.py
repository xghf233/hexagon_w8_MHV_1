"""Convert the pinned weight-6 WXF to audited arrays and random-row splits.

Requires only NumPy and the standard library. This is deliberately a restricted
reader for one verified data release, not a general Wolfram evaluator or WXF
implementation. Nothing in the input is executed. Existing outputs are refused.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import signal
import struct
import sys
import tempfile
import time
import zlib

import numpy as np

from .alphabet import ALPHABET_VERSION, LETTERS, WORD_LENGTH, WXF_LETTER_TO_ID

VERSION = "heptagon-w6-wxf-v1"
SOURCE_SHA256 = "8cae0bbcfa99b28bf5b20f102966db4125f26567b3f7fd975850cf037f9b70ba"
SOURCE_BYTES = 1_823_955
DECODED_BYTES = 42_915_610
ROWS = 467_250
MAX_INPUT = 4 * 1024**2
MAX_DECODED = 64 * 1024**2
COEFFICIENT_VALUES = (
    -48, -24, -16, -14, -13, -12, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 24, 48,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Reader:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.position = 0

    def take(self, count: int) -> bytes:
        end = self.position + count
        require(count >= 0 and end <= len(self.payload), "Truncated WXF payload")
        data = self.payload[self.position:end]
        self.position = end
        return data

    def varint(self) -> int:
        value = 0
        for shift in range(0, 63, 7):
            byte = self.take(1)[0]
            value |= (byte & 127) << shift
            if byte < 128:
                return value
        raise ValueError("WXF varint exceeds supported length")

    def symbol(self) -> str:
        require(self.take(1) == b"s", "Expected a WXF symbol")
        size = self.varint()
        require(size <= 32, "Unexpectedly long symbol")
        return self.take(size).decode("utf-8")

    def function(self) -> tuple[str, int]:
        require(self.take(1) == b"f", "Expected a WXF function")
        count = self.varint()
        return self.symbol(), count

    def integer(self) -> int:
        # This pinned release encodes every coefficient as signed Integer8.
        require(self.take(1) == b"C", "Expected signed Integer8 coefficient")
        return struct.unpack("<b", self.take(1))[0]


def parse_source(source: Path):
    with source.open("rb") as stream:
        raw = stream.read(MAX_INPUT + 1)
    require(len(raw) <= MAX_INPUT, "Input exceeds 4 MiB limit")
    require(len(raw) == SOURCE_BYTES, "Unexpected source file size")
    require(hashlib.sha256(raw).hexdigest() == SOURCE_SHA256, "Source hash mismatch")
    require(raw[:3] == b"8C:", "Expected compressed WXF header 8C:")
    decoder = zlib.decompressobj()
    payload = decoder.decompress(raw[3:], MAX_DECODED + 1)
    require(len(payload) <= MAX_DECODED, "Decompressed data exceeds 64 MiB limit")
    require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,
            "Incomplete compressed stream or trailing bytes")
    require(len(payload) == DECODED_BYTES, "Unexpected decompressed size")

    reader = Reader(payload)
    require(reader.function() == ("Plus", ROWS), "Unexpected top-level expression")
    words = np.empty((ROWS, WORD_LENGTH), dtype=np.uint8)
    coefficients = np.empty(ROWS, dtype="<i2")
    counts = Counter()
    letter_counts = Counter()
    forms = Counter()
    example_indices = sorted({0, 1, 2, ROWS - 1, *map(int,
        np.random.Generator(np.random.PCG64(20260906)).choice(ROWS, 12, replace=False))})
    examples = []
    for row in range(ROWS):
        head, arity = reader.function()
        if head == "Times":
            require(arity == 2, f"Unexpected Times arity at row {row}")
            coefficient = reader.integer()
            head, arity = reader.function()
            forms["explicit_times"] += 1
        else:
            coefficient = 1
            forms["implicit_positive_one"] += 1
        require(head == "Global`SB" and arity == WORD_LENGTH,
                f"Expected six-letter SB at row {row}")
        require(coefficient != 0, f"Zero coefficient at row {row}")
        ids = []
        for _ in range(WORD_LENGTH):
            name = reader.symbol()
            require(name in WXF_LETTER_TO_ID, f"Unknown letter {name!r} at row {row}")
            ids.append(WXF_LETTER_TO_ID[name])
        words[row] = ids
        coefficients[row] = coefficient
        counts[coefficient] += 1
        letter_counts.update(ids)
        if row in example_indices:
            examples.append({"row": row, "letters": [LETTERS[i] for i in ids],
                             "word_ids": ids, "coefficient": coefficient,
                             "tokens16": [42, *ids, 43, 46 if coefficient > 0 else 47,
                                          48 + abs(coefficient), 44, *([45] * 5)]})
    require(reader.position == len(payload), "Trailing expression content")
    require(tuple(sorted(counts)) == COEFFICIENT_VALUES, "Unexpected coefficient values")
    require(forms == {"explicit_times": 290850, "implicit_positive_one": 176400},
            "Unexpected explicit/implicit term counts")
    require(len(letter_counts) == 42, "Expected all 42 alphabet letters")
    return words, coefficients, {
        "decompressed_sha256": hashlib.sha256(payload).hexdigest(),
        "decompressed_bytes": len(payload), "all_bytes_consumed": True,
        "term_forms": dict(forms), "coefficient_histogram": dict(sorted(counts.items())),
        "letter_histogram": {LETTERS[i]: letter_counts[i] for i in range(42)},
        "examples": examples,
    }


def varint_bytes(value: int) -> bytes:
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def symbol_bytes(name: str) -> bytes:
    data = name.encode("utf-8")
    return b"s" + varint_bytes(len(data)) + data


def reconstructed_payload_hash(words, coefficients) -> str:
    """Independently serialize every array row back to the source byte format."""
    digest = hashlib.sha256()
    digest.update(b"f" + varint_bytes(len(words)) + symbol_bytes("Plus"))
    sb_header = b"f\x06" + symbol_bytes("Global`SB")
    times_header = b"f\x02" + symbol_bytes("Times")
    alphabet_bytes = [symbol_bytes(f"Global`{letter}") for letter in LETTERS]
    for ids, value in zip(words, coefficients, strict=True):
        coefficient = int(value)
        if coefficient != 1:
            digest.update(times_header + b"C" + struct.pack("<b", coefficient))
        digest.update(sb_header)
        digest.update(b"".join(alphabet_bytes[int(i)] for i in ids))
    return digest.hexdigest()


def histogram(values) -> dict[str, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(unique, counts, strict=True)}


def write_json(path: Path, data: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def convert(source: Path, output: Path, seed: int) -> dict:
    start = time.perf_counter()
    source = source.expanduser().resolve(strict=True)
    # Keep final component unresolved so even a dangling destination symlink is refused.
    output = Path(os.path.abspath(output.expanduser()))
    if os.path.lexists(output):
        raise FileExistsError(f"Output already exists; choose a new version: {output}")
    print("Parsing pinned weight-6 WXF (no Wolfram evaluation)...", flush=True)
    words, coefficients, audit = parse_source(source)
    require(words.shape == (ROWS, 6) and words.dtype == np.dtype("uint8"), "Invalid words")
    require(coefficients.shape == (ROWS,) and coefficients.dtype == np.dtype("<i2"),
            "Invalid coefficients")
    unique_words = len(np.unique(words.view("V6").reshape(-1)))
    require(unique_words == ROWS, "Duplicate word detected; refusing silent aggregation")
    require(histogram(coefficients) == {str(k): v for k, v in audit["coefficient_histogram"].items()},
            "Array histogram differs from source parse")
    array_letter_counts = np.bincount(words.reshape(-1), minlength=42)
    require(all(int(array_letter_counts[i]) == audit["letter_histogram"][LETTERS[i]]
                for i in range(42)), "Array letter counts differ from source parse")

    print("Creating PCG64 random-row splits and writing staging arrays...", flush=True)
    permutation = np.random.Generator(np.random.PCG64(seed)).permutation(ROWS).astype("<i4")
    splits = {"train": permutation[:373800], "val": permutation[373800:420525],
              "test": permutation[420525:]}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-staging-", dir=output.parent))
    print(f"Staging directory: {staging}", flush=True)
    np.save(staging / "words.npy", words, allow_pickle=False)
    np.save(staging / "coefficients.npy", coefficients, allow_pickle=False)
    np.savez(staging / "splits.npz", **splits)

    print("Reading back all arrays and reconstructing the complete WXF payload...", flush=True)
    loaded_words = np.load(staging / "words.npy", allow_pickle=False)
    loaded_coefficients = np.load(staging / "coefficients.npy", allow_pickle=False)
    require(loaded_words.dtype == words.dtype and np.array_equal(loaded_words, words),
            "Saved words differ from parsed words")
    require(loaded_coefficients.dtype == coefficients.dtype and
            np.array_equal(loaded_coefficients, coefficients), "Saved coefficients differ")
    with np.load(staging / "splits.npz", allow_pickle=False) as archive:
        require(set(archive.files) == set(splits), "Unexpected split keys")
        loaded_splits = {key: archive[key] for key in splits}
    for key, ids in loaded_splits.items():
        require(ids.dtype == np.dtype("<i4") and np.array_equal(ids, splits[key]),
                f"Saved split differs: {key}")
    combined = np.concatenate(list(loaded_splits.values()))
    require(np.array_equal(np.sort(combined), np.arange(ROWS)),
            "Splits are overlapping, incomplete, or out of range")
    roundtrip_hash = reconstructed_payload_hash(loaded_words, loaded_coefficients)
    require(roundtrip_hash == audit["decompressed_sha256"],
            "Full source-byte reconstruction failed")
    require(sha256_file(source) == SOURCE_SHA256, "Source changed during conversion")
    for example in audit["examples"]:
        ids = loaded_words[example["row"]].tolist()
        require([LETTERS[i] for i in ids] == example["letters"], "Letter round-trip failed")
        tokens = example["tokens16"]
        decoded = (1 if tokens[8] == 46 else -1) * (tokens[9] - 48)
        require(decoded == int(loaded_coefficients[example["row"]]), "Example token mismatch")

    split_summary = {}
    train_values = set(map(int, np.unique(coefficients[splits["train"]])))
    for key, ids in splits.items():
        values = coefficients[ids]
        split_summary[key] = {"n_samples": len(ids), "coefficient_histogram": histogram(values),
            "coefficient_values_absent_from_train": sorted(set(map(int, np.unique(values))) - train_values)}
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_bytes = int(usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024)
    audit.update({"status": "passed", "unique_words": unique_words,
        "n_samples": ROWS, "word_length": 6, "n_letters": 42,
        "coefficient_min": int(coefficients.min()), "coefficient_max": int(coefficients.max()),
        "n_coefficient_values": len(COEFFICIENT_VALUES), "n_zero": int((coefficients == 0).sum()),
        "arrays_readback_equal": True, "splits_disjoint_and_complete": True,
        "full_payload_reconstruction_equal": True, "reconstructed_payload_sha256": roundtrip_hash,
        "split_summary": split_summary,
        "resources": {"elapsed_seconds_before_metadata_write": time.perf_counter() - start,
                      "peak_rss_bytes_process_lifetime": peak_bytes}})
    write_json(staging / "audit.json", audit)
    files = {name: {"sha256": sha256_file(staging / name), "bytes": (staging / name).stat().st_size}
             for name in ("words.npy", "coefficients.npy", "splits.npz", "audit.json")}
    metadata = {
        "schema_version": 1, "dataset_id": "heptagon_mhv_w6_phy_v1",
        "weight": 6, "loop_order": 3, "sector": "MHV", "nonzero_only": True,
        "coefficient_normalization": "Unmodified source coefficients; physical convention not independently established",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(source), "format": "WXF 8C: zlib", "bytes": SOURCE_BYTES,
                   "sha256": SOURCE_SHA256, "root": "Plus", "symbol_head": "Global`SB"},
        "alphabet": {"version": ALPHABET_VERSION, "id_to_letter": list(LETTERS),
                     "mapping": "id(a_ij) = 7*(i-1)+(j-1)"},
        "arrays": {"words": {"file": "words.npy", "dtype": "uint8", "shape": [ROWS, 6]},
                   "coefficients": {"file": "coefficients.npy", "dtype": "<i2", "shape": [ROWS]}},
        "row_order": "Original top-level WXF Plus argument order",
        "split": {"file": "splits.npz", "dtype": "<i4", "split_type": "random_row",
                  "seed": seed, "generator": "numpy.random.Generator(PCG64)",
                  "counts": {key: len(ids) for key, ids in splits.items()},
                  "slice_order": ["train", "val", "test"],
                  "orbit_grouped": False, "symmetry_related_rows_may_cross_splits": True},
        "training_encoding_proposal": {"vocab_size": 1048, "sequence_len": 16,
            "BOS": 42, "COEFF": 43, "EOS": 44, "PAD": 45, "PLUS": 46,
            "MINUS": 47, "NUM_OFFSET": 48, "integer_base": 1000,
            "note": "Stored arrays contain only letter IDs and integer coefficients, not training sequences"},
        "converter": {"version": VERSION,
                      "source_files_sha256": {name: sha256_file(Path(__file__).with_name(name))
                                              for name in ("convert_data.py", "alphabet.py")},
                      "python": platform.python_version(), "python_executable": sys.executable,
                      "numpy": np.__version__, "zlib": zlib.ZLIB_RUNTIME_VERSION,
                      "platform": platform.platform(), "argv": sys.argv},
        "files": files, "validation_status": "passed",
    }
    write_json(staging / "metadata.json", metadata)
    for name, expected in (("metadata.json", metadata), ("audit.json", audit)):
        with (staging / name).open(encoding="utf-8") as stream:
            require(json.load(stream) == json.loads(json.dumps(expected)),
                    f"JSON readback differs: {name}")
    if os.path.lexists(output):
        raise FileExistsError(f"Output appeared during conversion; validated staging preserved: {staging}")
    staging.rename(output)
    print(json.dumps({"output": str(output), "n_samples": ROWS,
                      "split_counts": metadata["split"]["counts"], "audit": "passed",
                      "elapsed_seconds": time.perf_counter() - start,
                      "peak_rss_mib": peak_bytes / 1024**2}, ensure_ascii=False, indent=2), flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args()
    require(args.timeout_seconds > 0, "Timeout must be positive")
    if hasattr(signal, "SIGALRM"):
        def timed_out(signum, frame):
            raise TimeoutError("Conversion time limit exceeded; source is unchanged")
        signal.signal(signal.SIGALRM, timed_out)
        signal.alarm(args.timeout_seconds)
    try:
        convert(args.source, args.output, args.seed)
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)


if __name__ == "__main__":
    main()
