"""GPU-server entry: necessary host I/O prepares exact random-row training arrays.

This is not a separate CPU model/test pass. The published x32 data is never rescaled.
No existing source, output, or split is overwritten.
"""

import argparse
from collections import Counter
from itertools import permutations
import json
import gzip
from pathlib import Path
import tempfile

import numpy as np

from .alphabet import ALPHABET_VERSION, LETTERS, LETTER_TO_ID, TRAINING_ALIASES
from .data_contract import (DATASET_ID, SOURCE_SHA256, SOURCE_MANIFEST_SHA256,
                            N_SOURCE_ROWS, N_ROWS, SPLIT_SEED, SPLIT_COUNTS, SPLIT_VERSION,
                            sha256, outside_repository)
from .runtime import require_cuda_server


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def read_source(source):
    source = Path(source).expanduser().resolve(strict=True)
    manifest_path = source.parent / "scaling_manifest.json"
    if sha256(source) != SOURCE_SHA256 or sha256(manifest_path) != SOURCE_MANIFEST_SHA256:
        raise ValueError("Untrusted or changed x32 source/manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest["coefficient_scaling"]["factor"] != 32 or manifest["loop"] != 4
            or manifest["weight"] != 8 or manifest["alphabet"] != list(LETTERS)):
        raise ValueError("Expected four-loop weight8 native x32 data")
    words, coefficients, source_ids = [], [], []
    previous = None
    count = 0
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index >= N_SOURCE_ROWS:
                raise ValueError("Unexpected extra source rows")
            row = json.loads(line)
            if set(row) != {"word", "numerator", "denominator"} or row["denominator"] != "1":
                raise ValueError("Expected already-scaled integer rows; do not multiply again")
            value = row["numerator"]
            if not isinstance(value, str) or str(int(value)) != value or int(value) == 0:
                raise ValueError("Expected a canonical nonzero decimal integer string")
            if not isinstance(row["word"], list) or len(row["word"]) != 8:
                raise ValueError("Expected eight atomic letters")
            try:
                word = tuple(LETTER_TO_ID[x] for x in row["word"])
            except (KeyError, TypeError) as exc:
                raise ValueError("Unknown native letter") from exc
            if previous is not None and word <= previous:
                raise ValueError("Source word order is not strictly increasing")
            previous = word
            count += 1
            if max(word) < 6:
                coefficient = int(value)
                if word[0] >= 3 or word[-1] < 3 or not -156 <= coefficient <= 1920:
                    raise ValueError("Unexpected 0y entry or coefficient")
                words.append(word)
                coefficients.append(coefficient)
                source_ids.append(index)
    if count != N_SOURCE_ROWS or len(words) != N_ROWS:
        raise ValueError("Source or 0y count mismatch")
    hist = Counter(coefficients)
    if (len(hist) != 59 or min(hist) != -156 or max(hist) != 1920 or hist[1920] != 6
            or sum(c > 0 for c in coefficients) != 5610
            or sum(abs(c) >= 100 for c in coefficients) != 66):
        raise ValueError("0y coefficient distribution mismatch")
    return (np.array(words, dtype=np.uint8), np.array(coefficients, dtype="<i2"),
            np.array(source_ids, dtype="<i4"))


def random_splits():
    order = np.random.Generator(np.random.PCG64(SPLIT_SEED)).permutation(N_ROWS)
    a, b = SPLIT_COUNTS["train"], SPLIT_COUNTS["train"] + SPLIT_COUNTS["val"]
    return {name: np.sort(values).astype("<i4") for name, values in
            zip(("train", "val", "test"), (order[:a], order[a:b], order[b:]), strict=True)}


def orbit_ids(words, coefficients):
    """Audit only; these groups do NOT influence the random-row partition."""
    maps = [tuple(p[i % 3] + 3 * (i // 3) for i in range(6)) for p in permutations(range(3))]
    table = {tuple(map(int, word)): int(c) for word, c in zip(words, coefficients, strict=True)}
    canonical = []
    for word, coefficient in table.items():
        images = [tuple(mapping[x] for x in word) for mapping in maps]
        if any(table.get(image) != coefficient for image in images):
            raise ValueError("Source S3 images disagree")
        canonical.append(min(images))
    keys = {word: i for i, word in enumerate(sorted(set(canonical)))}
    result = np.array([keys[word] for word in canonical], dtype="<i4")
    if len(keys) != 1868 or not np.all(np.bincount(result) == 6):
        raise ValueError("Unexpected S3 orbit sizes")
    return result


def distribution(values):
    values = [int(value) for value in values]
    def histogram(items):
        return {str(k): v for k, v in sorted(Counter(items).items())}
    return {"rows": len(values), "coefficients": histogram(values),
            "high": histogram(abs(c) // 100 for c in values),
            "low": histogram(abs(c) % 100 for c in values),
            "numeric_tokens": histogram(x for c in values for x in divmod(abs(c), 100))}


def prepare_data(source, output):
    require_cuda_server()  # fails on Mac/CPU before reading or writing training artifacts
    output = outside_repository(output)
    source = Path(source).expanduser().resolve(strict=True)
    if output.exists() or output == source.parent or source.parent in output.parents:
        raise ValueError("Select a new output directory outside the source directory")
    words, coefficients, source_ids = read_source(source)
    splits = random_splits()
    orbits = orbit_ids(words, coefficients)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    arrays = {"words": words, "coefficients": coefficients,
              "source_row_ids": source_ids, "orbit_ids": orbits}
    for name, array in arrays.items():
        with (staging / f"{name}.npy").open("xb") as stream:
            np.save(stream, array, allow_pickle=False)
    with (staging / "splits.npz").open("xb") as stream:
        np.savez(stream, **splits)
    for name, array in arrays.items():
        readback = np.load(staging / f"{name}.npy", allow_pickle=False)
        if readback.dtype != array.dtype or not np.array_equal(readback, array):
            raise ValueError(f"Array serialization failed: {name}")
    with np.load(staging / "splits.npz", allow_pickle=False) as archive:
        if set(archive.files) != set(splits) or any(not np.array_equal(archive[k], v) for k, v in splits.items()):
            raise ValueError("Split serialization failed")
        if not np.array_equal(np.sort(np.concatenate([archive[k] for k in splits])), np.arange(N_ROWS)):
            raise ValueError("Splits overlap or omit rows")
    # A fresh source pass proves retained values/order, not just output self-consistency.
    for expected, actual in zip((words, coefficients, source_ids), read_source(source), strict=True):
        if not np.array_equal(expected, actual):
            raise ValueError("Source changed or retained arrays do not match")
    split_distributions = {key: distribution(coefficients[ids]) for key, ids in splits.items()}
    train_tokens = set(split_distributions["train"]["numeric_tokens"])
    audit = {"status": "passed", "scope": "host_data_io_checks_inside_gpu_server_workflow",
             "n_samples": N_ROWS, "unique_words": N_ROWS, "source_rows": N_SOURCE_ROWS,
             "all_rows_read_back": True, "no_extra_scaling": True,
             "distribution": distribution(coefficients), "split_distributions": split_distributions,
             "unseen_train_numeric_tokens": {s: sorted(set(d["numeric_tokens"]) - train_tokens, key=int)
                                              for s, d in split_distributions.items()},
             "orbit_count": 1868, "orbit_grouped": False,
             "cross_split_orbits": {s: len(set(orbits[ids].tolist()) & set(orbits[splits["train"]].tolist()))
                                    for s, ids in splits.items() if s != "train"},
             "model_tested": False, "full_integrability_checked": False}
    write_json(staging / "audit.json", audit)
    filenames = [f"{name}.npy" for name in arrays] + ["splits.npz", "audit.json"]
    metadata = {
        "schema_version": 1, "dataset_id": DATASET_ID, "validation_status": "passed",
        "weight": 8, "loop_order": 4, "sector": "MHV", "y_count": 0, "nonzero_only": True,
        "normalization": "ordinary BDS-like full weight8 symbol", "coefficient_scale": 32,
        "coefficient_relation": "C4=32*c4; c4=C4/32", "rows": N_ROWS,
        "source": {"path": str(source), "sha256": SOURCE_SHA256,
                   "scaling_manifest_sha256": SOURCE_MANIFEST_SHA256},
        "alphabet": {"version": ALPHABET_VERSION, "id_to_letter": list(LETTERS),
                     "training_aliases": list(TRAINING_ALIASES)},
        "arrays": {name: {"file": f"{name}.npy", "dtype": array.dtype.str,
                          "shape": list(array.shape)} for name, array in arrays.items()},
        "split": {"version": SPLIT_VERSION, "split_type": "random_row", "orbit_grouped": False,
                  "file": "splits.npz", "dtype": "<i4", "slice_order": list(splits),
                  "generator": "numpy.random.Generator(PCG64)", "seed": SPLIT_SEED,
                  "counts": SPLIT_COUNTS, "within_split_order": "ascending source-filtered row ID"},
        "files": {name: {"bytes": (staging / name).stat().st_size, "sha256": sha256(staging / name)}
                  for name in filenames},
        "converter_sha256": sha256(Path(__file__)), "numpy_version": np.__version__,
    }
    write_json(staging / "metadata.json", metadata)
    if output.exists():
        raise FileExistsError(output)
    staging.rename(output)
    return {"data_dir": str(output), "metadata_sha256": sha256(output / "metadata.json"),
            "rows": N_ROWS, "split_counts": SPLIT_COUNTS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare_data(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
