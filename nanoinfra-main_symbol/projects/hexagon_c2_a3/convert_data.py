"""CUDA-gated host preprocessing and atomic, write-once M2 dataset publication."""

import argparse
import json
from pathlib import Path
import tempfile
import numpy as np

from .alphabet import ALPHABET_VERSION, LETTERS, TRAINING_ALIASES
from .build_samples import build_arrays, verify_against_source
from .data_contract import (ARRAY_SCHEMA, DATASET_ID, N_ROWS, PHYSICAL_OBJECT, REPRESENTATION,
                            SOURCE_SHA256, SOURCE_MANIFEST_SHA256, SPLIT_COUNTS, SPLIT_SEED,
                            SPLIT_VERSION, dependency_hashes, outside_repository, separate_path,
                            require, sha256, write_json)
from .oracle import ExactOracle
from .runtime import require_cuda_server
from .splits import random_splits, split_diagnostics, validate_splits
from .tokenizer import encoding_metadata


def prepare_data(source, output):
    require_cuda_server()  # before source reading, data preparation or artifact writes
    output = outside_repository(output)
    source = Path(source).expanduser().resolve(strict=True)
    separate_path(output, source.parent)
    require(not output.exists(), "Choose a NEW dataset directory; never overwrite a release")
    oracle = ExactOracle(source)
    arrays, summary = build_arrays(oracle)
    splits = random_splits()
    validate_splits(splits)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    print(f"Preparing dataset in {staging}; retained for diagnosis on failure", flush=True)
    for name, array in arrays.items():
        with (staging / f"{name}.npy").open("xb") as stream:
            np.save(stream, array, allow_pickle=False)
    with (staging / "splits.npz").open("xb") as stream:
        np.savez(stream, **splits)
    readback = {}
    for name, expected in arrays.items():
        actual = np.load(staging / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        require(actual.dtype.str == expected.dtype.str and np.array_equal(actual, expected),
                f"Serialized array mismatch: {name}")
        readback[name] = actual
    with np.load(staging / "splits.npz", allow_pickle=False) as archive:
        validate_splits({key: archive[key] for key in archive.files})
    # Re-read the fixed source, then independently replace every target's two y's.
    verify_against_source(readback, ExactOracle(source))
    audit = {
        "status": "passed", "scope": "host_preprocessing_inside_GPU_workflow", **summary,
        "source_rows": 243000, "independent_condition_lookups": N_ROWS * 36,
        "all_arrays_read_back": True, "fresh_source_pass": True, "no_extra_scaling": True,
        "split_type": "random_row", "input_grouped": False, "family_grouped": False, "orbit_grouped": False,
        "split_diagnostics": split_diagnostics(arrays, splits),
        "exploratory_subset_selected_after_full_pool_audit": True,
        "model_tested": False, "full_integrability_checked": False,
    }
    write_json(staging / "audit.json", audit)
    names = [f"{name}.npy" for name in ARRAY_SCHEMA] + ["splits.npz", "audit.json"]
    metadata = {
        "schema_version": 1, "dataset_id": DATASET_ID, "rows": N_ROWS, "validation_status": "passed",
        "loop_order": 4, "weight": 8, "sector": "MHV", "y_count": 2, "adjacent_only": True,
        "nonzero_only": True, "physical_object": PHYSICAL_OBJECT, "coefficient_scale": 32,
        "coefficient_relation": "C4=32*c4; source already scaled", "representation": REPRESENTATION,
        "source": {**oracle.provenance, "sha256": SOURCE_SHA256,
                   "scaling_manifest_sha256": SOURCE_MANIFEST_SHA256},
        "alphabet": {"version": ALPHABET_VERSION, "id_to_letter": list(LETTERS),
                     "training_aliases": list(TRAINING_ALIASES)},
        "arrays": {name: {"file": f"{name}.npy", "dtype": dtype, "shape": list(shape)}
                   for name, (dtype, shape) in ARRAY_SCHEMA.items()},
        "split": {"version": SPLIT_VERSION, "split_type": "random_row", "counts": SPLIT_COUNTS,
                  "input_grouped": False, "family_grouped": False, "orbit_grouped": False,
                  "seed": SPLIT_SEED, "generator": "numpy.random.Generator(PCG64)",
                  "file": "splits.npz", "dtype": "<i4", "slice_order": list(SPLIT_COUNTS),
                  "within_split_order": "ascending source-filtered row ID"},
        "files": {name: {"bytes": (staging / name).stat().st_size, "sha256": sha256(staging / name)}
                  for name in names},
        "encoding": encoding_metadata(), "numpy_version": np.__version__,
        "converter_dependency_sha256": dependency_hashes(),
    }
    write_json(staging / "metadata.json", metadata)
    digest = sha256(staging / "metadata.json")
    # Exercise the consumer's full release contract before making the release visible.
    from .dataset import HexagonDataset
    HexagonDataset(staging, expected_metadata_sha256=digest)
    require(not output.exists(), "Dataset destination appeared during publication")
    staging.rename(output)
    return {"data_dir": str(output), "metadata_sha256": digest,
            "rows": N_ROWS, "unique_inputs": summary["unique_inputs"], "split_counts": SPLIT_COUNTS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_data(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
