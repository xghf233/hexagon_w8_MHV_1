"""Versioned, immutable M2/38 data contract and write-once artifact helpers."""

import hashlib
import json
from pathlib import Path

DATASET_ID = "hexagon-mhv-w8-2y-adjacent-nonzero-m2-38-x32-random-row-v1"
SOURCE_SHA256 = "363bc50186af77f7a8b2f726964ff305ab2522b576a8a47f2b9811419dd69de2"
SOURCE_MANIFEST_SHA256 = "fdd7c25ec846fd32426055dc26500536c86d4fdbd9c5aff9c5db9bb0c358a00a"
PHYSICAL_OBJECT = "ordinary BDS-like four-loop Hexagon MHV full weight-8 symbol"
N_SOURCE_ROWS, N_ROWS, N_INPUTS, SPLIT_SEED = 243000, 40776, 13338, 42
SPLIT_COUNTS = {"train": 32620, "val": 4077, "test": 4079}
SPLIT_VERSION = "hexagon-m2-adjacent38-random-row-pcg64-seed42-v1"
POSITION_COUNTS = {2: 7200, 3: 6684, 4: 8184, 5: 7788, 6: 10920}  # zero-based r
ARRAY_SCHEMA = {
    "conditions": ("<i2", (N_ROWS, 36)),
    "y_positions": ("|u1", (N_ROWS, 2)),
    "y_types": ("|u1", (N_ROWS, 2)),
    "coefficients": ("<i2", (N_ROWS,)),
    "words": ("|u1", (N_ROWS, 8)),
    "source_row_ids": ("<i4", (N_ROWS,)),
    "family_ids": ("<i4", (N_ROWS,)),
    "input_group_ids": ("<i4", (N_ROWS,)),
}
REPRESENTATION = {
    "logical_inputs": 38, "model_fields": ["conditions", "y_types"],
    "condition_order": "slot=6*i+j; ordinary native IDs i,j in 0..5",
    "input_group_key": ["ordered_C36", "ordered_y_types"],
    "positions_in_model": False, "word_in_model": False,
    "family_placeholder_id": 9, "group_id_order": "lexicographic exact tuple",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def separate_path(path, protected):
    path, protected = Path(path).expanduser().resolve(), Path(protected).expanduser().resolve()
    require(path != protected and protected not in path.parents and path not in protected.parents,
            f"Artifact destination overlaps protected path: {protected}")
    return path


def outside_repository(path):
    repository = Path(__file__).resolve().parents[3]
    path = separate_path(path, repository)
    # Protect the local source collection even if --source was omitted on reuse.
    source_collection = repository.parent / "Symbol_Data"
    return separate_path(path, source_collection)


def dependency_hashes():
    """Actual task/core dependency closure, not unrelated old task files/docs."""
    project = Path(__file__).resolve().parent
    edition = project.parents[1]
    paths = list(project.glob("*.py")) + list((project / "configs").glob("*"))
    paths += [p for p in (edition / "core").rglob("*.py") if "tests" not in p.parts]
    paths.append(project.parent / "amplitude_symbol/blocks/attention.py")
    return {str(p.relative_to(edition)): sha256(p) for p in sorted(set(paths))
            if p.is_file() and p.suffix in (".py", ".yaml")}
