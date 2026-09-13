"""Immutable source identity and small random-row array contract."""

import hashlib
from pathlib import Path

DATASET_ID = "hexagon-mhv-w8-0y-x32-random-row-v1"
SOURCE_SHA256 = "363bc50186af77f7a8b2f726964ff305ab2522b576a8a47f2b9811419dd69de2"
SOURCE_MANIFEST_SHA256 = "fdd7c25ec846fd32426055dc26500536c86d4fdbd9c5aff9c5db9bb0c358a00a"
N_SOURCE_ROWS, N_ROWS, SPLIT_SEED = 243000, 11208, 42
SPLIT_COUNTS = {"train": 8966, "val": 1120, "test": 1122}
SPLIT_VERSION = "hexagon-random-row-pcg64-seed42-v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def outside_repository(path):
    path = Path(path).expanduser().resolve()
    repository = Path(__file__).resolve().parents[3]
    if path == repository or repository in path.parents:
        raise ValueError("Data and run artifacts must be outside the code repository")
    return path
