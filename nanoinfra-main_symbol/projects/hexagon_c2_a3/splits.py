"""Frozen random-row partition; exact-input/family groups are diagnostics ONLY."""

from collections import Counter
import numpy as np

from .data_contract import N_ROWS, SPLIT_COUNTS, SPLIT_SEED, require


def random_splits():
    order = np.random.Generator(np.random.PCG64(SPLIT_SEED)).permutation(N_ROWS)
    a, b = SPLIT_COUNTS["train"], SPLIT_COUNTS["train"] + SPLIT_COUNTS["val"]
    return {key: np.sort(values).astype("<i4") for key, values in
            zip(SPLIT_COUNTS, (order[:a], order[a:b], order[b:]), strict=True)}


def validate_splits(splits):
    require(set(splits) == set(SPLIT_COUNTS), "Wrong split keys")
    expected = random_splits()
    for name, count in SPLIT_COUNTS.items():
        ids = splits[name]
        require(ids.dtype.str == "<i4" and ids.shape == (count,)
                and np.array_equal(ids, expected[name]), f"Split differs from locked random-row seed42: {name}")
    require(np.array_equal(np.sort(np.concatenate(list(splits.values()))), np.arange(N_ROWS)),
            "Partition overlaps or omits rows")


def histogram(values):
    return {str(k): int(v) for k, v in sorted(Counter(map(int, values)).items())}


def split_diagnostics(arrays, splits):
    train = splits["train"]
    train_inputs = set(map(int, arrays["input_group_ids"][train]))
    sibling_counts = Counter(map(int, arrays["family_ids"][train]))
    result = {}
    for name, ids in splits.items():
        target = arrays["coefficients"][ids].astype(np.int64)
        cond = arrays["conditions"][ids].astype(np.int64)
        y = arrays["y_types"][ids]
        groups, families = arrays["input_group_ids"][ids], arrays["family_ids"][ids]
        seen = sum(int(g) in train_inputs for g in groups)
        siblings = [sibling_counts[int(f)] - int(name == "train") for f in families]
        result[name] = {
            "rows": len(ids), "unique_inputs": len(set(map(int, groups))),
            "unique_targets": len(set(map(int, target))), "families": len(set(map(int, families))),
            "unique_C36": len(set(map(tuple, cond))),
            "seen_input38_in_train_rows": seen, "seen_input38_in_train_fraction": seen / len(ids),
            "seen_family_in_train_rows": sum(int(f) in sibling_counts for f in families),
            "train_sibling_count_histogram": histogram(siblings), "train_siblings_exclude_self": True,
            "coefficients": histogram(target), "magnitudes": histogram(np.abs(target)),
            "signs": histogram(np.sign(target)), "target_high": histogram(np.abs(target) // 100),
            "target_low": histogram(np.abs(target) % 100),
            "condition_signs": histogram(np.sign(cond).ravel()),
            "condition_high": histogram((np.abs(cond) // 100).ravel()),
            "condition_low": histogram((np.abs(cond) % 100).ravel()),
            "r_one_based": histogram(arrays["y_positions"][ids, 0] + 1),
            "ordered_y_types": dict(sorted(Counter(f"{a},{b}" for a, b in y).items())),
        }
    return result
