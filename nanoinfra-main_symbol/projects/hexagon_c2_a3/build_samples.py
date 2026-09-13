"""Source-ordered adjacent/nonzero rows; metadata never enters the M2 input key."""

from collections import Counter
import numpy as np

from .alphabet import adjacent_positions
from .data_contract import ARRAY_SCHEMA, N_ROWS, N_INPUTS, N_SOURCE_ROWS, POSITION_COUNTS, require


def input_key(conditions, y_types):
    return tuple(map(int, conditions)), tuple(map(int, y_types))


def family_key(word):
    return tuple(9 if letter >= 6 else int(letter) for letter in word)


def stable_ids(keys):
    mapping = {key: i for i, key in enumerate(sorted(set(keys)))}
    return np.array([mapping[key] for key in keys], dtype="<i4")


def validate_arrays(arrays):
    require(set(arrays) == set(ARRAY_SCHEMA), "Wrong M2 array set")
    for name, (dtype, shape) in ARRAY_SCHEMA.items():
        require(arrays[name].dtype.str == dtype and arrays[name].shape == shape,
                f"Wrong shape/dtype: {name}")
    words, target = arrays["words"], arrays["coefficients"]
    pos, ys, cond = arrays["y_positions"], arrays["y_types"], arrays["conditions"]
    require(np.all(words < 9) and np.all(words[:, 0] < 3) and np.all(words[:, 1] < 6)
            and np.all(words[:, -1] >= 3), "Illegal native word/entry restriction")
    require(np.all((target != 0) & (target >= -240) & (target <= 216)), "Wrong adjacent target range/support")
    expected_pos = [adjacent_positions(word) for word in words]
    require(all(p is not None for p in expected_pos), "Every target must contain exactly two adjacent y letters")
    require(np.array_equal(pos, np.array(expected_pos, dtype=np.uint8)), "Stored positions disagree with words")
    require(np.array_equal(ys, np.take_along_axis(words, pos.astype(np.int64), axis=1)), "Ordered y types mismatch")
    require(dict(Counter(map(int, pos[:, 0]))) == POSITION_COUNTS, "Position counts mismatch")
    word_keys = [bytes(w) for w in words]
    require(all(a < b for a, b in zip(word_keys, word_keys[1:])), "Words must be unique and source sorted")
    source_ids = arrays["source_row_ids"]
    require(np.all(source_ids[1:] > source_ids[:-1]) and 0 <= source_ids[0]
            and source_ids[-1] < N_SOURCE_ROWS, "Invalid source line provenance")
    require(np.all((cond >= -156) & (cond <= 1920)) and np.max(np.abs(cond)) == 1920,
            "Condition values outside audited 0y range")
    require(np.all(np.any(cond != 0, axis=1)), "Unexpected all-zero C36 table")
    keys = [input_key(c, y) for c, y in zip(cond, ys, strict=True)]
    groups = stable_ids(keys)
    families = stable_ids([family_key(w) for w in words])
    require(np.array_equal(groups, arrays["input_group_ids"]), "Input groups must use C36/y types ONLY")
    require(np.array_equal(families, arrays["family_ids"]), "Family group ID mismatch")
    labels = {}
    for key, value in zip(keys, target, strict=True):
        value = int(value)
        require(key not in labels or labels[key] == value, "Conflicting M2 input; never drop or relabel it")
        labels[key] = value
    sizes = Counter(keys)
    require(len(sizes) == N_INPUTS and sum(n == 1 for n in sizes.values()) == 3570
            and max(sizes.values()) == 32, "Input multiplicities differ from audited adjacent pool")
    require(len(set(map(tuple, cond))) == 2067 and len(set(map(int, target))) == 76
            and int(target.min()) == -240 and int(target.max()) == 216, "Adjacent pool statistics mismatch")
    return {"n_samples": N_ROWS, "unique_words": len(word_keys), "unique_inputs": len(sizes),
            "conflicting_inputs": 0, "families": int(families.max()) + 1, "unique_C36": 2067,
            "singleton_inputs": 3570, "repeated_inputs": N_INPUTS - 3570,
            "max_input_multiplicity": 32, "input_size_histogram": dict(sorted(Counter(sizes.values()).items()))}


def build_arrays(oracle):
    rows = []
    cache = {}
    for word, value, line in oracle.targets:
        positions = adjacent_positions(word)
        if positions is None:
            continue
        r, s = positions
        family = family_key(word)
        if family not in cache:
            replacement = list(word)
            table = []
            for i in range(6):
                replacement[r] = i
                for j in range(6):
                    replacement[s] = j
                    table.append(oracle.coefficient(replacement))
            cache[family] = tuple(table)
        rows.append((word, value, line - 1, positions, (word[r], word[s]), cache[family]))
    require(len(rows) == N_ROWS, "Wrong number of adjacent targets")
    arrays = {
        "words": np.array([list(row[0]) for row in rows], dtype=np.uint8),
        "coefficients": np.array([row[1] for row in rows], dtype="<i2"),
        "source_row_ids": np.array([row[2] for row in rows], dtype="<i4"),
        "y_positions": np.array([row[3] for row in rows], dtype=np.uint8),
        "y_types": np.array([row[4] for row in rows], dtype=np.uint8),
        "conditions": np.array([row[5] for row in rows], dtype="<i2"),
        "family_ids": stable_ids([family_key(row[0]) for row in rows]),
        "input_group_ids": stable_ids([input_key(row[5], row[4]) for row in rows]),
    }
    return arrays, validate_arrays(arrays)


def verify_against_source(arrays, oracle):
    """Fresh oracle pass + each original target, deliberately no family cache."""
    row = 0
    for word, coefficient, line in oracle.targets:
        positions = adjacent_positions(word)
        if positions is None:
            continue
        require(row < N_ROWS and bytes(arrays["words"][row]) == word
                and int(arrays["coefficients"][row]) == coefficient
                and int(arrays["source_row_ids"][row]) == line - 1, "Source/retained target mismatch")
        r, s = positions
        for slot in range(36):
            i, j = divmod(slot, 6)
            replaced = bytes(i if k == r else j if k == s else letter for k, letter in enumerate(word))
            require(int(arrays["conditions"][row, slot]) == oracle.coefficient(replaced),
                    f"Independent C36 check failed: row {row}, slot {slot}")
        row += 1
    require(row == N_ROWS, "Fresh source pass omitted adjacent rows")
