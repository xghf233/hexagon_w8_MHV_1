"""Exact collision audit of 38/40-field inputs on nonzero four-loop 2y targets.

Standard library only. No model, training, split generation, or rescaling.
Local CPU execution for this data audit was explicitly approved by the user.
This exception does not authorize local model checks or CPU training.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
from itertools import groupby
import json
from pathlib import Path
import platform
import sys
import time


SOURCE_SHA256 = "363bc50186af77f7a8b2f726964ff305ab2522b576a8a47f2b9811419dd69de2"
MANIFEST_SHA256 = "fdd7c25ec846fd32426055dc26500536c86d4fdbd9c5aff9c5db9bb0c358a00a"
LETTERS = ("a", "b", "c", "mu", "mv", "mw", "yu", "yv", "yw")
ALIASES = ("hat_a", "hat_b", "hat_c", "hat_d", "hat_e", "hat_f", "y_U", "y_V", "y_W")
LETTER_IDS = {letter: index for index, letter in enumerate(LETTERS)}
EXPECTED_COUNTS = {0: 11208, 2: 110082, 4: 115878, 6: 5832}
REPRESENTATIONS = ("c36_y_types", "c36_y_types_positions")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def histogram(values):
    return {str(key): value for key, value in sorted(Counter(values).items())}


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def read_source(source):
    manifest_path = source.parent / "scaling_manifest.json"
    require(sha256(source) == SOURCE_SHA256, "Source SHA-256 mismatch")
    require(sha256(manifest_path) == MANIFEST_SHA256, "Manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest["loop"] == 4 and manifest["weight"] == 8, "Wrong loop or weight")
    require(manifest["alphabet"] == list(LETTERS), "Wrong native alphabet")
    scaling = manifest["coefficient_scaling"]
    require(scaling["factor"] == 32 and scaling["integer_label_scaling_applied"] is True,
            "Expected already-scaled C4 integers")
    require(scaling["apply_to_conditioning_and_target"] is True, "Scaling contract mismatch")
    zero_y, targets = {}, []
    counts, coefficient_counts = Counter(), Counter()
    previous = None
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            require(set(row) == {"word", "numerator", "denominator"}, "Wrong row fields")
            require(row["denominator"] == "1", "Noninteger denominator")
            raw = row["numerator"]
            require(isinstance(raw, str), "Numerator must be a decimal string")
            value = int(raw)
            require(str(value) == raw and value != 0, "Noncanonical or zero source coefficient")
            require(isinstance(row["word"], list) and len(row["word"]) == 8, "Wrong word length")
            word = bytes(LETTER_IDS[letter] for letter in row["word"])
            require(previous is None or word > previous, "Duplicate or unsorted source word")
            previous = word
            require(word[0] < 3 and word[1] < 6 and word[-1] >= 3,
                    "Source violates declared first/two-first/final-entry restrictions")
            y_count = sum(letter >= 6 for letter in word)
            require(y_count in EXPECTED_COUNTS, "Unexpected y count")
            require(-240 <= value <= 1920, "Unexpected integer range")
            counts[y_count] += 1
            coefficient_counts[value] += 1
            if y_count == 0:
                zero_y[word] = value
            elif y_count == 2:
                targets.append((word, value, line_number))
    require(dict(counts) == EXPECTED_COUNTS, "Source y-sector counts mismatch")
    require(sum(counts.values()) == manifest["rows"] == 243000, "Source row count mismatch")
    require(dict(coefficient_counts) == {int(k): v for k, v in manifest["coefficient_histogram"].items()},
            "Full source coefficient histogram mismatch")
    return zero_y, targets, {
        "path": str(source), "sha256": SOURCE_SHA256,
        "manifest_path": str(manifest_path), "manifest_sha256": MANIFEST_SHA256,
        "source_rows": sum(counts.values()), "nonzero_rows_by_y_count": dict(counts),
        "physical_object": manifest["physical_object"], "coefficient_scale": 32,
        "coefficient_relation": "C4 = 32*c4; source already scaled, no multiplication performed",
        "complete_source_hash_counts_order_and_histogram_checked": True,
    }


@dataclass(frozen=True)
class Sample:
    word: bytes
    value: int
    source_line: int
    family: bytes
    positions: tuple[int, int]
    y_types: tuple[int, int]
    conditions: tuple[int, ...]


def construct_samples(zero_y, targets):
    family_tables = {}
    samples = []
    for word, value, line in targets:
        positions = tuple(index for index, letter in enumerate(word) if letter >= 6)
        require(len(positions) == 2, "Expected exactly two y letters")
        r, s = positions
        family = bytes(9 if letter >= 6 else letter for letter in word)
        if family not in family_tables:
            replacement = bytearray(family)
            conditions = []
            for i in range(6):
                replacement[r] = i
                for j in range(6):
                    replacement[s] = j
                    require(all(letter < 6 for letter in replacement), "Condition contains a y")
                    conditions.append(zero_y.get(bytes(replacement), 0))
            family_tables[family] = tuple(conditions)
        samples.append(Sample(word, value, line, family, positions,
                              (word[r], word[s]), family_tables[family]))
    return samples, family_tables


def verify_conditions_without_family_cache(samples, zero_y):
    """Independent construction from each original target, not the cached template."""
    checked = 0
    for row_index, sample in enumerate(samples, 1):
        r, s = sample.positions
        require(sample.word[r] == sample.y_types[0] and sample.word[s] == sample.y_types[1],
                "Query ordering mismatch")
        for slot, expected in enumerate(sample.conditions):
            i, j = divmod(slot, 6)
            replacement = bytes(i if k == r else j if k == s else sample.word[k]
                                for k in range(8))
            require(all(letter < 6 for letter in replacement), "Recheck source is not 0y")
            require(zero_y.get(replacement, 0) == expected, "Condition reconstruction mismatch")
            # Prove fixed-two-block base100 encoding is exact and injective here.
            high, low = divmod(abs(expected), 100)
            require(0 <= high < 100 and 0 <= low < 100, "Condition encoding overflow")
            decoded = (1 if expected >= 0 else -1) * (100 * high + low)
            require(decoded == expected, "Condition encoding round-trip mismatch")
            checked += 1
        require(abs(sample.value) < 10000, "Target encoding overflow")
        if row_index % 25000 == 0:
            print(f"Independently checked {row_index:,} target rows", flush=True)
    return checked


def input_key(sample, representation):
    if representation == "c36_y_types":
        return (sample.conditions, sample.y_types)
    if representation == "c36_y_types_positions":
        return (sample.conditions, sample.y_types, sample.positions)
    raise ValueError(representation)


def input_payload(sample, representation):
    result = {"ordered_coefficients_C4": list(sample.conditions),
              "ordered_y_types": [ALIASES[letter] for letter in sample.y_types]}
    if representation == "c36_y_types_positions":
        result["y_positions_one_based"] = [position + 1 for position in sample.positions]
    return result


def witness(sample):
    return {"word": [LETTERS[x] for x in sample.word],
            "word_aliases": [ALIASES[x] for x in sample.word],
            "y_positions_one_based": [x + 1 for x in sample.positions],
            "target_C4": sample.value, "source_line_one_based": sample.source_line}


def summarize(groups):
    rows = unique_pairs = conflict_groups = conflict_rows = forced_errors = 0
    magnitude_correct = sign_correct = repeat_groups = 0
    sizes = Counter()
    for members in groups.values():
        counts = Counter(sample.value for sample in members)
        size = len(members)
        rows += size
        unique_pairs += len(counts)
        sizes[size] += 1
        repeat_groups += size > 1
        if len(counts) > 1:
            conflict_groups += 1
            conflict_rows += size
        forced_errors += size - max(counts.values())
        magnitudes, signs = Counter(), Counter()
        for value, count in counts.items():
            magnitudes[abs(value)] += count
            signs[1 if value > 0 else -1] += count
        magnitude_correct += max(magnitudes.values())
        sign_correct += max(signs.values())
    require(rows > 0, "Empty audit pool")
    return {
        "rows": rows, "unique_inputs": len(groups),
        "unique_input_label_pairs": unique_pairs,
        "duplicate_input_rows_beyond_first": rows - len(groups),
        "duplicate_input_label_rows_beyond_first": rows - unique_pairs,
        "repeated_input_groups": repeat_groups,
        "deterministic_input_groups": len(groups) - conflict_groups,
        "conflicting_input_groups": conflict_groups, "rows_in_conflicting_groups": conflict_rows,
        "rows_in_conflicting_groups_fraction": conflict_rows / rows,
        "unavoidable_errors_on_uniform_original_rows": forced_errors,
        "maximum_correct_on_uniform_original_rows": rows - forced_errors,
        "exact_accuracy_ceiling_percent": 100 * (rows - forced_errors) / rows,
        "magnitude_accuracy_ceiling_percent": 100 * magnitude_correct / rows,
        "sign_accuracy_ceiling_percent": 100 * sign_correct / rows,
        "group_size_histogram": dict(sorted(sizes.items())), "largest_group_rows": max(sizes),
    }


def audit_representation(samples, representation, output):
    groups = defaultdict(list)
    for sample in samples:
        groups[input_key(sample, representation)].append(sample)
    summary = summarize(groups)
    # A second aggregation route (sort/groupby) cross-checks the exact bound.
    sorted_pairs = sorted((input_key(sample, representation), sample.value) for sample in samples)
    alternative_groups = alternative_correct = 0
    for _, pairs in groupby(sorted_pairs, key=lambda pair: pair[0]):
        counts = Counter(pair[1] for pair in pairs)
        alternative_groups += 1
        alternative_correct += max(counts.values())
    require(alternative_groups == summary["unique_inputs"], "Independent group count mismatch")
    require(alternative_correct == summary["maximum_correct_on_uniform_original_rows"],
            "Independent accuracy ceiling mismatch")
    del sorted_pairs
    summary["sort_groupby_independent_crosscheck_passed"] = True
    all_zero = {key: members for key, members in groups.items() if not any(key[0])}
    summary["all_zero_condition_subset"] = summarize(all_zero) if all_zero else None
    all_groups_path = output / f"{representation}.groups.jsonl.gz"
    conflicts_path = output / f"{representation}.conflicts.jsonl"
    examples = []
    with gzip.open(all_groups_path, "xt", encoding="utf-8") as all_stream, \
            conflicts_path.open("x", encoding="utf-8") as conflict_stream:
        for index, (key, members) in enumerate(sorted(groups.items())):
            first = members[0]
            counts = Counter(sample.value for sample in members)
            payload = input_payload(first, representation)
            serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            entry = {"group_id": index,
                     "input_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
                     "input": payload, "rows": len(members),
                     "target_histogram": dict(sorted(counts.items())),
                     "source_lines_one_based": [sample.source_line for sample in members],
                     "distinct_families": len({sample.family for sample in members}),
                     "conflicting": len(counts) > 1}
            all_stream.write(json.dumps(entry, separators=(",", ":")) + "\n")
            if len(counts) > 1:
                representative_by_label = {}
                for sample in members:
                    representative_by_label.setdefault(sample.value, sample)
                conflicting = {k: v for k, v in entry.items() if k != "source_lines_one_based"}
                conflicting["witnesses"] = [witness(representative_by_label[c]) for c in sorted(counts)]
                conflict_stream.write(json.dumps(conflicting, separators=(",", ":")) + "\n")
                # Prefer compact and nonzero-condition examples for the human report.
                score = (not any(first.conditions), len(counts), len(members), index)
                examples.append((score, conflicting))
    # Check the serialized partition covers every original target exactly once.
    serialized_lines = []
    read_groups = read_correct = read_conflicts = 0
    with gzip.open(all_groups_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            entry = json.loads(line)
            counts = entry["target_histogram"]
            require(sum(counts.values()) == entry["rows"], "Serialized histogram mismatch")
            require(len(entry["source_lines_one_based"]) == entry["rows"], "Serialized members mismatch")
            require(entry["conflicting"] == (len(counts) > 1), "Serialized conflict flag mismatch")
            serialized_lines.extend(entry["source_lines_one_based"])
            read_groups += 1
            read_correct += max(counts.values())
            read_conflicts += len(counts) > 1
    require(sorted(serialized_lines) == sorted(sample.source_line for sample in samples),
            "Serialized groups do not partition target pool")
    require(read_groups == summary["unique_inputs"] and
            read_correct == summary["maximum_correct_on_uniform_original_rows"] and
            read_conflicts == summary["conflicting_input_groups"], "Serialized summary mismatch")
    summary["serialized_partition_readback_passed"] = True
    summary["representative_conflicts"] = [entry for _, entry in sorted(examples, key=lambda x: x[0])[:3]]
    summary["files"] = {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                        for path in (all_groups_path, conflicts_path)}
    return summary


def render_report(report):
    summaries = report["representations"]
    lines = [
        "# Hexagon 四圈 2y 非零目标：38/40 项输入冲突审计", "",
        f"执行时间（UTC）：{report['created_at_utc']}", "",
        "本次为用户单独批准的本机 CPU 精确数据审计。未运行模型、训练、推理或划分训练集。", "",
        "## 范围与定义", "",
        "- 仅审计完整源数据中的 110,082 个非零 2y 目标，不包括零目标或其他圈数。",
        "- 条件和目标均直接使用已缩放的 C4=32*c4；未再次缩放。",
        "- 普通字母顺序：hat_a, hat_b, hat_c, hat_d, hat_e, hat_f。",
        "- 条件矩阵行对应较早的 y 位置，列对应较晚的 y 位置；slot=6*i+j。",
        "- 38 项输入：36 个有序整数 + 两个有序 y 类型。",
        "- 40 项输入：再加入两个 y 的绝对位置。报告位置从 1 开始。",
        "- 相同输入同标签是重复；相同输入不同标签是表示冲突，不能删除或覆盖标签。", "",
        "## 结果", "",
        "| 表示 | 不同输入数 | 冲突输入组数 | 冲突组涉及行数 | 最少必错行数 | 行加权 exact 上限 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, title in zip(REPRESENTATIONS, ("38 项", "40 项")):
        s = summaries[name]
        lines.append(f"| {title} | {s['unique_inputs']:,} | {s['conflicting_input_groups']:,} | "
                     f"{s['rows_in_conflicting_groups']:,} | {s['unavoidable_errors_on_uniform_original_rows']:,} | "
                     f"{s['exact_accuracy_ceiling_percent']:.6f}% |")
    lines += ["", "上限 = Σ_x max_c n(x,c) / N。其中每个原始目标 word 等权；假定任意确定性预测器仅看所声明输入。",
              "这是使用全审计池标签计算的表示可辨识性上限，不是训练成绩，也不是对某次测试集或未见数据的保证。",
              "冲突组涉及的行数不等于最少必错行数；组内仍可正确预测出现最多的标签。", "",
              "## 数据统计", "",
              f"- 不同 family：{report['pool']['families']:,}。",
              f"- 不同 36 项条件表：{report['pool']['distinct_condition_tables']:,}。",
              f"- 36 项全零的目标行数：{report['pool']['all_zero_condition_target_rows']:,}。",
              f"- 非零目标系数范围：{report['pool']['minimum_target']} 至 {report['pool']['maximum_target']}。",
              f"- 不同非零目标系数：{report['pool']['distinct_target_coefficients']}。", "",
              "## 冲突见证", ""]
    for name, title in zip(REPRESENTATIONS, ("38 项", "40 项")):
        lines += [f"### {title}", ""]
        examples = summaries[name]["representative_conflicts"]
        if not examples:
            lines += ["本审计池未发现冲突；这不是对零目标或其他圈数的充分性证明。", ""]
        for example in examples[:2]:
            lines += [f"输入组 {example['group_id']}；共 {example['rows']} 行；标签分布 {example['target_histogram']}。", "",
                      "```json", json.dumps(example["input"], ensure_ascii=False), "```", ""]
            for item in example["witnesses"][:3]:
                lines.append(f"- {' '.join(item['word_aliases'])} → C4={item['target_C4']}；源文件第 "
                             f"{item['source_line_one_based']} 行；y 位置 {item['y_positions_one_based']}。")
            lines.append("")
    lines += ["## 核验与限制", "",
              "- 源文件和 manifest SHA-256、全部行数、排序、字母、整数格式及全源系数直方图已检查。",
              f"- 从每个原始目标独立重构并核对 {report['checks']['independent_condition_lookups']:,} 次条件查询；不依赖 family 缓存。",
              "- 字典聚合与独立排序聚合得到相同组数和准确率上限；输出分组文件已完整读回核对。",
              "- 两种表示均使用精确整数元组分组，不以摘要是否碰撞来判断输入相等。",
              "- 40 项是 38 项的细分，上限不得降低；完整目标 word 作为参考表示没有重复或冲突。",
              "- 未生成 random-row split；以后应防止完全相同压缩输入跨 split，但这不等于必须隔离整个 family。",
              "- 未丢弃冲突、未选多数标签作为新真值、未改变原始目标分布。",
              "- 未重做完整可积性验证；源数据的物理正确性仍依赖既有提取与缩放证据。", "",
              "## 可复现性", "", f"源文件：`{report['source']['path']}`", "",
              f"源 SHA-256：`{SOURCE_SHA256}`", "",
              f"审计脚本：`{report['execution']['script']}`", "",
              f"脚本 SHA-256：`{report['execution']['script_sha256']}`", "",
              "audit.json 包含完整统计、环境、检查项及产物哈希；*.groups.jsonl.gz 包含全部输入组及其源行号；",
              "*.conflicts.jsonl 包含全部冲突组和每种标签的一个原始 word 见证。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="New directory outside code repository and Symbol_Data")
    args = parser.parse_args()
    started = time.perf_counter()
    source = args.source.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    repository = Path(__file__).resolve().parents[3]
    symbol_data = source.parent.parent
    require(not output.exists(), "Output already exists; choose a fresh directory")
    for forbidden in (repository, symbol_data):
        require(output != forbidden and forbidden not in output.parents, "Output inside protected tree")
    print("Checking immutable source and reading all 243,000 rows", flush=True)
    zero_y, targets, source_info = read_source(source)
    samples, family_tables = construct_samples(zero_y, targets)
    print(f"Constructed {len(samples):,} nonzero targets across {len(family_tables):,} families", flush=True)
    checked = verify_conditions_without_family_cache(samples, zero_y)
    output.mkdir(parents=True, exist_ok=False)
    summaries = {}
    for representation in REPRESENTATIONS:
        print(f"Auditing {representation}", flush=True)
        summaries[representation] = audit_representation(samples, representation, output)
        s = summaries[representation]
        print(f"  unique inputs={s['unique_inputs']:,}; conflicting groups={s['conflicting_input_groups']:,}; "
              f"exact ceiling={s['exact_accuracy_ceiling_percent']:.6f}%", flush=True)
    a, b = (summaries[name] for name in REPRESENTATIONS)
    require(b["unique_inputs"] >= a["unique_inputs"], "Positions reduced number of input groups")
    require(b["maximum_correct_on_uniform_original_rows"] >= a["maximum_correct_on_uniform_original_rows"],
            "Positions worsened empirical ceiling")
    require(len({sample.word for sample in samples}) == len(samples), "Full-word reference is not unique")
    require(sha256(source) == SOURCE_SHA256 and
            sha256(source.parent / "scaling_manifest.json") == MANIFEST_SHA256, "Source changed during audit")
    family_sizes = Counter(sample.family for sample in samples)
    zero_condition_rows = [sample for sample in samples if not any(sample.conditions)]
    report = {
        "schema_version": 1, "status": "completed_exact_data_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "four-loop weight8 nonzero 2y targets only; no training or split",
        "source": source_info,
        "pool": {"rows": len(samples), "families": len(family_tables),
                 "nonzero_queries_per_family_histogram": histogram(family_sizes.values()),
                 "distinct_condition_tables": len(set(family_tables.values())),
                 "all_zero_condition_families": sum(not any(c) for c in family_tables.values()),
                 "all_zero_condition_target_rows": len(zero_condition_rows),
                 "all_zero_condition_target_histogram": histogram(s.value for s in zero_condition_rows),
                 "nonzero_condition_slots_per_family_histogram": histogram(sum(c != 0 for c in values)
                                                                           for values in family_tables.values()),
                 "minimum_target": min(s.value for s in samples), "maximum_target": max(s.value for s in samples),
                 "distinct_target_coefficients": len({s.value for s in samples}),
                 "target_histogram": histogram(s.value for s in samples),
                 "ordered_y_pair_histogram": histogram(tuple(ALIASES[x] for x in s.y_types) for s in samples)},
        "representations": summaries,
        "checks": {"independent_condition_lookups": checked,
                   "source_hashes_unchanged_at_end": True, "positions_refinement_ceiling_monotonic": True,
                   "full_word_reference_unique_inputs": len(samples),
                   "full_word_reference_conflicts": 0,
                   "model_run": False, "split_generated": False, "source_modified": False},
        "execution": {"script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__),
                      "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                      "argv": sys.argv, "dependencies": "Python standard library only",
                      "authorization": "User explicitly approved local CPU data audit; no model execution",
                      "elapsed_seconds_before_report_write": time.perf_counter() - started},
    }
    write_json(output / "audit.json", report)
    with (output / "REPORT.md").open("x", encoding="utf-8") as stream:
        stream.write(render_report(report))
    print(f"Completed. Report: {output / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
