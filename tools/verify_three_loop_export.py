#!/usr/bin/env python3
"""Read back the step-2 artifact and check every row by fixed-word recursion."""

import argparse
from collections import Counter
from fractions import Fraction
import gzip
import json
from pathlib import Path
import random
import re
import time

from extract_three_loop_symbol import (
    LETTERS, Parser, SymbolExtractor, base_point_checks, compile_tables,
    read_definitions, require, sha256, write_json,
)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--directory", type=Path, required=True)
    args = cli.parse_args()
    destination = args.directory / "export_verification.json"
    require(not destination.exists(), "Verification report exists; refusing overwrite")
    started = time.monotonic()
    manifest_path = args.directory / "extraction_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    implementation = Path(__file__).with_name("extract_three_loop_symbol.py")
    require(sha256(implementation) == manifest["implementation_sha256"], "Extractor source changed")
    filename = "three_loop_mhv_symbol_native.jsonl.gz"
    archive = args.directory / filename
    require(sha256(archive) == manifest["outputs"][filename]["sha256"], "Export hash mismatch")
    require(sha256(args.directory / "dependency_graph.json") ==
            manifest["outputs"]["dependency_graph.json"]["sha256"], "Dependency graph hash mismatch")
    definitions, _ = read_definitions(Path(manifest["source"]["path"]))
    transitions, roots = compile_tables(definitions)
    base_point_checks(definitions, roots)
    extractor = SymbolExtractor(transitions)
    row_count, total_bytes, previous = 0, 0, None
    words, histogram, y_counts, denominators = set(), Counter(), Counter(), Counter()
    with gzip.open(archive, "rt", encoding="ascii") as stream:
        for line in stream:
            total_bytes += len(line)
            require(len(line) <= 1024 and total_bytes <= 2_000_000, "Export size limit")
            row = json.loads(line)
            require(set(row) == {"word", "numerator", "denominator"}, "Unexpected row fields")
            require(isinstance(row["word"], list) and len(row["word"]) == 6, "Bad word")
            word = bytes(LETTERS.index(letter) for letter in row["word"])
            require(previous is None or previous < word, "Duplicate or unsorted word")
            previous = word
            for name in ("numerator", "denominator"):
                require(isinstance(row[name], str) and re.fullmatch(r"-?[0-9]+", row[name]),
                        "Coefficient must be an integer string")
            value = Fraction(int(row["numerator"]), int(row["denominator"]))
            require(value != 0 and row["numerator"] == str(value.numerator) and
                    row["denominator"] == str(value.denominator), "Noncanonical rational")
            require(extractor.query(roots[3], word) == value, "Independent coefficient query mismatch")
            words.add(word)
            histogram[str(value)] += 1
            y_counts[str(sum(letter >= 6 for letter in word))] += 1
            denominators[str(value.denominator)] += 1
            row_count += 1
    require(row_count == manifest["rows"], "Row count mismatch")
    require(dict(histogram) == manifest["coefficient_histogram"], "Coefficient histogram mismatch")
    require(dict(y_counts) == manifest["nonzero_rows_by_y_count"], "Y-count histogram mismatch")
    require(dict(denominators) == manifest["denominator_histogram"], "Denominator histogram mismatch")
    rng, zero_words = random.Random(20260914), set()
    while len(zero_words) < 1024:
        word = bytes([rng.randrange(3), rng.randrange(6), *[rng.randrange(9) for _ in range(3)],
                      rng.randrange(3, 9)])
        if word not in words:
            zero_words.add(word)
    for word in sorted(zero_words):
        require(extractor.query(roots[3], word) == 0, "Absent word has a nonzero queried coefficient")
    report = {"all_exported_rows_verified_by_fixed_word_query": row_count,
              "additional_absent_words_verified": len(zero_words), "zero_query_seed": 20260914,
              "gzip_crc_read_to_eof": True, "canonical_reduced_rationals": True,
              "strict_word_order_and_uniqueness": True, "histograms_match": True,
              "source_and_output_hashes_match": True, "uncompressed_export_bytes": total_bytes,
              "extraction_manifest_sha256": sha256(manifest_path),
              "verifier_sha256": sha256(Path(__file__)),
              "elapsed_seconds": round(time.monotonic() - started, 3),
              "scope_note": "Independent word-query recursion shares parsed coproduct tables; not an independent physical data source or proof of full integrability."}
    write_json(destination, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
