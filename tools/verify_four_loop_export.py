#!/usr/bin/env python3
"""Verify every four-loop exported coefficient by forward prefix propagation.

The extractor contracts the root backward and expands cached low-weight symbols.
This verifier instead starts from the empty word, propagates coefficient vectors
forward through inverse coproduct edges and projects on the root at the last slot.
No expanded basis dictionaries or suffix partitions are consulted. Both algorithms
share parsed source tables, not an independent physical data source.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import math
from pathlib import Path
import random
import time

import extract_four_loop_symbol as source


def forward_verify(transitions, root, symbol, *, verbose=False):
    """Exhaust all reachable prefixes; match all nonzero outputs, including support."""
    require = source.require
    weights = {node[1] for node in root}
    require(len(weights) == 1, "Mixed root weights")
    weight = next(iter(weights))
    require(weight >= 2, "Verifier requires weight>=2")
    denominator = math.lcm(*(value.denominator for value in root.values()))
    root_int = {node: value.numerator * (denominator // value.denominator) for node, value in root.items()}
    inverse = {}
    for parent, components in transitions.items():
        if parent[1] >= weight:
            continue
        for letter, row in components.items():
            for child, coefficient in row.items():
                key = (parent[1], child)
                inverse.setdefault(key, []).append((letter, parent, coefficient))
    final_rows = {}
    for parent, scalar in root_int.items():
        for letter, row in transitions[parent].items():
            combined = final_rows.setdefault(letter, {})
            for child, coefficient in row.items():
                combined[child] = combined.get(child, 0) + scalar * coefficient
    final_rows = {letter: {child: c for child, c in row.items() if c}
                  for letter, row in final_rows.items()}
    final_rows = {letter: row for letter, row in final_rows.items() if row}
    visited, matched, projected_zero = 0, 0, 0
    started = time.monotonic()

    def visit(prefix, vector):
        nonlocal visited, matched, projected_zero
        visited += 1
        if visited % 4096 == 0:
            require(time.monotonic() - started < 300, "Forward verification time limit")
            require(source.peak_rss_bytes() < source.MAX_RSS_BYTES, "Forward verification RSS limit")
            require(visited < 2_000_000, "Forward prefix count limit")
        if len(prefix) == weight - 1:
            for letter, row in final_rows.items():
                numerator = sum(value * row.get(node, 0) for node, value in vector.items())
                if not numerator:
                    projected_zero += 1
                    continue
                word = prefix + bytes([letter])
                observed = symbol.get(word)
                require(observed is not None, f"Missing nonzero word {list(word)}")
                require(numerator * observed.denominator == observed.numerator * denominator,
                        f"Wrong coefficient at {list(word)}")
                matched += 1
            return
        components = {}
        for node, scalar in vector.items():
            for letter, parent, coefficient in inverse.get((len(prefix) + 1, node), ()):
                component = components.setdefault(letter, {})
                component[parent] = component.get(parent, 0) + scalar * coefficient
        for letter, component in sorted(components.items()):
            component = {node: value for node, value in component.items() if value}
            if component:
                visit(prefix + bytes([letter]), component)
                if verbose and not prefix:
                    print(f"Verified first entry {source.LETTERS[letter]}: {matched:,} nonzero rows matched.", flush=True)

    visit(b"", {None: 1})
    require(matched == len(symbol), "Export contains extra unsupported words")
    return {"all_nonzero_rows_verified_by_forward_propagation": matched,
            "forward_reachable_prefixes": visited, "zero_final_projections": projected_zero,
            "complete_nonzero_support_match": True,
            "forward_elapsed_seconds": round(time.monotonic() - started, 3)}


def fixed_word_checks(transitions, root, symbol):
    # A third, scalar recursion with no forward coefficient vectors or expansions.
    rng = random.Random(20260915)
    query = source.previous.SymbolExtractor(transitions)
    denominator = math.lcm(*(c.denominator for c in root.values()))
    scaled = {node: c.numerator * (denominator // c.denominator) for node, c in root.items()}
    nonzero = rng.sample(sorted(symbol), min(256, len(symbol)))
    zeros = set()
    while len(zeros) < 256:
        word = bytes([rng.randrange(3), rng.randrange(6), *[rng.randrange(9) for _ in range(5)],
                      rng.randrange(3, 9)])
        if word not in symbol and sum(i >= 6 for i in word) % 2 == 0:
            zeros.add(word)
    for word in [*nonzero, *sorted(zeros)]:
        numerator = sum(c * query.query_basis(node, word) for node, c in scaled.items())
        source.require(Fraction(numerator, denominator) == symbol.get(word, Fraction(0)),
                       "Fixed-word scalar query disagrees")
    return {"additional_scalar_queries_nonzero": len(nonzero),
            "additional_scalar_queries_zero_even_y": len(zeros), "query_seed": 20260915}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--directory", type=Path, required=True)
    args = cli.parse_args()
    destination = args.directory / "export_verification.json"
    source.require(not destination.exists(), "Verification report exists; refusing overwrite")
    started = time.monotonic()
    manifest_path = args.directory / "extraction_manifest.json"
    manifest = source.json.loads(manifest_path.read_text())
    source.require(source.sha256(Path(source.__file__)) == manifest["implementation_sha256"], "Extractor source changed")
    for name, info in manifest["outputs"].items():
        source.require(source.sha256(args.directory / name) == info["sha256"], "Output hash mismatch")
    source.require(source.sha256(Path(manifest["regression_reference"]["path"])) == source.PREVIOUS_DATA_SHA,
                   "Three-loop reference changed")
    definitions, _ = source.read_definitions(Path(manifest["source"]["path"]))
    transitions, roots = source.compile_tables(definitions)
    source.base_point_checks(definitions, roots)
    symbol = source.read_export(args.directory / source.DATA_NAME, 8)
    stats = source.statistics(symbol)
    for key, value in stats.items():
        source.require(manifest[key] == value, f"Statistic mismatch: {key}")
    source.check_structural(symbol, 8)
    source.check_symmetry(symbol)
    print(f"Read {len(symbol):,} canonical rows. Running independent forward propagation.", flush=True)
    report = forward_verify(transitions, roots[4], symbol, verbose=True)
    report.update(fixed_word_checks(transitions, roots[4], symbol))
    report.update({"gzip_crc_read_to_eof": True, "canonical_reduced_rationals": True,
                   "strict_word_order_and_uniqueness": True, "histograms_match": True,
                   "source_and_output_hashes_match": True, "structural_and_symmetry_checks": True,
                   "extraction_manifest_sha256": source.sha256(manifest_path),
                   "verifier_sha256": source.sha256(Path(__file__)),
                   "peak_rss_bytes": source.peak_rss_bytes(),
                   "elapsed_seconds": round(time.monotonic() - started, 3),
                   "scope_note": "Independent forward propagation and scalar queries share parsed coproduct tables; not an independent physical source or full integrability proof."})
    source.write_json(destination, report)
    print(source.json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
