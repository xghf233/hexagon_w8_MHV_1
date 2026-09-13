#!/usr/bin/env python3
"""Bounded exact extraction of the pinned native-alphabet four-loop MHV symbol.

No Wolfram execution, eval, network, alphabet conversion, label rescaling,
training samples or training. The three-loop implementation remains immutable.
"""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import resource
import sys
import time
import zipfile

import extract_three_loop_symbol as previous

require = previous.require
sha256 = previous.sha256
write_json = previous.write_json
LETTERS = previous.LETTERS
ALIASES = ("hat_a", "hat_b", "hat_c", "hat_d", "hat_e", "hat_f", "y_U", "y_V", "y_W")
PREVIOUS_SCRIPT_SHA = "04ee06a2b370675294dd42b0189f70cb092da519b7ac61c17509861949a47a80"
PREVIOUS_DATA_SHA = "e46802fb750778e769adb89e73171e935ddf4414ca1efbc843746c6acd4f48b1"
DIMENSIONS = {**previous.DIMENSIONS, ("YO", 6): 13, ("YE", 7): 170,
              ("YO", 7): 30, ("YE", 8): 313}
CONSTANTS = {**previous.CONSTANTS, ("YE", 8, 313): ("Zeta", 8)}
MAX_MONOMIALS = 8192
MAX_WORDS = 2_000_000
MAX_SECONDS = 300
MAX_RSS_BYTES = 768 * 1024**2
DATA_NAME = "four_loop_mhv_symbol_native.jsonl.gz"


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def clean(poly):
    result = {key: value for key, value in poly.items() if value}
    require(len(result) <= MAX_MONOMIALS, "Polynomial size limit exceeded")
    require(all(max(abs(v.numerator).bit_length(), v.denominator.bit_length()) <= 4096
                for v in result.values()), "Coefficient bit limit exceeded")
    return result


def add(left, right, factor=1):
    result = dict(left)
    for key, value in right.items():
        result[key] = result.get(key, Fraction(0)) + factor * value
    return clean(result)


def multiply(left, right):
    require(len(left) * len(right) <= 100_000, "Polynomial product size limit")
    result = {}
    for a, va in left.items():
        for b, vb in right.items():
            key = tuple(sorted(a + b))
            require(len(key) <= 4, "Monomial degree limit")
            result[key] = result.get(key, Fraction(0)) + va * vb
    return clean(result)


class Parser(previous.Parser):
    """Arithmetic whitelist extended only for bounded powers and MZV[{5,3}]."""

    token = re.compile(r"\d+|[A-Za-z][A-Za-z0-9]*|[+*/(),\[\]{}^\-]")

    def expression(self, depth):
        result = self.product(depth)
        while self.peek() in ("+", "-"):
            operator = self.take()
            result = add(result, self.product(depth), 1 if operator == "+" else -1)
        return result

    def product(self, depth):
        result = self.unary(depth)
        while self.peek() in ("*", "/"):
            operator = self.take()
            right = self.unary(depth)
            if operator == "*":
                result = multiply(result, right)
            else:
                require(set(right) == {()} and right[()] != 0, "Only nonzero scalar division")
                result = clean({key: value / right[()] for key, value in result.items()})
        return result

    def unary(self, depth):
        require(depth < 64, "Expression nesting limit")
        if self.peek() in ("+", "-"):
            sign = self.take()
            value = self.unary(depth + 1)
            return {key: v if sign == "+" else -v for key, v in value.items()}
        result = self.atom(depth)
        if self.peek() == "^":
            self.take("^")
            exponent = self.take()
            require(exponent.isdecimal() and 0 <= int(exponent) <= 4, "Power must be 0..4")
            powered = {(): Fraction(1)}
            for _ in range(int(exponent)):
                powered = multiply(powered, result)
            result = powered
        return result

    def atom(self, depth):
        if self.peek() == "(":
            self.take("(")
            result = self.expression(depth + 1)
            self.take(")")
            return result
        token = self.take()
        if token.isdecimal():
            return clean({(): Fraction(int(token))})
        require(token in ("g", "YE", "YO", "Zeta", "MZV"), f"Unknown atom {token}")
        self.take("[")
        if token == "MZV":
            for part in ("{", "5", ",", "3", "}", "]"):
                self.take(part)
            return {(("MZV", 5, 3),): Fraction(1)}
        indices = []
        for index in range(2 if token in ("YE", "YO") else 1):
            if index:
                self.take(",")
            value = self.take()
            require(value.isdecimal() and 0 < int(value) <= 1000, "Invalid atom index")
            indices.append(int(value))
        self.take("]")
        return {((token, *indices),): Fraction(1)}


def expected_names():
    names = {f"e{loop}uvw" for loop in range(1, 5)}
    names.update(f"YE111{weight}" for weight in range(1, 9))
    for family, weight in DIMENSIONS:
        names.update(f"M{family}{weight}[{letter}]" for letter in LETTERS)
    require(len(names) == 129, "Wrong definition set")
    return names


def read_definitions(archive):
    require(sha256(Path(previous.__file__)) == PREVIOUS_SCRIPT_SHA, "Three-loop helper changed")
    require(archive.stat().st_size == 21_944_268, "Archive size mismatch")
    require(sha256(archive) == previous.ARCHIVE_SHA, "Archive SHA-256 mismatch")
    wanted = expected_names()
    definitions, locations = {}, {}
    header = re.compile(r"^([A-Za-z][A-Za-z0-9]*(?:\[[^\]\r\n]+\])?)\s*=\s*(.*)")
    active, chunks, start, last = None, [], 0, 0

    def finish():
        if active is None:
            return
        require(active not in definitions, f"Duplicate definition {active}")
        body = "".join(chunks).strip()
        definitions[active] = Parser(body).parse()
        locations[active] = {"first_line": start, "last_line": last,
                             "rhs_sha256": hashlib.sha256(body.encode()).hexdigest()}

    h, size = hashlib.sha256(), 0
    with zipfile.ZipFile(archive) as package:
        require(package.namelist() == [previous.MEMBER_NAME], "Unexpected ZIP members")
        info = package.getinfo(previous.MEMBER_NAME)
        require(info.file_size == previous.MEMBER_BYTES and not info.flag_bits & 1,
                "Invalid member size or encryption")
        with package.open(info) as stream:
            for number, raw in enumerate(stream, 1):
                size += len(raw)
                require(size <= previous.MEMBER_BYTES and len(raw) <= 16_384, "Stream size limit")
                h.update(raw)
                line = raw.decode("ascii")
                match = header.match(line)
                if match:
                    finish()
                    name = match[1].replace(" ", "")
                    active = name if name in wanted else None
                    chunks = [match[2] + "\n"] if active else []
                    start = last = number
                elif active is not None:
                    require(not line.lstrip().startswith("(*"), "Unexpected definition boundary")
                    chunks.append(line)
                    if line.strip():
                        last = number
                    require(sum(map(len, chunks)) <= 100_000, "Definition size limit")
            finish()
    require(size == previous.MEMBER_BYTES and h.hexdigest() == previous.MEMBER_SHA,
            "Member SHA-256 mismatch")
    require(set(definitions) == wanted, f"Missing definitions: {wanted - definitions.keys()}")
    return definitions, locations


def coefficient_of_selector(poly, index):
    result = {}
    for monomial, value in poly.items():
        selectors = [atom for atom in monomial if atom[0] == "g"]
        require(len(selectors) == 1, "Invalid base-point selector expression")
        require(all(atom[0] in ("g", "Zeta", "MZV") for atom in monomial), "Invalid base-point atom")
        if selectors[0] == ("g", index):
            key = tuple(atom for atom in monomial if atom[0] != "g")
            result[key] = result.get(key, Fraction(0)) + value
    return clean(result)


def compile_tables(definitions):
    nodes = {(family, weight, index) for (family, weight), count in DIMENSIONS.items()
             for index in range(1, count + 1)}
    transitions = {node: {} for node in nodes}
    for (family, weight), count in DIMENSIONS.items():
        for letter_id, letter in enumerate(LETTERS):
            poly = definitions[f"M{family}{weight}[{letter}]"]
            for monomial, coefficient in poly.items():
                selectors = [atom for atom in monomial if atom[0] == "g"]
                children = [atom for atom in monomial if atom[0] in ("YE", "YO")]
                require(len(selectors) == 1 and 1 <= selectors[0][1] <= count, "Invalid selector")
                require(len(monomial) == (1 if weight == 1 else 2), "Invalid table monomial")
                require(len(children) == (0 if weight == 1 else 1), "Invalid child count")
                child = children[0] if children else None
                require(child is None or (child in nodes and child[1] == weight - 1),
                        "Missing child or non-decreasing dependency")
                require(child is None or (child[0] == family) == (letter_id < 6), "Parity mismatch")
                require(coefficient.denominator == 1, "Noninteger transition in pinned source")
                node = (family, weight, selectors[0][1])
                row = transitions[node].setdefault(letter_id, {})
                row[child] = row.get(child, 0) + int(coefficient)
    for table in transitions.values():
        for letter in list(table):
            table[letter] = {node: c for node, c in table[letter].items() if c}
            if not table[letter]:
                del table[letter]
    require({node for node, table in transitions.items() if not table} == set(CONSTANTS),
            "Unexpected derivative-free node")
    for node, zeta in CONSTANTS.items():
        require(coefficient_of_selector(definitions[f"YE111{node[1]}"], node[2]) == {(zeta,): Fraction(1)},
                "Constant base-point mismatch")
    roots = {}
    for loop in range(1, 5):
        root = {}
        for monomial, value in definitions[f"e{loop}uvw"].items():
            require(len(monomial) == 1 and monomial[0] in nodes, "Invalid root")
            node = monomial[0]
            require(node[:2] == ("YE", 2 * loop), "Wrong root parity/weight")
            root[node] = value
        roots[loop] = root
    require(len(roots[3]) == 64 and len(roots[4]) == 249, "Root dimension mismatch")
    return transitions, roots


def base_point_checks(definitions, roots):
    expected = {1: {}, 2: {(("Zeta", 4),): Fraction(-10)},
                3: {(("Zeta", 6),): Fraction(413, 3)},
                4: {(("Zeta", 8),): Fraction(-5477, 3), (("MZV", 5, 3),): Fraction(24),
                    (("Zeta", 3), ("Zeta", 5)): Fraction(120),
                    (("Zeta", 2), ("Zeta", 3), ("Zeta", 3)): Fraction(-24)}}
    for loop, root in roots.items():
        value = {}
        for node, coefficient in root.items():
            value = add(value, coefficient_of_selector(definitions[f"YE111{2*loop}"], node[2]), coefficient)
        require(value == expected[loop], f"Base-point check failed at loop {loop}")


class PartitionedExtractor:
    """Contract root combinations by suffix; expand only low-weight basis nodes."""

    def __init__(self, transitions, cutoff=4):
        require(1 <= cutoff <= 4, "Low-weight cutoff must be 1..4")
        self.transitions = transitions
        self.low = previous.SymbolExtractor(transitions)
        self.cutoff = cutoff
        self.started = time.monotonic()
        self.blocks = 0
        self.max_block_words = 0

    def guard(self, words=0):
        require(time.monotonic() - self.started <= MAX_SECONDS, "Extraction time limit")
        require(words <= MAX_WORDS, "Output word count limit")
        require(peak_rss_bytes() <= MAX_RSS_BYTES, "Peak RSS limit")

    def amplitude(self, root, progress=False):
        weights = {node[1] for node in root}
        require(len(weights) == 1, "Mixed root weights")
        weight = next(iter(weights))
        denominator = math.lcm(*(c.denominator for c in root.values()))
        initial = {node: c.numerator * (denominator // c.denominator) for node, c in root.items() if c}
        result = {}

        def visit(state, suffix, remaining):
            self.guard(len(result))
            if remaining <= self.cutoff:
                block = {}
                for node, scalar in state.items():
                    for prefix, coefficient in self.low.basis_symbol(node).items():
                        block[prefix] = block.get(prefix, 0) + scalar * coefficient
                block = {prefix: c for prefix, c in block.items() if c}
                self.blocks += 1
                self.max_block_words = max(self.max_block_words, len(block))
                self.guard(len(result) + len(block))
                for prefix, value in block.items():
                    word = prefix + suffix
                    require(len(word) == weight and word not in result, "Overlapping suffix partition")
                    result[word] = Fraction(value, denominator)
                return
            components = {}
            for node, scalar in state.items():
                require(node in self.transitions and node[1] == remaining, "Invalid recursive node")
                for letter, row in self.transitions[node].items():
                    component = components.setdefault(letter, {})
                    for child, coefficient in row.items():
                        component[child] = component.get(child, 0) + scalar * coefficient
            for letter, component in sorted(components.items()):
                component = {node: c for node, c in component.items() if c}
                if component:
                    visit(component, bytes([letter]) + suffix, remaining - 1)
                    if progress and not suffix:
                        print(f"Completed final entry {LETTERS[letter]}: {len(result):,} words accumulated.", flush=True)

        visit(initial, b"", weight)
        return result


def check_structural(symbol, weight):
    require(bool(symbol), "Empty symbol")
    for word, value in symbol.items():
        require(len(word) == weight and value != 0, "Invalid word/value")
        require(word[0] < 3 and word[1] < 6, "First/second entry mismatch")
        require(word[-1] >= 3, "MHV final entry mismatch")
        require(sum(letter >= 6 for letter in word) % 2 == 0, "Odd y parity")


def check_symmetry(symbol):
    # Cyclic U,V,W plus U<->V generate S3. Any accompanying global inversion
    # of y letters is invisible here because all MHV words have even y parity.
    permutations = ((1, 2, 0), (1, 0, 2))
    for permutation in permutations:
        mapping = bytes(3 * (i // 3) + permutation[i % 3] for i in range(9))
        for word, value in symbol.items():
            transformed = bytes(mapping[i] for i in word)
            require(symbol.get(transformed) == value, "S3 coefficient symmetry mismatch")


def read_export(path, weight):
    result, last = {}, None
    total = 0
    with gzip.open(path, "rt", encoding="ascii") as stream:
        for line in stream:
            total += len(line)
            require(len(line) <= 1024 and total <= 256 * 1024**2, "Export size limit")
            row = json.loads(line)
            require(set(row) == {"word", "numerator", "denominator"}, "Wrong export fields")
            require(isinstance(row["word"], list) and len(row["word"]) == weight, "Bad word shape")
            word = bytes(LETTERS.index(letter) for letter in row["word"])
            require(last is None or last < word, "Unsorted or duplicate word")
            for field in ("numerator", "denominator"):
                require(isinstance(row[field], str) and re.fullmatch(r"-?\d+", row[field]), "Invalid coefficient string")
            value = Fraction(int(row["numerator"]), int(row["denominator"]))
            require(value != 0 and str(value.numerator) == row["numerator"] and
                    str(value.denominator) == row["denominator"], "Noncanonical rational")
            result[word] = value
            last = word
            require(len(result) <= MAX_WORDS, "Export row limit")
    return result


def dependency_graph(transitions, root):
    reached, todo = set(), list(root)
    while todo:
        node = todo.pop()
        if node is not None and node not in reached:
            reached.add(node)
            todo.extend(child for row in transitions[node].values() for child in row)
    return [{"node": previous.node_name(node), "weight": node[1],
             "derivative_free_constant": node in CONSTANTS,
             "root_coefficient": str(root[node]) if node in root else None,
             "components": {LETTERS[letter]: {previous.node_name(child): str(value) for child, value in row.items()}
                            for letter, row in transitions[node].items()}}
            for node in sorted(reached, key=lambda n: (n[1], n[0], n[2]))]


def statistics(symbol):
    histogram = Counter(str(c) for c in symbol.values())
    y_counts = Counter(str(sum(i >= 6 for i in word)) for word in symbol)
    denominators = Counter(str(c.denominator) for c in symbol.values())
    return {"rows": len(symbol), "nonzero_rows_by_y_count": dict(sorted(y_counts.items())),
            "denominator_histogram": dict(sorted(denominators.items(), key=lambda x: int(x[0]))),
            "coefficient_histogram": dict(sorted(histogram.items(), key=lambda x: Fraction(x[0]))),
            "minimum_coefficient": str(min(symbol.values())), "maximum_coefficient": str(max(symbol.values())),
            "denominator_lcm_observed_only": math.lcm(*(int(d) for d in denominators))}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source", type=Path, required=True)
    cli.add_argument("--three-loop-reference", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    require(not args.output.exists(), "Output directory exists; refusing overwrite")
    require(sha256(args.three_loop_reference) == PREVIOUS_DATA_SHA, "Three-loop reference hash mismatch")
    started = time.monotonic()
    print("Parsing 129 pinned definitions; exact four-loop base-point check.", flush=True)
    definitions, locations = read_definitions(args.source)
    transitions, roots = compile_tables(definitions)
    base_point_checks(definitions, roots)
    extractor = PartitionedExtractor(transitions)
    for loop in (1, 2, 3):
        symbol = extractor.amplitude(roots[loop])
        check_structural(symbol, 2 * loop)
        if loop == 1:
            expected = {bytes(pair): Fraction(-1, 2)
                        for pair in ((1, 3), (2, 3), (0, 4), (2, 4), (0, 5), (1, 5))}
            require(symbol == expected, "One-loop reference mismatch")
        if loop == 3:
            require(symbol == read_export(args.three_loop_reference, 6), "Three-loop full regression mismatch")
        print(f"Loop {loop} regression: {len(symbol):,} nonzero words.", flush=True)
    symbol = extractor.amplitude(roots[4], progress=True)
    check_structural(symbol, 8)
    check_symmetry(symbol)
    graph = dependency_graph(transitions, roots[4])
    require(len(graph) == 652, "Dependency closure count changed")
    stats = statistics(symbol)
    extractor.guard(len(symbol))
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / DATA_NAME).open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            for word, value in sorted(symbol.items()):
                row = {"word": [LETTERS[i] for i in word], "numerator": str(value.numerator),
                       "denominator": str(value.denominator)}
                stream.write((json.dumps(row, separators=(",", ":")) + "\n").encode("ascii"))
    require(read_export(args.output / DATA_NAME, 8) == symbol, "Export read-back mismatch")
    write_json(args.output / "dependency_graph.json", graph)
    report = {"schema_version": 1, "stage": "four_loop_native_symbol_extraction",
              "source": {"path": str(args.source.resolve()), "archive_sha256": previous.ARCHIVE_SHA,
                         "member": previous.MEMBER_NAME, "member_bytes": previous.MEMBER_BYTES,
                         "member_sha256": previous.MEMBER_SHA, "root": "e4uvw",
                         "paper": "https://arxiv.org/abs/1903.10890"},
              "normalization": {"source": "2019 cosmic-normalized MHV", "target_symbol": "ordinary BDS-like MHV",
                                "function_relation": "E_BDS_like^(4) = e4uvw + 8*zeta(3)^2*e1uvw - 160*zeta(3)*zeta(5)",
                                "full_weight_8_symbols_equal": True,
                                "coupling": "g^2 = N*g_YM^2/(16*pi^2)", "integer_label_scaling_applied": False},
              "alphabet": list(LETTERS), "training_aliases": list(ALIASES),
              "alphabet_version": "hexagon_hat_native_v1", "alphabet_conversion_applied": False,
              "word_order": "first entry to last entry", "row_order": "lexicographic letter IDs 0..8",
              "coefficient_format": "exact reduced rational numerator/denominator strings",
              "loop": 4, "weight": 8, **stats, "dependency_nodes": len(graph),
              "algorithm": {"name": "suffix contraction with weight<=4 basis cache", "low_weight_cutoff": 4,
                            "partition_blocks_including_regression": extractor.blocks,
                            "maximum_block_words": extractor.max_block_words,
                            "cached_low_weight_word_entries": extractor.low.cached_words},
              "limits": {"max_seconds": MAX_SECONDS, "max_peak_rss_bytes": MAX_RSS_BYTES,
                         "max_words": MAX_WORDS, "max_polynomial_monomials": MAX_MONOMIALS},
              "checks": {"base_point_loops_1_2_3_4": True, "one_loop_exact_reference": True,
                         "three_loop_full_export_regression": True,
                         "first_and_second_entry_all_rows": True, "mhv_final_entry_all_rows": True,
                         "even_y_parity_all_rows": True, "cyclic_and_transposition_all_rows": True,
                         "exact_export_readback_all_rows": True,
                         "not_yet_checked": ["full integrability", "independent physical source or R-to-E cross-check",
                                             "training sample construction and split"]},
              "regression_reference": {"path": str(args.three_loop_reference.resolve()), "sha256": PREVIOUS_DATA_SHA},
              "selected_definitions": locations,
              "outputs": {DATA_NAME: {"sha256": sha256(args.output / DATA_NAME),
                                      "bytes": (args.output / DATA_NAME).stat().st_size},
                          "dependency_graph.json": {"sha256": sha256(args.output / "dependency_graph.json")}},
              "implementation_sha256": sha256(Path(__file__)), "three_loop_helper_sha256": PREVIOUS_SCRIPT_SHA,
              "python": platform.python_version(), "peak_rss_bytes": peak_rss_bytes(),
              "elapsed_seconds": round(time.monotonic() - started, 3),
              "complete_sparse_symbol_in_native_alphabet": True, "ready_for_training": False}
    write_json(args.output / "extraction_manifest.json", report)
    print(json.dumps({key: report[key] for key in ("rows", "nonzero_rows_by_y_count", "denominator_histogram",
                      "minimum_coefficient", "maximum_coefficient", "peak_rss_bytes", "elapsed_seconds")}))
    print(f"Wrote {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
