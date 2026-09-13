#!/usr/bin/env python3
"""Extract only the native-alphabet weight-6 symbol from one pinned release.

Standard library only. No Wolfram evaluator, eval, symbolic integration, target
alphabet conversion, integer-label scaling, training samples, or training.
"""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import re
import time
import zipfile


ARCHIVE_SHA = "00c26ece7c1301048ff8121217754024313528549311a10774180377db7d1d00"
MEMBER_SHA = "40f690428de310fd261c83d0ea740cde99c6c45d4845c17c1a9ac06991505adc"
MEMBER_NAME = "SixGluonAmpsAndCops.m"
MEMBER_BYTES = 82_917_570
LETTERS = ("a", "b", "c", "mu", "mv", "mw", "yu", "yv", "yw")
DIMENSIONS = {("YE", 1): 3, ("YE", 2): 6, ("YE", 3): 12,
              ("YO", 3): 1, ("YE", 4): 25, ("YO", 4): 2,
              ("YE", 5): 48, ("YO", 5): 6, ("YE", 6): 92}
CONSTANTS = {("YE", 4, 25): ("Zeta", 4), ("YE", 6, 92): ("Zeta", 6)}
MAX_MONOMIALS = 2048
MAX_CACHED_WORDS = 2_000_000
MAX_SECONDS = 180


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# A polynomial is {tuple_of_whitelisted_atoms: Fraction}. It is used only to
# distribute parentheses in the small source definitions, not to expand symbols.
def clean(poly):
    result = {key: value for key, value in poly.items() if value}
    require(len(result) <= MAX_MONOMIALS, "Polynomial size limit exceeded")
    require(all(max(abs(v.numerator).bit_length(), v.denominator.bit_length())
                <= 4096 for v in result.values()), "Coefficient size limit")
    return result


def add(left, right, factor=1):
    result = dict(left)
    for key, value in right.items():
        result[key] = result.get(key, Fraction(0)) + factor * value
    return clean(result)


def multiply(left, right):
    require(len(left) * len(right) <= 100_000, "Product size limit")
    result = {}
    for a, va in left.items():
        for b, vb in right.items():
            key = tuple(sorted(a + b))
            require(len(key) <= 4, "Monomial degree limit")
            result[key] = result.get(key, Fraction(0)) + va * vb
    return clean(result)


class Parser:
    """Strict recursive descent for integer arithmetic and g/YE/YO/Zeta atoms."""

    token = re.compile(r"\d+|[A-Za-z][A-Za-z0-9]*|[+*/(),\[\]-]")

    def __init__(self, text):
        require(len(text) <= 100_000, "Expression byte limit")
        self.tokens = []
        at = 0
        while at < len(text):
            if text[at].isspace():
                at += 1
                continue
            match = self.token.match(text, at)
            require(match is not None, f"Unsupported syntax at offset {at}")
            self.tokens.append(match.group())
            at = match.end()
        require(len(self.tokens) <= 40_000, "Token count limit")
        self.at = 0

    def peek(self):
        return self.tokens[self.at] if self.at < len(self.tokens) else None

    def take(self, token=None):
        actual = self.peek()
        require(actual is not None and (token is None or token == actual),
                f"Expected {token}, got {actual}")
        self.at += 1
        return actual

    def parse(self):
        result = self.expression(0)
        require(self.peek() is None, "Unconsumed expression tokens")
        return result

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
                require(set(right) == {()} and right[()] != 0,
                        "Only division by a nonzero rational scalar is allowed")
                result = clean({key: value / right[()] for key, value in result.items()})
        return result

    def unary(self, depth):
        require(depth < 64, "Expression nesting limit")
        if self.peek() in ("+", "-"):
            sign = self.take()
            result = self.unary(depth + 1)
            return {key: value if sign == "+" else -value for key, value in result.items()}
        if self.peek() == "(":
            self.take("(")
            result = self.expression(depth + 1)
            self.take(")")
            return result
        token = self.take()
        if token.isdecimal():
            return clean({(): Fraction(int(token))})
        require(token in ("g", "YE", "YO", "Zeta"), f"Unknown atom {token}")
        self.take("[")
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
    names = {f"e{loop}uvw" for loop in (1, 2, 3)}
    names.update(f"YE111{weight}" for weight in range(1, 7))
    for family, weight in DIMENSIONS:
        names.update(f"M{family}{weight}[{letter}]" for letter in LETTERS)
    return names


def read_definitions(archive):
    require(archive.stat().st_size == 21_944_268, "Archive size mismatch")
    require(sha256(archive) == ARCHIVE_SHA, "Archive SHA-256 mismatch")
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
        require(package.namelist() == [MEMBER_NAME], "Unexpected ZIP members")
        info = package.getinfo(MEMBER_NAME)
        require(info.file_size == MEMBER_BYTES, "Member size mismatch")
        require(not info.flag_bits & 1, "Encrypted member is unsupported")
        with package.open(info) as stream:
            for number, raw in enumerate(stream, 1):
                size += len(raw)
                require(size <= MEMBER_BYTES and len(raw) <= 16_384, "Stream size limit")
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
                    # Selected definitions contain arithmetic only, no comments.
                    require(not line.lstrip().startswith("(*"), "Unexpected definition boundary")
                    chunks.append(line)
                    if line.strip():
                        last = number
                    require(sum(map(len, chunks)) <= 100_000, "Definition size limit")
            finish()
    require(size == MEMBER_BYTES and h.hexdigest() == MEMBER_SHA, "Member SHA-256 mismatch")
    require(set(definitions) == wanted, f"Missing definitions: {wanted - definitions.keys()}")
    return definitions, locations


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
                require(len(selectors) == 1 and 1 <= selectors[0][1] <= count,
                        "Expected exactly one valid selector")
                require(len(monomial) == (1 if weight == 1 else 2), "Invalid table monomial")
                require(len(children) == (0 if weight == 1 else 1), "Invalid child count")
                child = children[0] if children else None
                require(child is None or (child in nodes and child[1] == weight - 1),
                        "Missing child or non-decreasing dependency")
                require(child is None or (child[0] == family) == (letter_id < 6),
                        "Coproduct parity mismatch")
                require(coefficient.denominator == 1, "Pinned table has noninteger transition")
                node = (family, weight, selectors[0][1])
                row = transitions[node].setdefault(letter_id, {})
                row[child] = row.get(child, 0) + int(coefficient)
    for node, table in transitions.items():
        for letter in list(table):
            table[letter] = {child: value for child, value in table[letter].items() if value}
            if not table[letter]:
                del table[letter]
    require({node for node, table in transitions.items() if not table} == set(CONSTANTS),
            "Unexpected derivative-free node")
    for node, zeta in CONSTANTS.items():
        value = coefficient_of_selector(definitions[f"YE111{node[1]}"], node[2])
        require(value == {(zeta,): Fraction(1)}, "Constant boundary mismatch")
    roots = {}
    for loop in (1, 2, 3):
        root = {}
        for monomial, value in definitions[f"e{loop}uvw"].items():
            require(len(monomial) == 1 and monomial[0] in nodes,
                    "Invalid amplitude basis expression")
            node = monomial[0]
            require(node[:2] == ("YE", 2 * loop), "Wrong amplitude parity or weight")
            root[node] = value
        roots[loop] = root
    require(len(roots[3]) == 64, "Three-loop root reference count mismatch")
    return transitions, roots


def coefficient_of_selector(poly, index):
    result = {}
    for monomial, value in poly.items():
        selectors = [atom for atom in monomial if atom[0] == "g"]
        require(len(selectors) == 1, "Invalid base-point selector expression")
        require(all(atom[0] in ("g", "Zeta") for atom in monomial), "Invalid base-point atom")
        if selectors[0] == ("g", index):
            key = tuple(atom for atom in monomial if atom[0] != "g")
            result[key] = result.get(key, Fraction(0)) + value
    return clean(result)


def base_point_checks(definitions, roots):
    expected = {1: {}, 2: {(("Zeta", 4),): Fraction(-10)},
                3: {(("Zeta", 6),): Fraction(413, 3)}}
    for loop, root in roots.items():
        result = {}
        for node, coefficient in root.items():
            result = add(result, coefficient_of_selector(definitions[f"YE111{2*loop}"], node[2]),
                         coefficient)
        require(result == expected[loop], f"Base-point check failed at loop {loop}")


class SymbolExtractor:
    def __init__(self, transitions):
        self.transitions = transitions
        self.started = time.monotonic()
        self.cached_words = 0

    @lru_cache(maxsize=None)
    def basis_symbol(self, node):
        require(time.monotonic() - self.started < MAX_SECONDS, "Extraction time limit")
        if node is None:
            return {b"": 1}
        require(node in self.transitions, "Unknown basis node")
        result = {}
        for letter, row in self.transitions[node].items():
            for child, scalar in row.items():
                for prefix, coefficient in self.basis_symbol(child).items():
                    word = prefix + bytes([letter])
                    result[word] = result.get(word, 0) + scalar * coefficient
        result = {word: coefficient for word, coefficient in result.items() if coefficient}
        require(all(len(word) == node[1] for word in result), "Basis word length mismatch")
        self.cached_words += len(result)
        require(self.cached_words <= MAX_CACHED_WORDS, "Cached symbol size limit")
        return result

    def amplitude(self, root):
        denominator = math.lcm(*(value.denominator for value in root.values()))
        result = {}
        for node, value in root.items():
            scalar = value.numerator * (denominator // value.denominator)
            for word, coefficient in self.basis_symbol(node).items():
                result[word] = result.get(word, 0) + scalar * coefficient
        return {word: Fraction(value, denominator) for word, value in result.items() if value}

    @lru_cache(maxsize=100_000)
    def query_basis(self, node, word):
        """Independent fixed-word recursion: never consult the expanded dictionaries."""
        if node is None:
            return int(not word)
        require(node in self.transitions, "Unknown query node")
        if len(word) != node[1]:
            return 0
        return sum(scalar * self.query_basis(child, word[:-1])
                   for child, scalar in self.transitions[node].get(word[-1], {}).items())

    def query(self, root, word):
        return sum((scalar * self.query_basis(node, word) for node, scalar in root.items()), Fraction(0))


def check_symbols(extractor, roots, symbols):
    expected_one_loop = {bytes(pair): Fraction(-1, 2)
                         for pair in ((1, 3), (2, 3), (0, 4), (2, 4), (0, 5), (1, 5))}
    require(symbols[1] == expected_one_loop, "One-loop normalization/order check failed")
    for loop, symbol in symbols.items():
        require(bool(symbol), "Empty amplitude symbol")
        for word, value in symbol.items():
            require(len(word) == 2 * loop and value != 0, "Invalid symbol row")
            require(word[0] < 3, "First entry is not a/b/c")
            require(word[1] < 6, "Second entry contains y")
            require(word[-1] >= 3, "MHV final entry contains a/b/c")
            require(sum(letter >= 6 for letter in word) % 2 == 0, "Odd y parity")
    rng = random.Random(20260913)
    nonzero = rng.sample(sorted(symbols[3]), min(512, len(symbols[3])))
    zero = set()
    while len(zero) < 512:
        # Restrict to elementary allowed first/final entries; this includes
        # nontrivial absent words, not just obviously forbidden first entries.
        word = bytes([rng.randrange(3), rng.randrange(6), *[rng.randrange(9) for _ in range(3)],
                      rng.randrange(3, 9)])
        if word not in symbols[3]:
            zero.add(word)
    for word in [*nonzero, *sorted(zero)]:
        require(extractor.query(roots[3], word) == symbols[3].get(word, Fraction(0)),
                "Fixed-word recursion disagrees with expansion")
    return {"one_loop_exact_reference": True, "base_point_loops_1_2_3": True,
            "first_entry_all_rows": True, "second_entry_no_y_all_rows": True,
            "mhv_final_entry_all_rows": True, "even_y_parity_all_rows": True,
            "fixed_word_queries_nonzero": len(nonzero), "fixed_word_queries_zero": len(zero),
            "query_seed": 20260913,
            "not_yet_checked": ["full integrability", "dihedral symmetry",
                                "independent release or R-to-E cross-check",
                                "target u/1-u alphabet conversion", "training samples"]}


def node_name(node):
    return "1" if node is None else f"{node[0]}[{node[1]},{node[2]}]"


def dependencies(transitions, root):
    reached = set()

    def visit(node):
        if node is None or node in reached:
            return
        reached.add(node)
        for row in transitions[node].values():
            for child in row:
                visit(child)

    for node in root:
        visit(node)
    rows = []
    for node in sorted(reached, key=lambda n: (n[1], n[0], n[2])):
        rows.append({"node": node_name(node), "weight": node[1],
                     "derivative_free_constant": node in CONSTANTS,
                     "root_coefficient": str(root[node]) if node in root else None,
                     "components": {LETTERS[letter]: {node_name(child): str(value)
                                    for child, value in row.items()}
                                    for letter, row in transitions[node].items()}})
    return rows


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    require(not args.output.exists(), "Output directory exists; refusing overwrite")
    started = time.monotonic()
    print("Reading pinned ZIP member; selecting 90 small definitions.", flush=True)
    definitions, locations = read_definitions(args.source)
    transitions, roots = compile_tables(definitions)
    base_point_checks(definitions, roots)
    extractor = SymbolExtractor(transitions)
    symbols = {}
    for loop in (1, 2, 3):
        symbols[loop] = extractor.amplitude(roots[loop])
        print(f"Loop {loop}: {len(symbols[loop]):,} nonzero native-alphabet words.", flush=True)
    checks = check_symbols(extractor, roots, symbols)
    graph = dependencies(transitions, roots[3])
    symbol = symbols[3]
    histogram = Counter(str(value) for value in symbol.values())
    y_counts = Counter(str(sum(letter >= 6 for letter in word)) for word in symbol)
    denominator_counts = Counter(str(value.denominator) for value in symbol.values())
    output_name = "three_loop_mhv_symbol_native.jsonl.gz"
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / output_name).open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            for word, value in sorted(symbol.items()):
                row = {"word": [LETTERS[i] for i in word],
                       "numerator": str(value.numerator), "denominator": str(value.denominator)}
                stream.write((json.dumps(row, separators=(",", ":")) + "\n").encode("ascii"))
    write_json(args.output / "dependency_graph.json", graph)
    report = {"schema_version": 1, "stage": "step2_native_symbol_extraction",
              "source": {"path": str(args.source.resolve()), "archive_sha256": ARCHIVE_SHA,
                         "member": MEMBER_NAME, "member_bytes": MEMBER_BYTES, "member_sha256": MEMBER_SHA,
                         "paper": "https://arxiv.org/abs/1903.10890", "root": "e3uvw"},
              "normalization": {"source": "2019 cosmic-normalized MHV", "target_symbol": "ordinary BDS-like MHV",
                                "relation_at_three_loops": "E_BDS_like^(3) = e3uvw + 8*zeta(3)^2",
                                "coupling": "g^2 = N*g_YM^2/(16*pi^2)",
                                "integer_label_scaling_applied": False},
              "alphabet": list(LETTERS), "word_order": "first entry to last entry",
              "coefficient_format": "exact reduced rational numerator/denominator strings",
              "loop": 3, "weight": 6, "rows": len(symbol),
              "nonzero_rows_by_y_count": dict(sorted(y_counts.items())),
              "denominator_histogram": dict(sorted(denominator_counts.items())),
              "coefficient_histogram": dict(sorted(histogram.items(), key=lambda item: Fraction(item[0]))),
              "minimum_coefficient": str(min(symbol.values())), "maximum_coefficient": str(max(symbol.values())),
              "dependency_nodes": len(graph), "cached_basis_word_entries": extractor.cached_words,
              "selected_definitions": locations, "checks": checks,
              "outputs": {output_name: {"sha256": sha256(args.output / output_name),
                                        "bytes": (args.output / output_name).stat().st_size},
                          "dependency_graph.json": {"sha256": sha256(args.output / "dependency_graph.json")}},
              "implementation_sha256": sha256(Path(__file__)), "python": platform.python_version(),
              "elapsed_seconds": round(time.monotonic() - started, 3),
              "complete_sparse_symbol_in_native_alphabet": True,
              "ready_for_training": False}
    write_json(args.output / "extraction_manifest.json", report)
    print(json.dumps({key: report[key] for key in ("rows", "nonzero_rows_by_y_count",
                      "denominator_histogram", "dependency_nodes", "elapsed_seconds")}, ensure_ascii=False), flush=True)
    print(f"Wrote {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
