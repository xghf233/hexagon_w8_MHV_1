#!/usr/bin/env python3
"""Budget-gated, disk-backed exact five-loop native Hexagon MHV extraction.

Whitelisted arithmetic only. No eval, Wolfram execution, network, alphabet
conversion, integer-label scaling, sample generation or training. Existing
three/four-loop source and artifacts are immutable regression references.
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
import sqlite3
import struct
import subprocess
import tempfile
import time
import zipfile

import extract_four_loop_symbol as four

require, sha256, write_json = four.require, four.sha256, four.write_json
LETTERS, ALIASES = four.LETTERS, four.ALIASES
DIMENSIONS = {**four.DIMENSIONS, ("YO", 8): 59, ("YE", 9): 559,
              ("YO", 9): 120, ("YE", 10): 991}
CONSTANTS = {**four.CONSTANTS, ("YE", 10, 991): ("Zeta", 10)}
FOUR_SCRIPT_SHA = "214ad83d6c8628deba06e20503afa6c90c5f422074051084a9182a124b09c644"
FOUR_DATA_SHA = "5eaa2ad35811e8a093540115ae56cba9dda84f118447bdf34212f820dac5b694"
MAX_RSS = 768 * 1024**2
MAX_DISK = 4 * 1024**3
MAX_ROWS = 30_000_000
MAX_EXPR_CHARS = 2_000_000
MAX_POLY = 160_000
DATA_NAME = "five_loop_mhv_symbol_native.jsonl.gz"


class Budget:
    def __init__(self, seconds):
        self.started = time.monotonic()
        self.seconds = seconds

    def elapsed(self):
        return time.monotonic() - self.started

    def check(self):
        require(self.elapsed() < self.seconds, "Wall-clock safety budget exceeded")
        require(four.peak_rss_bytes() < MAX_RSS, "Peak RSS safety budget exceeded")


class Parser(four.Parser):
    """Lazy tokens and in-place sums avoid the old parser's quadratic copying."""

    scan = re.compile(r"\s*(\d+|[A-Za-z][A-Za-z0-9]*|[+*/(),\[\]{}^\-])")

    def __init__(self, text):
        require(len(text) <= MAX_EXPR_CHARS, "Expression size limit")
        self.text, self.position, self.current, self.end, self.count = text, 0, None, 0, 0
        self.atoms = {}

    def peek(self):
        if self.current is None:
            if not self.text[self.position:self.position + 1]:
                return None
            match = self.scan.match(self.text, self.position)
            if match is None and not self.text[self.position:].strip():
                self.position = len(self.text)
                return None
            require(match is not None, f"Unknown syntax at {self.position}")
            self.current, self.end = match[1], match.end()
        return self.current

    def take(self, token=None):
        actual = self.peek()
        require(actual is not None and (token is None or actual == token), f"Expected {token}, got {actual}")
        self.position, self.current = self.end, None
        self.count += 1
        require(self.count <= 1_200_000, "Token count limit")
        return actual

    def expression(self, depth):
        result = self.product(depth)
        while self.peek() in ("+", "-"):
            sign = 1 if self.take() == "+" else -1
            right = self.product(depth)
            for key, value in right.items():
                result[key] = result.get(key, Fraction(0)) + sign * value
                if not result[key]:
                    del result[key]
            require(len(result) <= MAX_POLY, "Polynomial size limit")
        require(all(max(abs(v.numerator).bit_length(), v.denominator.bit_length()) <= 4096
                    for v in result.values()), "Coefficient bit limit")
        return result

    def atom(self, depth):
        if self.peek() == "(":
            self.take("(")
            value = self.expression(depth + 1)
            self.take(")")
            return value
        token = self.take()
        if token.isdecimal():
            value = int(token)
            return {(): Fraction(value)} if value else {}
        require(token in ("g", "YE", "YO", "Zeta", "MZV"), f"Unknown atom {token}")
        self.take("[")
        if token == "MZV":
            self.take("{")
            first = self.take()
            require(first in ("5", "7"), "Unsupported MZV index")
            for part in (",", "3", "}", "]"):
                self.take(part)
            atom = ("MZV", int(first), 3)
        else:
            indices = []
            for index in range(2 if token in ("YE", "YO") else 1):
                if index:
                    self.take(",")
                value = self.take()
                require(value.isdecimal() and 0 < int(value) <= 1000, "Invalid atom index")
                indices.append(int(value))
            self.take("]")
            atom = (token, *indices)
        return {(self.atoms.setdefault(atom, atom),): Fraction(1)}

    def product(self, depth):
        result = self.unary(depth)
        while self.peek() in ("*", "/"):
            operator = self.take()
            right = self.unary(depth)
            if operator == "/":
                require(set(right) == {()} and right[()] != 0, "Nonzero scalar divisor required")
                result = {key: value / right[()] for key, value in result.items()}
                continue
            require(len(result) * len(right) <= MAX_POLY, "Product size limit")
            combined = {}
            for a, va in result.items():
                for b, vb in right.items():
                    key = tuple(sorted(a + b))
                    require(len(key) <= 4, "Monomial degree limit")
                    combined[key] = combined.get(key, Fraction(0)) + va * vb
            result = {key: value for key, value in combined.items() if value}
        return result


def read_source(path, budget):
    require(sha256(Path(four.__file__)) == FOUR_SCRIPT_SHA, "Four-loop helper changed")
    require(sha256(Path(four.previous.__file__)) == four.PREVIOUS_SCRIPT_SHA, "Three-loop helper changed")
    require(path.stat().st_size == 21_944_268 and sha256(path) == four.previous.ARCHIVE_SHA, "Source archive mismatch")
    nodes = {(f, w, n): (f, w, n) for (f, w), count in DIMENSIONS.items() for n in range(1, count + 1)}
    transitions = {node: {} for node in nodes}
    wanted = {f"M{f}{w}[{x}]" for f, w in DIMENSIONS for x in LETTERS}
    wanted |= {f"e{loop}uvw" for loop in range(1, 6)} | {f"YE111{w}" for w in range(1, 11)}
    require(len(wanted) == 168, "Wrong source selection")
    roots, bases, locations = {}, {}, {}
    header = re.compile(r"^([A-Za-z][A-Za-z0-9]*(?:\[[^\]\r\n]+\])?)\s*=\s*(.*)")
    active, chunks, first, last = None, [], 0, 0

    def finish():
        if active is None:
            return
        budget.check()
        require(active not in locations, "Duplicate definition")
        body = "".join(chunks).strip()
        poly = Parser(body).parse()
        locations[active] = {"first_line": first, "last_line": last,
                             "rhs_sha256": hashlib.sha256(body.encode()).hexdigest(), "monomials": len(poly)}
        match = re.fullmatch(r"M(Y[EO])(\d+)\[([a-z]+)\]", active)
        if match:
            family, weight, letter = match[1], int(match[2]), LETTERS.index(match[3])
            for atoms, value in poly.items():
                selectors = [a for a in atoms if a[0] == "g"]
                children = [a for a in atoms if a[0] in ("YE", "YO")]
                require(len(selectors) == 1 and 1 <= selectors[0][1] <= DIMENSIONS[family, weight], "Invalid selector")
                require(len(atoms) == (1 if weight == 1 else 2) and len(children) == (0 if weight == 1 else 1), "Bad monomial")
                child = nodes.get(children[0]) if children else None
                require(not children or (child is not None and child[1] == weight - 1), "Missing/nonlowering reference")
                require(child is None or (child[0] == family) == (letter < 6), "Parity mismatch")
                parent = nodes[family, weight, selectors[0][1]]
                row = transitions[parent].setdefault(letter, {})
                row[child] = row.get(child, Fraction(0)) + value
        elif active.startswith("YE111"):
            bases[active] = poly
        else:
            loop = int(active[1])
            require(all(len(atoms) == 1 and atoms[0] in nodes and atoms[0][:2] == ("YE", 2 * loop) for atoms in poly), "Bad root")
            roots[loop] = {nodes[atoms[0]]: c for atoms, c in poly.items()}
        if active in ("MYE8[yw]", "MYE9[yw]", "MYE10[yw]"):
            print(f"Parsed through {active}; elapsed {budget.elapsed():.1f}s.", flush=True)

    h, size = hashlib.sha256(), 0
    with zipfile.ZipFile(path) as package:
        require(package.namelist() == [four.previous.MEMBER_NAME], "Unexpected ZIP member")
        info = package.getinfo(four.previous.MEMBER_NAME)
        require(info.file_size == four.previous.MEMBER_BYTES and not info.flag_bits & 1, "Invalid ZIP metadata")
        with package.open(info) as stream:
            for number, raw in enumerate(stream, 1):
                h.update(raw)
                size += len(raw)
                require(size <= four.previous.MEMBER_BYTES and len(raw) <= 16384, "Source stream limit")
                line = raw.decode("ascii")
                match = header.match(line)
                if match:
                    finish()
                    name = match[1].replace(" ", "")
                    active = name if name in wanted else None
                    chunks = [match[2] + "\n"] if active else []
                    first = last = number
                elif active:
                    require(not line.lstrip().startswith("(*"), "Unexpected source boundary")
                    chunks.append(line)
                    if line.strip():
                        last = number
            finish()
    require(size == four.previous.MEMBER_BYTES and h.hexdigest() == four.previous.MEMBER_SHA, "Member checksum mismatch")
    require(set(locations) == wanted, "Missing definitions")
    for table in transitions.values():
        for letter in list(table):
            table[letter] = {child: c for child, c in table[letter].items() if c}
            if not table[letter]:
                del table[letter]
    require({n for n, table in transitions.items() if not table} == set(CONSTANTS), "Unexpected derivative-free node")
    for node, zeta in CONSTANTS.items():
        require(four.coefficient_of_selector(bases[f"YE111{node[1]}"], node[2]) == {(zeta,): Fraction(1)}, "Constant mismatch")
    require(len(roots[3]) == 64 and len(roots[4]) == 249 and len(roots[5]) == 944, "Root count mismatch")
    four.base_point_checks(bases, {loop: roots[loop] for loop in range(1, 5)})
    observed = {}
    for node, coefficient in roots[5].items():
        observed = four.add(observed, four.coefficient_of_selector(bases['YE11110'], node[2]), coefficient)
    expected = Parser('379957*Zeta[10]/15-12*(4*Zeta[2]*MZV[{5,3}]+25*Zeta[5]^2)-96*(2*MZV[{7,3}]+28*Zeta[3]*Zeta[7]+11*Zeta[5]^2-4*Zeta[2]*Zeta[3]*Zeta[5]-6*Zeta[4]*Zeta[3]^2)').parse()
    require(observed == expected, "Five-loop base-point mismatch")
    scales = integerize(transitions)
    print('Exact transition denominator clearing: '+json.dumps(scales['per_weight_lcm']),flush=True)
    return transitions, roots, locations, scales


def integerize(transitions):
    """Clear denominators uniformly at each weight, never round coefficients.

    Every full length-w path acquires the same product of layer factors. Divide
    the final symbol by that product; pure-constant/short paths remain zero.
    """
    per_weight, histogram = {}, Counter()
    for node, table in transitions.items():
        per_weight.setdefault(node[1],1)
        for row in table.values():
            for value in row.values():
                value=Fraction(value)
                per_weight[node[1]]=math.lcm(per_weight[node[1]],value.denominator)
                histogram[value.denominator]+=1
    for node, table in transitions.items():
        factor=per_weight[node[1]]
        for row in table.values():
            for child,value in row.items():
                value=Fraction(value)
                row[child]=value.numerator*(factor//value.denominator)
    cumulative, factor = {}, 1
    for weight in sorted(per_weight):
        factor*=per_weight[weight]
        cumulative[weight]=factor
    return {'per_weight_lcm':per_weight,'cumulative_symbol_scale':cumulative,
            'original_transition_denominator_histogram':dict(sorted(histogram.items()))}


def scaled_root(root, symbol_scale=1):
    denominator = math.lcm(*(c.denominator for c in root.values()))
    return {node: c.numerator * (denominator // c.denominator) for node, c in root.items()}, denominator*symbol_scale


def word_id(word):
    result = 0
    for letter in word:
        result = result * 9 + letter
    return result


def decode_word(code, weight=10):
    digits = bytearray(weight)
    for i in range(weight - 1, -1, -1):
        code, digits[i] = divmod(code, 9)
    require(code == 0, "Word code overflow")
    return bytes(digits)


class Backward:
    def __init__(self, transitions, budget):
        self.transitions, self.budget = transitions, budget
        self.calls = 0

    @lru_cache(maxsize=None)
    def low(self, node):
        if node is None:
            return {b'': 1}
        require(node[1] <= 4, "Low cache weight limit")
        result = {}
        for letter, row in self.transitions[node].items():
            for child, scalar in row.items():
                for prefix, coefficient in self.low(child).items():
                    word = prefix + bytes([letter])
                    result[word] = result.get(word, 0) + scalar * coefficient
        return {word: c for word, c in result.items() if c}

    def contract(self, state):
        parts = {}
        for node, scalar in state.items():
            for letter, row in self.transitions[node].items():
                part = parts.setdefault(letter, {})
                for child, c in row.items():
                    part[child] = part.get(child, 0) + scalar * c
        return {letter: {n: c for n, c in part.items() if c} for letter, part in parts.items()
                if any(part.values())}

    def parts(self, state, remaining, depth, suffix=b''):
        if depth == 0:
            yield suffix, state, remaining
            return
        for letter, part in sorted(self.contract(state).items()):
            yield from self.parts(part, remaining - 1, depth - 1, bytes([letter]) + suffix)

    def records(self, state, remaining, suffix=b''):
        self.calls += 1
        if self.calls % 1024 == 0:
            self.budget.check()
        if remaining <= 4:
            block = {}
            for node, scalar in state.items():
                for prefix, c in self.low(node).items():
                    block[prefix] = block.get(prefix, 0) + scalar * c
            power, tail = 9**len(suffix), word_id(suffix)
            for prefix, c in block.items():
                if c:
                    require(abs(c) < 2**63, "Exact SQLite integer overflow")
                    yield word_id(prefix) * power + tail, c
            return
        for letter, part in sorted(self.contract(state).items()):
            yield from self.records(part, remaining - 1, bytes([letter]) + suffix)


class Forward:
    def __init__(self, transitions, root, budget):
        self.budget, self.inverse, self.final, self.calls, self.zeros = budget, {}, {}, 0, 0
        self.weight = next(iter(root))[1]
        for parent, components in transitions.items():
            if parent[1] >= self.weight:
                continue
            for letter, row in components.items():
                for child, c in row.items():
                    self.inverse.setdefault((parent[1], child), []).append((letter, parent, c))
        for parent, scalar in root.items():
            for letter, row in transitions[parent].items():
                dest = self.final.setdefault(letter, {})
                for child, c in row.items():
                    dest[child] = dest.get(child, 0) + scalar * c
        self.final = {letter: {n: c for n, c in row.items() if c} for letter, row in self.final.items() if any(row.values())}

    def advance(self, vector, depth):
        parts = {}
        for child, scalar in vector.items():
            for letter, parent, c in self.inverse.get((depth + 1, child), ()):
                part = parts.setdefault(letter, {})
                part[parent] = part.get(parent, 0) + scalar * c
        return {letter: {n: c for n, c in part.items() if c} for letter, part in parts.items() if any(part.values())}

    def prefixes(self, depth=3):
        states = [(0, 0, {None: 1})]
        for _ in range(depth):
            states = [(code * 9 + letter, d + 1, part) for code, d, vector in states
                      for letter, part in sorted(self.advance(vector, d).items())]
        return states

    def records(self, code=0, depth=0, vector=None):
        vector = {None: 1} if vector is None else vector
        self.calls += 1
        if self.calls % 4096 == 0:
            self.budget.check()
        if depth == self.weight - 1:
            for letter, row in sorted(self.final.items()):
                value = sum(c * row.get(node, 0) for node, c in vector.items())
                if value:
                    yield code * 9 + letter, value
                else:
                    self.zeros += 1
            return
        for letter, part in sorted(self.advance(vector, depth).items()):
            yield from self.records(code * 9 + letter, depth + 1, part)


def insert_records(connection, records, budget):
    batch, count = [], 0
    for record in records:
        batch.append(record)
        if len(batch) >= 4096:
            connection.executemany('INSERT INTO coefficients VALUES (?,?)', batch)
            count += len(batch)
            batch.clear()
            budget.check()
            require(count <= MAX_ROWS, "Row safety limit")
    if batch:
        connection.executemany('INSERT INTO coefficients VALUES (?,?)', batch)
        count += len(batch)
    connection.commit()
    return count


def connect_new(path):
    require(not path.exists(), "Database already exists")
    connection = sqlite3.connect(path)
    connection.execute('PRAGMA cache_size=-16384')
    connection.execute('PRAGMA temp_store=FILE')
    connection.execute('PRAGMA synchronous=NORMAL')
    connection.execute('CREATE TABLE coefficients(word_id INTEGER PRIMARY KEY, numerator INTEGER NOT NULL)')
    return connection


def regression(transitions, roots, references, budget, scales):
    engine = Backward(transitions, budget)
    for loop, reference in references.items():
        state, denominator = scaled_root(roots[loop],scales['cumulative_symbol_scale'][2*loop])
        actual = {decode_word(code, 2 * loop): Fraction(c, denominator)
                  for code, c in engine.records(state, 2 * loop)}
        require(actual == four.read_export(reference, 2 * loop), f'Loop {loop} regression mismatch')
        print(f'Full loop-{loop} regression passed: {len(actual):,} rows.', flush=True)


def native_setup(transitions,state,budget,directory):
    source=Path(__file__).with_name('verify_symbol_forward.cpp')
    executable=directory/'verify_forward'
    subprocess.run(['clang++','-std=c++20','-O3',str(source),'-o',str(executable)],check=True,
                   capture_output=True,text=True,timeout=60)
    budget.check()
    nodes=sorted(transitions,key=lambda n:(n[1],n[0],n[2]))
    ids={node:i+1 for i,node in enumerate(nodes)}
    ids[None]=0
    weight=next(iter(state))[1]
    bounds={None:1}
    for node in nodes:
        bounds[node]=sum(abs(c)*bounds[child] for row in transitions[node].values() for child,c in row.items())
    bound=max(max(bounds.values()),sum(abs(c)*bounds[node] for node,c in state.items()))
    require(bound<2**120,'Native signed-128 arithmetic bound exceeded')
    edges=[(ids[child],letter,ids[parent],c) for parent,components in transitions.items() if parent[1]<weight
           for letter,row in components.items() for child,c in row.items()]
    final={}
    for parent,scalar in state.items():
        for letter,row in transitions[parent].items():
            for child,c in row.items():
                key=(letter,ids[child]);final[key]=final.get(key,0)+scalar*c
    terms=[(letter,child,c) for (letter,child),c in sorted(final.items()) if c]
    require(all(abs(edge[-1])<2**63 for edge in edges) and all(abs(term[-1])<2**63 for term in terms),'Native edge range exceeded')
    graph=directory/'graph.txt'
    with graph.open('x') as output:
        output.write(f'HX5V1\n{weight} {len(ids)} {len(edges)} {len(terms)}\n')
        output.write(' '.join(str(n[1]) if n is not None else '0' for n in [None,*nodes])+'\n')
        for edge in edges: output.write(' '.join(map(str,edge))+'\n')
        for term in terms: output.write(' '.join(map(str,term))+'\n')
    return {'executable':executable,'graph':graph,'source_sha256':sha256(source),
            'proven_absolute_intermediate_bound':str(bound),'bound_bits':bound.bit_length()}


def native_run(native,budget,reference=None,prefixes=None):
    remaining=min(300,budget.seconds-budget.elapsed())
    require(remaining>1,'Insufficient verification budget')
    result=subprocess.run([str(native['executable']),str(native['graph']),str(reference) if reference else '-',
                           ','.join(map(str,prefixes)) if prefixes is not None else 'all',str(remaining)],
                          check=True,capture_output=True,text=True,timeout=remaining+1)
    budget.check()
    return json.loads(result.stdout)


def binary_reference(conn,path,rows,budget):
    buffer=bytearray()
    with path.open('xb') as output:
        output.write(b'HX5B001\n'+struct.pack('<Q',rows))
        for code,numerator in conn.execute('SELECT word_id,numerator FROM coefficients ORDER BY word_id'):
            buffer.extend(struct.pack('<Iq',code,numerator))
            if len(buffer)>=1024**2:
                output.write(buffer);buffer.clear();budget.check()
        if buffer: output.write(buffer)


def pilot(backward, forward, state, budget, native):
    started = time.monotonic()
    suffixes = list(backward.parts(state, 10, 3))
    rng = random.Random(20260916)
    selected = rng.sample(suffixes, min(12, len(suffixes)))
    with tempfile.TemporaryDirectory(prefix='hexagon5-pilot-') as directory:
        conn = connect_new(Path(directory) / 'pilot.sqlite')
        before = time.monotonic()
        rows = sum(insert_records(conn, backward.records(part, remaining, suffix), budget)
                   for suffix, part, remaining in selected)
        back_seconds = time.monotonic() - before
        conn.close()
    prefixes = forward.prefixes()
    chosen = rng.sample(prefixes, min(9, len(prefixes)))
    before = time.monotonic()
    native_result=native_run(native,budget,prefixes=[code for code,_,_ in chosen])
    forward_rows = native_result['rows']
    forward_seconds = time.monotonic() - before
    projected_rows = rows * len(suffixes) / len(selected)
    projected_forward_rows = forward_rows * len(prefixes) / len(chosen)
    projected_rows = max(projected_rows, projected_forward_rows)
    # Conservative allowance for full-size B-tree writes, export, readback,
    # independent forward comparison and sampled random queries.
    estimate = (2 * back_seconds * len(suffixes) / len(selected)
                + 2 * forward_seconds * len(prefixes) / len(chosen)
                + projected_rows / 80_000 + 15)
    result = {'seed': 20260916, 'suffix_partitions': len(suffixes), 'sampled_suffix_partitions': len(selected),
              'backward_sample_rows': rows, 'backward_sample_seconds': round(back_seconds, 3),
              'prefix_partitions': len(prefixes), 'sampled_prefix_partitions': len(chosen),
              'forward_sample_rows': forward_rows, 'forward_sample_seconds': round(forward_seconds, 3),
              'projected_rows_not_observed': round(projected_rows),
              'estimated_remaining_seconds_conservative': round(estimate, 1),
              'pilot_seconds': round(time.monotonic() - started, 3)}
    result['forward_implementation']='C++ exact signed-128 with Python-proven arithmetic bound'
    print('PILOT ' + json.dumps(result), flush=True)
    return result


def chunk_encodings():
    encoded, ys = [], bytearray()
    for code in range(9**5):
        word = decode_word(code, 5)
        encoded.append(','.join(json.dumps(LETTERS[i]) for i in word).encode('ascii'))
        ys.append(sum(i >= 6 for i in word))
    return encoded, ys


def export_data(connection, destination, denominator, budget):
    encoded, ys = chunk_encodings()
    histogram, y_counts, denominators = Counter(), Counter(), Counter()
    rows, buffer, digest = 0, bytearray(), hashlib.sha256()
    samples = []
    @lru_cache(maxsize=100_000)
    def coefficient(numerator):
        divisor = math.gcd(numerator, denominator)
        n, d = numerator // divisor, denominator // divisor
        return n, d, f'],"numerator":"{n}","denominator":"{d}"}}\n'.encode('ascii')
    with destination.open('xb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=1) as output:
            for code, numerator in connection.execute('SELECT word_id,numerator FROM coefficients ORDER BY word_id'):
                hi, lo = divmod(code, 9**5)
                require(hi < 9**5 and hi // 9**4 < 3 and (hi // 9**3) % 9 < 6 and lo % 9 >= 3, 'Entry condition failed')
                y_count = ys[hi] + ys[lo]
                require(y_count % 2 == 0 and numerator != 0, 'Parity/zero entry failed')
                n, d, ending = coefficient(numerator)
                line = b'{"word":[' + encoded[hi] + b',' + encoded[lo] + ending
                buffer.extend(line)
                histogram[numerator] += 1
                denominators[d] += 1
                y_counts[y_count] += 1
                rows += 1
                if rows % 4093 == 0 and len(samples) < 4096:
                    samples.append((code, numerator))
                if len(buffer) >= 1024**2:
                    output.write(buffer)
                    digest.update(buffer)
                    buffer.clear()
                    budget.check()
            if buffer:
                output.write(buffer)
                digest.update(buffer)
    # Independently parse every emitted JSON record and compare to the SQLite
    # oracle that was already checked by forward recursion.
    checked, read_digest = 0, hashlib.sha256()
    stored = iter(connection.execute('SELECT word_id,numerator FROM coefficients ORDER BY word_id'))
    with gzip.open(destination, 'rb') as stream:
        for line in stream:
            read_digest.update(line)
            row = json.loads(line)
            require(set(row) == {'word','numerator','denominator'}, 'JSON schema mismatch')
            require(len(row['word']) == 10, 'JSON word length mismatch')
            code = word_id(LETTERS.index(x) for x in row['word'])
            observed_n, observed_d = int(row['numerator']), int(row['denominator'])
            reference = next(stored, None)
            require(reference is not None and code == reference[0], 'JSON word/order mismatch')
            n, d, _ = coefficient(reference[1])
            require(row['numerator'] == str(n) and row['denominator'] == str(d) and observed_d > 0,
                    'JSON coefficient mismatch')
            checked += 1
            if checked % 16384 == 0:
                budget.check()
    require(checked == rows and next(stored, None) is None and read_digest.digest() == digest.digest(), 'Readback mismatch')
    def rational(n):
        return str(Fraction(n, denominator))
    return {'rows': rows, 'nonzero_rows_by_y_count': {str(k): v for k,v in sorted(y_counts.items())},
            'denominator_histogram': {str(k): v for k,v in sorted(denominators.items())},
            'coefficient_histogram': {rational(k): v for k,v in sorted(histogram.items())},
            'minimum_coefficient': rational(min(histogram)), 'maximum_coefficient': rational(max(histogram)),
            'denominator_lcm_observed_only': math.lcm(*denominators),
            'uncompressed_jsonl_sha256': digest.hexdigest(), 'json_records_read_back': checked}, samples


def sampled_checks(conn, transitions, state, samples, budget):
    # Exact sampled symmetry and a third, fixed-word scalar recursion. Full
    # table symmetry is deliberately not claimed for these sampled checks.
    for code, c in samples:
        word = decode_word(code)
        for permutation in ((1,2,0), (1,0,2)):
            mapped = word_id(3*(i//3)+permutation[i%3] for i in word)
            require(conn.execute('SELECT numerator FROM coefficients WHERE word_id=?',(mapped,)).fetchone() == (c,), 'Sampled symmetry mismatch')
    query = four.previous.SymbolExtractor(transitions)
    rng, count = random.Random(20260917), 0
    selected = rng.sample(samples, min(32, len(samples)))
    zeros = []
    while len(zeros) < 32:
        word = bytes([rng.randrange(3),rng.randrange(6),*[rng.randrange(9) for _ in range(7)],rng.randrange(3,9)])
        if sum(i>=6 for i in word)%2 == 0 and conn.execute('SELECT 1 FROM coefficients WHERE word_id=?',(word_id(word),)).fetchone() is None:
            zeros.append((word_id(word),0))
    for code, expected in selected + zeros:
        word = decode_word(code)
        actual = sum(c*query.query_basis(node,word) for node,c in state.items())
        require(actual == expected, 'Fixed-word scalar query mismatch')
        count += 1
        budget.check()
    return {'exact_symmetry_sample_words': len(samples), 'symmetry_generators_per_sample': 2,
            'scalar_queries_nonzero': len(selected), 'scalar_queries_zero_even_y': len(zeros), 'query_seed': 20260917}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source', type=Path, required=True)
    cli.add_argument('--three-loop-reference', type=Path, required=True)
    cli.add_argument('--four-loop-reference', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--max-seconds', type=int, default=480)
    cli.add_argument('--pilot-only', action='store_true')
    args = cli.parse_args()
    require(60 <= args.max_seconds <= 600, 'Runtime budget must be 60..600 seconds')
    require(not args.output.exists(), 'Output exists; refusing overwrite')
    budget = Budget(args.max_seconds)
    require(sha256(args.three_loop_reference) == four.PREVIOUS_DATA_SHA, 'Three-loop reference changed')
    require(sha256(args.four_loop_reference) == FOUR_DATA_SHA, 'Four-loop reference changed')
    transitions, roots, locations, scales = read_source(args.source, budget)
    regression(transitions, roots, {3:args.three_loop_reference,4:args.four_loop_reference}, budget,scales)
    state, denominator = scaled_root(roots[5],scales['cumulative_symbol_scale'][10])
    back, forward = Backward(transitions,budget), Forward(transitions,state,budget)
    scratch=tempfile.TemporaryDirectory(prefix='hexagon5-verify-')
    scratch_path=Path(scratch.name)
    native=native_setup(transitions,state,budget,scratch_path)
    # Check the new native verifier against the full, previously verified
    # weight-8 export before allowing five-loop production.
    four_state,four_denominator=scaled_root(roots[4],scales['cumulative_symbol_scale'][8])
    regression_dir=scratch_path/'regression'
    regression_dir.mkdir()
    native4=native_setup(transitions,four_state,budget,regression_dir)
    reference4=regression_dir/'reference.bin'
    prior=four.read_export(args.four_loop_reference,8)
    with reference4.open('xb') as stream:
        stream.write(b'HX5B001\n'+struct.pack('<Q',len(prior)))
        for word,value in sorted(prior.items()):
            numerator=value*four_denominator
            require(numerator.denominator==1,'Regression reference denominator mismatch')
            stream.write(struct.pack('<Iq',word_id(word),int(numerator)))
    native_regression=native_run(native4,budget,reference=reference4)
    require(native_regression['rows']==len(prior),'Native four-loop regression failed')
    del prior
    print('Native verifier full four-loop regression passed.',flush=True)
    preflight = pilot(back,forward,state,budget,native)
    if args.pilot_only:
        return
    require(preflight['estimated_remaining_seconds_conservative'] + budget.elapsed() < args.max_seconds,
            'Pilot predicts exceeding the minutes-scale budget; no production data written')
    print('Pilot fits budget. Starting disk-backed production extraction.', flush=True)
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'preflight.json',preflight)
    database = args.output/'coefficients.partial.sqlite'
    conn = connect_new(database)
    extraction_started, rows = time.monotonic(), 0
    for suffix, part, remaining in back.parts(state,10,1):
        n = insert_records(conn,back.records(part,remaining,suffix),budget)
        rows += n
        require(rows <= MAX_ROWS and database.stat().st_size <= MAX_DISK, 'Dataset safety limit')
        print(f'Final entry {LETTERS[suffix[0]]}: {n:,} rows; cumulative {rows:,}; elapsed {budget.elapsed():.1f}s.',flush=True)
    extraction_seconds = time.monotonic()-extraction_started
    print('Starting complete forward-propagation verification.',flush=True)
    verify_started = time.monotonic()
    reference=scratch_path/'reference.bin'
    binary_reference(conn,reference,rows,budget)
    native_result=native_run(native,budget,reference=reference)
    checked=native_result['rows']
    require(checked == rows, 'Forward row count mismatch')
    forward_seconds = time.monotonic()-verify_started
    print('Writing canonical JSONL and reading back every record.',flush=True)
    data_partial = args.output/(DATA_NAME+'.partial')
    stats,samples = export_data(conn,data_partial,denominator,budget)
    extra_checks = sampled_checks(conn,transitions,state,samples,budget)
    require(conn.execute('PRAGMA quick_check').fetchall() == [('ok',)],'SQLite integrity failure')
    conn.execute('CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
    metadata = {'schema_version':'1','loop':'5','weight':'10','alphabet':json.dumps(LETTERS),
                'training_aliases':json.dumps(ALIASES),'common_denominator':str(denominator),
                'word_id_encoding':'base9, ten digits, first entry most significant',
                'complete':'true','integer_label_scaling_applied':'false'}
    conn.executemany('INSERT INTO metadata VALUES (?,?)',metadata.items())
    conn.commit()
    conn.close()
    budget.check()
    require(not (args.output/'coefficients.sqlite').exists() and not (args.output/DATA_NAME).exists(),'Final output exists')
    database.rename(args.output/'coefficients.sqlite')
    data_partial.rename(args.output/DATA_NAME)
    report = {'schema_version':1,'stage':'five_loop_native_symbol_extraction','loop':5,'weight':10,**stats,
              'source':{'path':str(args.source.resolve()),'archive_sha256':four.previous.ARCHIVE_SHA,
                        'member':four.previous.MEMBER_NAME,'member_sha256':four.previous.MEMBER_SHA,
                        'root':'e5uvw','paper':'https://arxiv.org/abs/1903.10890'},
              'normalization':{'source':'2019 cosmic-normalized MHV','target_symbol':'ordinary BDS-like MHV',
                    'function_relation':'E_BDS_like^(5)=e5uvw+8*zeta(3)^2*e2uvw-160*zeta(3)*zeta(5)*e1uvw+1680*zeta(3)*zeta(7)+912*zeta(5)^2-32*zeta(4)*zeta(3)^2',
                    'full_weight_10_symbols_equal':True,'coupling':'g^2=N*g_YM^2/(16*pi^2)',
                    'integer_label_scaling_applied':False},
              'alphabet':list(LETTERS),'training_aliases':list(ALIASES),'alphabet_version':'hexagon_hat_native_v1',
              'alphabet_conversion_applied':False,'word_order':'first entry to last entry',
              'row_order':'lexicographic letter IDs 0..8','coefficient_format':'reduced rational strings',
              'sqlite_encoding':{'word_id':'base9 integer with fixed ten-digit decoding',
                                 'coefficient':'numerator/common_denominator','common_denominator':denominator},
              'internal_exact_denominator_clearing':scales,
              'checks':{'base_point_loops_1_through_5':True,'three_and_four_loop_full_regression':True,
                        'all_nonzero_rows_verified_by_forward_propagation':checked,'complete_nonzero_support_match':True,
                        'first_second_final_entry_and_even_y_all_rows':True,'json_gzip_all_rows_read_back':True,
                        'sqlite_quick_check':True,**extra_checks,
                        'not_yet_checked':['full integrability','exhaustive dihedral symmetry','independent physical release cross-check','training samples and splits']},
              'verification_scope':'Backward expansion, independent forward propagation, and scalar queries share parsed coproduct tables; not an independent physical source.',
              'forward_reachable_prefixes':native_result['reachable_prefixes'],'zero_final_projections':native_result['zero_final_projections'],
              'native_verifier':{'source_sha256':native['source_sha256'],'integer_arithmetic':'signed __int128, no floating point',
                                 'proven_absolute_intermediate_bound':native['proven_absolute_intermediate_bound'],
                                 'bound_bits':native['bound_bits'],'complete_support_match':native_result['complete_support_match'],
                                 'four_loop_regression_rows':native_regression['rows']},
              'timing_seconds':{'extraction_to_sqlite':round(extraction_seconds,3),'complete_forward_verification':round(forward_seconds,3),
                                'total_including_parse_regression_pilot_export_checks':round(budget.elapsed(),3)},
              'peak_rss_bytes':four.peak_rss_bytes(),'python':platform.python_version(),
              'limits':{'max_seconds':args.max_seconds,'max_peak_rss_bytes':MAX_RSS,'max_database_bytes':MAX_DISK,'max_rows':MAX_ROWS},
              'compression':{'format':'gzip','level':1,'mtime':0},'selected_definitions':locations,
              'implementation_sha256':sha256(Path(__file__)),'four_loop_helper_sha256':FOUR_SCRIPT_SHA,
              'three_loop_helper_sha256':four.PREVIOUS_SCRIPT_SHA,
              'regression_reference_hashes':{'three_loop':four.PREVIOUS_DATA_SHA,'four_loop':FOUR_DATA_SHA},
              'outputs':{name:{'sha256':sha256(args.output/name),'bytes':(args.output/name).stat().st_size}
                         for name in (DATA_NAME,'coefficients.sqlite','preflight.json')},
              'complete_sparse_symbol_in_native_alphabet':True,'ready_for_training':False}
    write_json(args.output/'extraction_manifest.json',report)
    scratch.cleanup()
    print('COMPLETE '+json.dumps({k:report[k] for k in ('rows','nonzero_rows_by_y_count','denominator_lcm_observed_only','peak_rss_bytes','timing_seconds')}),flush=True)


if __name__ == '__main__':
    main()
