"""Data loading, orbit computation, and audit for the five-loop symbol.

Loads the raw symbol file using the official AIAmplitudes convert(),
computes D3 orbits via get_dihedral_images(), and produces a
comprehensive audit report with mandatory assertions.
"""

import argparse
import hashlib
import os
from collections import Counter
from pathlib import Path

try:
    from aiamplitudes_common_public.file_readers import convert
except ModuleNotFoundError as exc:
    if exc.name != "aiamplitudes_common_public":
        raise
    convert = None
    _AIAMPLITUDES_IMPORT_ERROR = exc
else:
    _AIAMPLITUDES_IMPORT_ERROR = None

from .paths import resolve_data_path, resolve_upstream_root
from .symmetry import alphabet, get_orbit_id

# Expected values from the official metadata
EXPECTED_SAMPLES = 263880
EXPECTED_WORD_LENGTH = 10
EXPECTED_MD5 = "71bb1949a160a7501c4b942fc329b64f"
EXPECTED_SIZE = 7704854


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------
def _get_git_commit(repo_path: Path) -> str:
    """Return the HEAD commit SHA of a git repository."""
    import subprocess
    repo_path = Path(repo_path)
    if not repo_path.is_dir():
        return "unavailable"
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_path),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return "unavailable"
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_symbol(path: str | Path | None = None, loop: int = 5) -> dict[str, int]:
    """Load the symbol data using the official AIAmplitudes convert().

    Args:
        path: Path to the raw symbol file. If omitted, resolve it from
            ``NANOINFRA_SYMBOL_DATA_PATH`` and then the compatibility layout.
        loop: Loop order (must be 5 for our baseline).

    Returns:
        Dict mapping word string (e.g. 'ababccdeff') to integer coefficient.
    """
    if convert is None:
        raise ModuleNotFoundError(
            "The Symbol project requires the external "
            "'aiamplitudes_common_public' package. Add the official "
            "AIAmplitudes_common_public checkout to the remote environment's "
            "Python path before loading data."
        ) from _AIAMPLITUDES_IMPORT_ERROR

    path = str(resolve_data_path(path))

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Symbol file not found: {path}. Set data.path in the Hydra config "
            "or NANOINFRA_SYMBOL_DATA_PATH."
        )

    if loop != 5:
        raise ValueError(f"This project only supports loop=5, got {loop}")

    print(f"Loading symbol from: {path}")
    symb = convert(path, loop=loop)
    print(f"  loaded {len(symb):,} entries")
    return symb


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------
def audit_symbol(
    symb: dict[str, int],
    data_path: str | Path | None = None,
    upstream_root: str | Path | None = None,
) -> dict:
    """Run a comprehensive audit on the symbol data.

    Returns a dict with all audit results.  Raises AssertionError
    on any invariant violation.
    """
    data_path = resolve_data_path(data_path)
    upstream_root = resolve_upstream_root(upstream_root)

    report: dict = {
        "data_file": data_path.name,
        "data_path": str(data_path),
        "upstream_root": str(upstream_root),
    }

    # --- file-level checks ---
    if data_path.exists():
        file_bytes = data_path.read_bytes()
        report["file_size"] = len(file_bytes)
        report["file_md5"] = hashlib.md5(file_bytes).hexdigest()
    else:
        report["file_size"] = None
        report["file_md5"] = None

    report["aiamplitudes_commit"] = _get_git_commit(upstream_root)

    # --- core assertions ---
    n = len(symb)
    report["n_samples"] = n

    print(f"\n  Checking sample count: {n} == {EXPECTED_SAMPLES}?")
    assert n == EXPECTED_SAMPLES, f"Expected {EXPECTED_SAMPLES} samples, got {n}"
    print(f"    ✓ PASS")

    # word length
    word_lengths = Counter(len(w) for w in symb)
    report["word_length_distribution"] = dict(word_lengths)

    all_len10 = all(len(w) == EXPECTED_WORD_LENGTH for w in symb)
    print(f"\n  All words length {EXPECTED_WORD_LENGTH}? {all_len10}")
    assert all_len10, f"Not all words are length {EXPECTED_WORD_LENGTH}: {word_lengths}"
    print(f"    ✓ PASS")

    # alphabet validity
    valid_letters = set(alphabet)
    invalid_letters = set()
    for w in symb:
        for ch in w:
            if ch not in valid_letters:
                invalid_letters.add(ch)
    report["invalid_letters"] = sorted(invalid_letters)

    print(f"\n  All letters in {alphabet}? (invalid: {invalid_letters or 'none'})")
    assert not invalid_letters, f"Invalid letters found: {invalid_letters}"
    print(f"    ✓ PASS")

    # coefficient checks
    coeffs = list(symb.values())
    report["coeff_min"] = min(coeffs)
    report["coeff_max"] = max(coeffs)
    report["coeff_n_positive"] = sum(1 for c in coeffs if c > 0)
    report["coeff_n_negative"] = sum(1 for c in coeffs if c < 0)
    report["coeff_n_zero"] = sum(1 for c in coeffs if c == 0)
    report["coeff_n_unique"] = len(set(coeffs))

    print(f"\n  All coefficients non-zero? (zeros: {report['coeff_n_zero']})")
    assert report["coeff_n_zero"] == 0, f"Found {report['coeff_n_zero']} zero coefficients"
    print(f"    ✓ PASS")

    # --- coefficient distribution ---
    abs_coeffs = [abs(c) for c in coeffs]
    report["abs_coeff_min"] = min(abs_coeffs)
    report["abs_coeff_max"] = max(abs_coeffs)

    # --- base-1000 block count ---
    # base-1000 encoding: each block represents 0-999,
    # max_blocks = floor(log_1000(max(|c|))) + 1, but 0 needs 1 block
    def base1000_blocks(c: int) -> int:
        """Number of base-1000 blocks needed to encode |c|."""
        if c == 0:
            return 1
        a = abs(c)
        blocks = 0
        while a > 0:
            a //= 1000
            blocks += 1
        return blocks

    block_counts = [base1000_blocks(c) for c in coeffs]
    report["base1000_max_blocks"] = max(block_counts)
    report["base1000_block_distribution"] = dict(Counter(block_counts))

    # --- D3 orbit audit ---
    print(f"\n  Computing D3 orbits for {n:,} words...")
    orbit_ids = {}
    for w in symb:
        orbit_ids[w] = get_orbit_id(w)

    report["n_orbits"] = len(set(orbit_ids.values()))

    orbit_sizes = Counter()
    for oid in orbit_ids.values():
        orbit_sizes[oid] += 1
    report["orbit_size_distribution"] = dict(Counter(orbit_sizes.values()))

    # --- orbit coefficient consistency ---
    # Group by orbit_id and check all coefficients within each orbit are equal
    print(f"  Checking orbit coefficient consistency...")
    orbit_coeffs: dict[str, set[int]] = {}
    for w, c in symb.items():
        oid = orbit_ids[w]
        if oid not in orbit_coeffs:
            orbit_coeffs[oid] = set()
        orbit_coeffs[oid].add(c)

    inconsistent_orbits = {
        oid: coeff_set
        for oid, coeff_set in orbit_coeffs.items()
        if len(coeff_set) > 1
    }
    report["n_inconsistent_orbits"] = len(inconsistent_orbits)

    print(f"\n  All orbits have consistent coefficients? (inconsistent: {len(inconsistent_orbits)})")
    if inconsistent_orbits:
        print(f"    EXAMPLES of inconsistent orbits:")
        for i, (oid, cs) in enumerate(inconsistent_orbits.items()):
            print(f"      orbit {oid}: coeffs {cs}")
            if i >= 4:
                print(f"      ... and {len(inconsistent_orbits) - 5} more")
                break
    assert not inconsistent_orbits, (
        f"Found {len(inconsistent_orbits)} orbits with inconsistent coefficients"
    )
    print(f"    ✓ PASS")

    report["orbits_consistent"] = True

    return report


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------
def print_audit_report(report: dict) -> None:
    """Print a formatted audit report."""
    print("\n" + "=" * 72)
    print("  FIVE-LOOP SYMBOL DATA AUDIT REPORT")
    print("=" * 72)

    print(f"\n  Source file:")
    print(f"    path:        {report['data_path']}")
    if report.get("file_size"):
        print(f"    size:        {report['file_size']:,} bytes")
    if report.get("file_md5"):
        print(f"    MD5:         {report['file_md5']}")
    print(f"    AIA commit:  {report['aiamplitudes_commit']}")

    print(f"\n  Samples:")
    print(f"    total:       {report['n_samples']:,}")
    print(f"    word length: {report['word_length_distribution']}")
    invalid = report.get("invalid_letters", [])
    print(f"    invalid letters: {invalid if invalid else 'none'}")

    print(f"\n  Coefficients:")
    print(f"    min:         {report['coeff_min']:,}")
    print(f"    max:         {report['coeff_max']:,}")
    print(f"    positive:    {report['coeff_n_positive']:,} ({100*report['coeff_n_positive']/report['n_samples']:.1f}%)")
    print(f"    negative:    {report['coeff_n_negative']:,} ({100*report['coeff_n_negative']/report['n_samples']:.1f}%)")
    print(f"    zeros:       {report['coeff_n_zero']}")
    print(f"    unique:      {report['coeff_n_unique']:,}")
    print(f"    |min|:       {report['abs_coeff_min']:,}")
    print(f"    |max|:       {report['abs_coeff_max']:,}")

    print(f"\n  Base-1000 encoding:")
    print(f"    max blocks:  {report['base1000_max_blocks']}")
    print(f"    block dist:  {report['base1000_block_distribution']}")

    print(f"\n  D3 Orbits:")
    print(f"    n_orbits:    {report['n_orbits']:,}")
    print(f"    avg size:    {report['n_samples'] / report['n_orbits']:.2f}")
    print(f"    size dist:   {dict(sorted(report['orbit_size_distribution'].items()))}")
    print(f"    inconsistent: {report['n_inconsistent_orbits']}")

    print(f"\n  All assertions: ✓ PASSED")
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    """Run the full data audit pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        default=None,
        help="Raw EZ_symb_new_norm path (or NANOINFRA_SYMBOL_DATA_PATH)",
    )
    parser.add_argument(
        "--aiamplitudes-root",
        default=None,
        help=("AIAmplitudes_common_public checkout root "
              "(or NANOINFRA_AIAMPLITUDES_ROOT)"),
    )
    args = parser.parse_args()

    data_path = resolve_data_path(args.data_path)
    upstream_root = resolve_upstream_root(args.aiamplitudes_root)

    print("=" * 72)
    print("  STAGE 3: DATA PARSING AND D3 ORBIT AUDIT")
    print("=" * 72)

    # 1. Load
    symb = load_symbol(data_path)

    # 2. Audit
    print(f"\n[Audit] Running assertions...")
    report = audit_symbol(symb, data_path=data_path, upstream_root=upstream_root)

    # 3. Print
    print_audit_report(report)

    # 4. Check file-level assertions (separate from in-memory audit)
    if report.get("file_size"):
        assert report["file_size"] == EXPECTED_SIZE, (
            f"File size mismatch: expected {EXPECTED_SIZE}, got {report['file_size']}"
        )
        print(f"  File size check: {report['file_size']} == {EXPECTED_SIZE} ✓")
    if report.get("file_md5"):
        assert report["file_md5"] == EXPECTED_MD5, (
            f"MD5 mismatch: expected {EXPECTED_MD5}, got {report['file_md5']}"
        )
        print(f"  MD5 check: {report['file_md5']} == {EXPECTED_MD5} ✓")

    print(f"\n{'=' * 72}")
    print(f"  ALL CHECKS PASSED")
    print(f"{'=' * 72}")

    return report


if __name__ == "__main__":
    main()
