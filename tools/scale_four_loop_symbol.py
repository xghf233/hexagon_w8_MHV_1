#!/usr/bin/env python3
"""Export C4(W) = 32*c4(W), without changing the original rational artifact."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import io
import json
from math import gcd, lcm
from pathlib import Path


FACTOR = 32
SOURCE_SHA256 = "5eaa2ad35811e8a093540115ae56cba9dda84f118447bdf34212f820dac5b694"
MANIFEST_SHA256 = "b2019c8d20db77f9e62eb437a00ec329df8b8872ba065df5c55c222295c6886c"
ALPHABET = ["a", "b", "c", "mu", "mv", "mw", "yu", "yv", "yw"]
LETTER_IDS = {letter: i for i, letter in enumerate(ALPHABET)}
OUTPUT_NAME = "four_loop_mhv_symbol_native_x32.jsonl.gz"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def scale_row(row):
    if set(row) != {"word", "numerator", "denominator"}:
        raise ValueError("Unexpected row fields")
    word = row["word"]
    if not isinstance(word, list) or len(word) != 8:
        raise ValueError("Expected an eight-letter word")
    if any(not isinstance(x, str) or x not in LETTER_IDS for x in word):
        raise ValueError("Unknown letter")
    for field in ("numerator", "denominator"):
        if not isinstance(row[field], str) or str(int(row[field])) != row[field]:
            raise ValueError("Expected canonical decimal integer strings")
    numerator, denominator = int(row["numerator"]), int(row["denominator"])
    if numerator == 0 or denominator <= 0 or gcd(numerator, denominator) != 1:
        raise ValueError("Expected a nonzero reduced rational with positive denominator")
    scaled, remainder = divmod(FACTOR * numerator, denominator)
    if remainder:
        raise ValueError("Scale factor does not clear this denominator")
    return {"word": word, "numerator": str(scaled), "denominator": "1"}


def verify_export(source, output):
    """Read both gzip streams to EOF and verify via independent Fraction arithmetic."""
    rows = 0
    with gzip.open(source, "rt", encoding="utf-8") as original, gzip.open(
        output, "rt", encoding="utf-8"
    ) as scaled:
        for rows, (raw_line, new_line) in enumerate(zip(original, scaled, strict=True), 1):
            raw, new = json.loads(raw_line), json.loads(new_line)
            if set(new) != {"word", "numerator", "denominator"}:
                raise ValueError(f"Unexpected output fields at row {rows}")
            if raw["word"] != new["word"] or new["denominator"] != "1":
                raise ValueError(f"Word or denominator mismatch at row {rows}")
            value = new["numerator"]
            if not isinstance(value, str) or str(int(value)) != value:
                raise ValueError(f"Noncanonical output integer at row {rows}")
            original_value = Fraction(int(raw["numerator"]), int(raw["denominator"]))
            if Fraction(int(value), FACTOR) != original_value:
                raise ValueError(f"Exact inverse scaling failed at row {rows}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    source = args.source.resolve()
    source_manifest = source.parent / "extraction_manifest.json"
    if sha256(source) != SOURCE_SHA256 or sha256(source_manifest) != MANIFEST_SHA256:
        raise ValueError("Source data or extraction manifest differs from the verified version")
    metadata = json.loads(source_manifest.read_text(encoding="utf-8"))
    if metadata["alphabet"] != ALPHABET or metadata["rows"] != 243000:
        raise ValueError("Unexpected source metadata")

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / OUTPUT_NAME
    histogram, denominators, y_counts = Counter(), Counter(), Counter()
    previous = None
    rows = 0
    integer_gcd = 0
    denominator_lcm = 1
    with gzip.open(source, "rt", encoding="utf-8") as original, output.open("xb") as binary:
        with gzip.GzipFile(filename="", fileobj=binary, mode="wb", mtime=0, compresslevel=9) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as target:
                for line in original:
                    row = json.loads(line)
                    new = scale_row(row)
                    key = tuple(LETTER_IDS[x] for x in row["word"])
                    if previous is not None and key <= previous:
                        raise ValueError("Source words are not sorted and unique")
                    previous = key
                    coefficient = int(new["numerator"])
                    denominator = int(row["denominator"])
                    histogram[coefficient] += 1
                    denominators[str(denominator)] += 1
                    y_counts[str(sum(x.startswith("y") for x in row["word"]))] += 1
                    integer_gcd = gcd(integer_gcd, coefficient)
                    denominator_lcm = lcm(denominator_lcm, denominator)
                    rows += 1
                    target.write(json.dumps(new, separators=(",", ":")) + "\n")

    if rows != metadata["rows"] or denominators != metadata["denominator_histogram"]:
        raise ValueError("Source counts disagree with the extraction manifest")
    if y_counts != metadata["nonzero_rows_by_y_count"] or denominator_lcm != FACTOR:
        raise ValueError("Source support or minimal scaling factor mismatch")
    verified = verify_export(source, output)
    if verified != rows or sha256(source) != SOURCE_SHA256 or sha256(source_manifest) != MANIFEST_SHA256:
        raise ValueError("Verification count mismatch or original files changed")

    manifest = {
        "schema_version": 1,
        "stage": "four_loop_native_symbol_integer_scaling",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(source), "sha256": SOURCE_SHA256,
                   "extraction_manifest_path": str(source_manifest),
                   "extraction_manifest_sha256": MANIFEST_SHA256},
        "physical_object": "ordinary BDS-like four-loop Hexagon MHV full weight-8 symbol",
        "physical_normalization_changed": False,
        "coefficient_scaling": {"version": "hexagon_mhv_4loop_native_x32_v1",
                                "integer_label_scaling_applied": True, "factor": FACTOR,
                                "forward": "C4(W) = 32*c4(W)", "inverse": "c4(W) = C4(W)/32",
                                "apply_to_conditioning_and_target": True},
        "loop": 4, "weight": 8,
        "alphabet": ALPHABET,
        "training_aliases": metadata["training_aliases"],
        "alphabet_version": metadata["alphabet_version"],
        "alphabet_conversion_applied": False,
        "word_order": metadata["word_order"], "row_order": metadata["row_order"],
        "coefficient_format": "scaled integer in numerator string; denominator string always 1",
        "rows": rows, "nonzero_rows_by_y_count": dict(sorted(y_counts.items())),
        "source_denominator_histogram": dict(sorted(denominators.items(), key=lambda item: int(item[0]))),
        "source_denominator_lcm": denominator_lcm,
        "denominator_histogram": {"1": rows},
        "minimum_coefficient": min(histogram), "maximum_coefficient": max(histogram),
        "distinct_nonzero_coefficients": len(histogram), "integer_coefficient_gcd": integer_gcd,
        "coefficient_histogram": {str(k): v for k, v in sorted(histogram.items())},
        "output": {"filename": OUTPUT_NAME, "bytes": output.stat().st_size, "sha256": sha256(output)},
        "generator": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "verification": {"all_rows_verified": verified, "exact_inverse_scaling_passed": True,
                         "word_support_and_order_unchanged": True, "gzip_eof_passed": True,
                         "original_data_and_manifest_hashes_unchanged": True,
                         "full_integrability_newly_checked": False},
        "training_samples_constructed": False,
    }
    manifest_path = output_dir / "scaling_manifest.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(output), "rows": rows, "verified_rows": verified,
                      "range": [min(histogram), max(histogram)], "gcd": integer_gcd,
                      "sha256": manifest["output"]["sha256"], "bytes": output.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
