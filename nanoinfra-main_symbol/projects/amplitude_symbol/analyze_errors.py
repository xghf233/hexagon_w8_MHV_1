"""Error-set analysis: compare per-sample predictions of two checkpoints.

Takes the per-sample predictions saved by eval_checkpoint.py for two
checkpoints (default: the 500K and 150K word_bidi final checkpoints) and
answers, on the full val set:

  1. Error overlap — do the two models fail on the SAME samples?
     (Jaccard overlap, and for the shared errors: same wrong answer?)
  2. Error types — invalid / sign-only / magnitude (per model).
  3. Error distribution — error rate by word length, coefficient size
     (base-1000 blocks), decimal digit count, and D3 orbit size; plus the
     largest orbits containing errors and the top error samples.
  4. Disagreements — where exactly one model errs, which one is right?

Pure CPU analysis over saved prediction JSONs — no model loading, no CUDA.

Usage:
    python -m projects.amplitude_symbol.analyze_errors
    python -m projects.amplitude_symbol.analyze_errors \
        --predictions-a <path> --predictions-b <path> \
        --label-a NAME --label-b NAME
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_NANOINFRA_ROOT = Path(__file__).resolve().parents[2]
if str(_NANOINFRA_ROOT) not in sys.path:
    sys.path.insert(0, str(_NANOINFRA_ROOT))

from projects.amplitude_symbol.symmetry import get_orbit_id

DEFAULT_A = ("outputs/amplitude_symbol/big_eval/word_bidi/step_499999/"
             "predictions_val.json")
DEFAULT_B = ("outputs/amplitude_symbol/big_eval/word_bidi_150k/step_149999/"
             "predictions_val.json")


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def load_predictions(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    assert isinstance(data, list) and data and "word" in data[0], \
        f"unexpected predictions format in {path}"
    return data


# ----------------------------------------------------------------------
# Per-sample classification
# ----------------------------------------------------------------------

def classify(pred: int | None, true: int) -> str:
    """Classify one prediction: correct / invalid / sign / magnitude."""
    if pred == true:
        return "correct"
    if pred is None:
        return "invalid"
    if abs(pred) == abs(true):
        return "sign"          # right magnitude, wrong sign
    if (pred >= 0) == (true >= 0):
        return "magnitude"     # right sign, wrong magnitude
    return "sign_and_magnitude"


def coeff_blocks(true: int) -> int:
    """Number of base-1000 blocks of |true| (1 for |c| < 1000)."""
    return (len(str(abs(true))) + 2) // 3


def digit_count(true: int) -> int:
    """Number of decimal digits of |true| (0 -> 1)."""
    return max(len(str(abs(true))), 1)


# ----------------------------------------------------------------------
# Reporting helpers
# ----------------------------------------------------------------------

def print_table(title: str, header: list[str], rows: list[list[str]]) -> None:
    print(f"\n{title}")
    widths = [max(len(str(r[i])) for r in [header] + rows)
              for i in range(len(header))]
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*header))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*row))


def rate_table(buckets: dict, errors: dict, total: int) -> list[list[str]]:
    """Render per-bucket n / error / error-rate rows, buckets sorted by key."""
    rows = []
    for key in sorted(buckets, key=lambda k: (str(type(k)), k)):
        n = buckets[key]
        e = errors.get(key, 0)
        rows.append([str(key), str(n), str(e), f"{100 * e / n:.2f}%"])
    return rows


# ----------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions-a", default=DEFAULT_A)
    parser.add_argument("--predictions-b", default=DEFAULT_B)
    parser.add_argument("--label-a", default="500K")
    parser.add_argument("--label-b", default="150K")
    parser.add_argument("--save-summary", default=None,
                        help="Optional JSON path for the summary (default: none)")
    args = parser.parse_args()

    pa = load_predictions(args.predictions_a)
    pb = load_predictions(args.predictions_b)
    assert len(pa) == len(pb) and [s["word"] for s in pa] == [s["word"] for s in pb], \
        "prediction files disagree on sample order — different eval sets"
    n = len(pa)
    print(f"Loaded {n} samples each from {args.label_a} and {args.label_b}")

    labels = (args.label_a, args.label_b)
    sets = (pa, pb)
    results = {}   # per-model aggregated facts, for the optional summary

    # ---- 1. per-model stats ------------------------------------------
    for label, preds in zip(labels, sets):
        counts = Counter()
        for s in preds:
            counts[classify(s["predicted_coefficient"], s["true_coefficient"])] += 1
        err = n - counts["correct"]
        print(f"\n=== {label}: {err} errors / {n} "
              f"({100 * counts['correct'] / n:.4f}% exact)")
        print_table(f"{label} error-type breakdown", ["type", "n", "share of errors"],
                    [[k, str(counts[k]), f"{100 * counts[k] / err:.1f}%"]
                     for k in ("invalid", "sign", "magnitude", "sign_and_magnitude")])
        results[label] = {"counts": dict(counts), "n_errors": err,
                          "error_idx": {i for i, s in enumerate(preds)
                                        if s["predicted_coefficient"] != s["true_coefficient"]}}

    err_a, err_b = results[args.label_a]["error_idx"], results[args.label_b]["error_idx"]

    # ---- 2. error overlap --------------------------------------------
    both = err_a & err_b
    a_only = err_a - err_b
    b_only = err_b - err_a
    jaccard = len(both) / len(err_a | err_b)
    same_wrong = sum(
        1 for i in both
        if sets[0][i]["predicted_coefficient"] == sets[1][i]["predicted_coefficient"]
        and sets[0][i]["predicted_coefficient"] is not None
    )
    print(f"\n=== Error overlap")
    print_table("overlap", [args.label_a, args.label_b, "both", "jaccard",
                            "shared errors w/ same wrong pred"],
                [[str(len(err_a)), str(len(err_b)), str(len(both)),
                  f"{jaccard:.3f}", f"{same_wrong} ({100 * same_wrong / len(both):.0f}%)"
                  if both else "-"]])

    # ---- 3. disagreement winners -------------------------------------
    a_right_b_wrong = sum(
        1 for i in b_only if sets[0][i]["predicted_coefficient"] == sets[0][i]["true_coefficient"])
    b_right_a_wrong = sum(
        1 for i in a_only if sets[1][i]["predicted_coefficient"] == sets[1][i]["true_coefficient"])
    print_table("disagreements (exactly one model errs)",
                ["case", "n"],
                [[f"{args.label_a} wrong, {args.label_b} right", str(b_right_a_wrong)],
                 [f"{args.label_a} right, {args.label_b} wrong", str(a_right_b_wrong)]])

    # ---- 4. error distribution by sample properties ------------------
    words = [s["word"] for s in pa]
    truths = [s["true_coefficient"] for s in pa]
    orbit_sizes = Counter()
    for w in words:
        orbit_sizes[get_orbit_id(w)] += 1

    for label, preds in zip(labels, sets):
        err_idx = results[label]["error_idx"]
        buckets_len = Counter(len(w) for w in words)
        buckets_blocks = Counter(coeff_blocks(t) for t in truths)
        buckets_digits = Counter(digit_count(t) for t in truths)
        buckets_orbit = Counter(
            1 if orbit_sizes[get_orbit_id(w)] == 1 else
            2 if orbit_sizes[get_orbit_id(w)] == 2 else
            3 if orbit_sizes[get_orbit_id(w)] <= 5 else 6
            for w in words)

        print(f"\n=== {label}: error rate by sample property "
              f"(buckets ordered; overall rate {100 * len(err_idx) / n:.2f}%)")

        def err_counter(fn):
            c = Counter()
            for i in err_idx:
                c[fn(i)] += 1
            return c

        print_table("by word length", ["len(word)", "n", "errors", "rate"],
                    rate_table(buckets_len, err_counter(lambda i: len(words[i])), n))
        print_table("by coefficient blocks (base-1000)",
                    ["blocks", "n", "errors", "rate"],
                    rate_table(buckets_blocks, err_counter(lambda i: coeff_blocks(truths[i])), n))
        print_table("by decimal digits of |c|",
                    ["digits", "n", "errors", "rate"],
                    rate_table(buckets_digits, err_counter(lambda i: digit_count(truths[i])), n))
        print_table("by D3 orbit size",
                    ["orbit size", "n", "errors", "rate"],
                    rate_table(buckets_orbit,
                               err_counter(lambda i: 1 if orbit_sizes[get_orbit_id(words[i])] == 1
                                           else 2 if orbit_sizes[get_orbit_id(words[i])] == 2
                                           else 3 if orbit_sizes[get_orbit_id(words[i])] <= 5
                                           else 6), n))

    # ---- 5. orbits containing errors ----------------------------------
    for label, preds in zip(labels, sets):
        err_idx = results[label]["error_idx"]
        orbit_errs = defaultdict(list)
        for i in err_idx:
            orbit_errs[get_orbit_id(words[i])].append(i)
        multi = {oid: idxs for oid, idxs in orbit_errs.items() if len(idxs) >= 2}
        print(f"\n=== {label}: {len(orbit_errs)} orbits contain errors; "
              f"{len(multi)} of them have >=2 error members")
        if multi:
            rows = []
            for oid, idxs in sorted(multi.items(), key=lambda kv: -len(kv[1]))[:8]:
                detail = ", ".join(f"{words[i]} -> {preds[i]['predicted_coefficient']}"
                                   for i in idxs[:4])
                rows.append([oid, str(orbit_sizes[oid]), str(len(idxs)), detail])
            print_table("top multi-error orbits", ["orbit id", "size", "errors", "examples"], rows)

    # ---- 6. top error samples per model --------------------------------
    for label, preds in zip(labels, sets):
        err_idx = results[label]["error_idx"]
        rows = []
        for i in sorted(err_idx)[:20]:
            s = preds[i]
            rows.append([s["word"], str(s["true_coefficient"]),
                         str(s["predicted_coefficient"]),
                         classify(s["predicted_coefficient"], s["true_coefficient"])])
        print_table(f"first 20 error samples ({label})",
                    ["word", "true", "pred", "type"], rows)

    # ---- optional summary JSON ----------------------------------------
    if args.save_summary:
        summary = {
            "n_samples": n,
            args.label_a: {"n_errors": results[args.label_a]["n_errors"],
                           "counts": results[args.label_a]["counts"]},
            args.label_b: {"n_errors": results[args.label_b]["n_errors"],
                           "counts": results[args.label_b]["counts"]},
            "overlap": {"both": len(both), "a_only": len(a_only),
                        "b_only": len(b_only), "jaccard": jaccard},
        }
        Path(args.save_summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_summary).write_text(json.dumps(summary, indent=2))
        print(f"\nSummary saved to {args.save_summary}")


if __name__ == "__main__":
    main()
