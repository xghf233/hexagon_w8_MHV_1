"""Standalone checkpoint evaluation for amplitude-symbol experiments.

Free-generation eval on a saved checkpoint, no training involved.  Exists so
final checkpoints can be evaluated on a much larger sample budget than the
periodic 512-sample evals: periodic evals keep the fixed first-512 val set
(their curves are comparable across runs), and this script re-evaluates the
final checkpoints of the 500K and 150K runs on the same large set (default:
the entire split — 26,388 val samples) so the two runs can be compared at
±0.12pp binomial noise instead of ±0.9pp.

The word-bidirectional attention mask is a runtime attribute (it is NOT in the
checkpoint state_dict). New checkpoints record an ``attention_config`` metadata
entry so the script can restore it. Legacy checkpoints require an explicit
``--attention-mode`` argument and are never guessed silently.

Usage:
    # Big-eval on the 500K run's final checkpoint (full val set):
    python -m projects.amplitude_symbol.eval_checkpoint \
        --checkpoint models/amplitude_symbol/word_bidi/step_499999 \
        --attention-mode word_bidirectional --split val

    # Big-eval on the 150K run's final checkpoint, keeping predictions
    # for the later error-set analysis:
    python -m projects.amplitude_symbol.eval_checkpoint \
        --checkpoint models/amplitude_symbol/word_bidi_150k/step_149999 \
        --attention-mode word_bidirectional --split val --save-predictions
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_NANOINFRA_ROOT = Path(__file__).resolve().parents[2]
if str(_NANOINFRA_ROOT) not in sys.path:
    sys.path.insert(0, str(_NANOINFRA_ROOT))

import torch

from core.model.gpt import GPT
from core.model.checkpoint_manager import load_metadata
from core.training.model_setup import load_system
from core.utils import print0

from projects.amplitude_symbol.blocks.attention import (
    WORD_END,
    WORD_START,
    attach_word_bidirectional_attention,
)
from projects.amplitude_symbol.evaluator import evaluate, save_evaluation
from projects.amplitude_symbol.paths import resolve_data_path, resolve_manifest_path
from projects.amplitude_symbol.prepare_data import load_symbol
from projects.amplitude_symbol.split import get_split_coeffs, load_manifest

def resolve_attention_config(metadata: dict, requested_mode: str) -> dict:
    """Resolve runtime attention without guessing legacy checkpoint semantics."""
    recorded = metadata.get("attention_config")
    recorded_mode = recorded.get("mode") if isinstance(recorded, dict) else None

    if requested_mode == "auto":
        if recorded_mode is None:
            raise ValueError(
                "Checkpoint metadata has no attention_config. This is expected for "
                "legacy checkpoints; pass --attention-mode word_bidirectional or "
                "--attention-mode causal explicitly."
            )
        requested_mode = recorded_mode
    elif recorded_mode is not None and recorded_mode != requested_mode:
        raise ValueError(
            f"Checkpoint records attention mode {recorded_mode!r}, but "
            f"--attention-mode requested {requested_mode!r}."
        )

    if requested_mode not in {"word_bidirectional", "causal"}:
        raise ValueError(f"Unsupported checkpoint attention mode: {requested_mode!r}")

    if requested_mode == "word_bidirectional":
        return {
            "mode": requested_mode,
            "word_start": int((recorded or {}).get("word_start", WORD_START)),
            "word_end": int((recorded or {}).get("word_end", WORD_END)),
        }
    return {"mode": "causal"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a step_XXXXXX checkpoint directory")
    parser.add_argument(
        "--attention-mode",
        choices=("auto", "word_bidirectional", "causal"),
        default="auto",
        help=("Runtime attention mode. 'auto' requires attention_config in "
              "checkpoint metadata; legacy checkpoints need an explicit mode."),
    )
    parser.add_argument("--split", default="val",
                        help="Split to evaluate (default: val)")
    parser.add_argument("--n-samples", type=int, default=0,
                        help="Sample budget (default: 0 = entire split)")
    parser.add_argument("--gen-batch-size", type=int, default=64,
                        help="Batch size for generation (default: 64)")
    parser.add_argument("--save-predictions", action="store_true",
                        help="Save per-sample predictions JSON next to the result")
    parser.add_argument("--out-dir", default=None,
                        help="Output dir (default: "
                             "outputs/amplitude_symbol/big_eval/<model>/<step>/)")
    parser.add_argument(
        "--data-path",
        default=None,
        help="Raw EZ_symb_new_norm path (or NANOINFRA_SYMBOL_DATA_PATH)",
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Split manifest path (or NANOINFRA_SYMBOL_MANIFEST_PATH)",
    )
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint).resolve()
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_dir}")
    metadata = load_metadata(str(ckpt_dir))
    attention_config = resolve_attention_config(metadata, args.attention_mode)

    assert torch.cuda.is_available(), "CUDA is required for evaluation"
    torch.cuda.set_device(0)
    step_name = ckpt_dir.name          # e.g. step_499999
    model_name = ckpt_dir.parent.name  # e.g. word_bidi

    print0("=" * 72)
    print0("  CHECKPOINT EVALUATION (free generation)")
    print0(f"  checkpoint: {ckpt_dir}")
    print0(f"  split:      {args.split}")
    print0(f"  attention:  {attention_config['mode']}")
    print0("=" * 72)

    # Data: same manifest + symbol loading path as training.
    symb = load_symbol(resolve_data_path(args.data_path))
    manifest = load_manifest(resolve_manifest_path(args.manifest_path))
    pairs = get_split_coeffs(manifest, symb, args.split, "orbit_grouped")
    n = len(pairs) if args.n_samples == 0 else min(args.n_samples, len(pairs))
    pairs = pairs[:n]
    print0(f"  Evaluating {len(pairs)} {args.split} samples")

    # Model: self-describing checkpoint -> standard inference assembly path.
    setup = load_system(str(ckpt_dir), trunk_cls=GPT, use_compile=False)
    system = setup["system"]
    sequence_len = setup["gpt_config"].sequence_len
    if attention_config["mode"] == "word_bidirectional":
        attach_word_bidirectional_attention(
            system,
            sequence_len,
            word_start=attention_config["word_start"],
            word_end=attention_config["word_end"],
        )
        print0(
            "  Word-bidirectional attention mask attached "
            f"(positions {attention_config['word_start']}-"
            f"{attention_config['word_end'] - 1})"
        )
    else:
        system.attention_config = {"mode": "causal"}

    # Output: default location keeps big-eval results out of the training
    # outputs dirs, so no existing artifacts are touched.
    out_dir = Path(args.out_dir) if args.out_dir else (
        _NANOINFRA_ROOT / "outputs" / "amplitude_symbol" / "big_eval"
        / model_name / step_name
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    result = evaluate(
        system, pairs,
        split_name=args.split,
        checkpoint=step_name,
        sequence_len=sequence_len,
        gen_batch_size=args.gen_batch_size,
        save_predictions=args.save_predictions,
        predictions_dir=str(out_dir) if args.save_predictions else None,
    )

    eval_path = out_dir / f"evaluation_{args.split}.json"
    save_evaluation(result, eval_path)
    print0(f"  Evaluation saved to {eval_path}")

    sm = result.sample
    om = result.orbit
    print0(f"  Sample: exact={sm.exact_accuracy:.4f}  mag={sm.magnitude_accuracy:.4f}  "
           f"sign={sm.sign_accuracy:.4f}  invalid={sm.invalid_rate:.4f}")
    print0(f"  Orbit:  consistency={om.orbit_consistency:.4f}  "
           f"whole={om.whole_orbit_accuracy:.4f}  "
           f"(eligible={om.n_eligible_orbits}, singleton={om.n_singleton_orbits})")


if __name__ == "__main__":
    main()
