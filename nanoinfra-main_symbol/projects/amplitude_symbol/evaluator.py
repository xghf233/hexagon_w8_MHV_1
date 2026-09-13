"""Free-generation evaluator for amplitude symbol coefficient prediction.

Uses NanoInfra's autoregressive_generate to produce coefficient tokens,
then applies strict decoding and computes sample-level and D3 orbit-level
metrics.  Never uses teacher forcing, constrained decoding, D3 augmentation,
canonicalization, or orbit averaging.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from core.model.inference import autoregressive_generate

from .symmetry import get_orbit_id
from .tokenizer import (
    BOS_ID,
    COEFF_ID,
    COEFF_OFFSET,
    COEFF_NUM_START,
    COEFF_SIZE,
    EOS_ID,
    MINUS_ID,
    PAD_ID,
    PLUS_ID,
    WORD_OFFSET,
    WORD_TOKEN_IDS,
    token_type_of,
)


# ======================================================================
# Strict coefficient decoder
# ======================================================================

def strict_decode_coefficient(gen_ids: list[int]) -> int | None:
    """Decode generated token IDs into a coefficient integer, or None.

    Enforces every validity constraint.  gen_ids is the raw output of
    ``autoregressive_generate``, which may contain *padding* EOS tokens
    after the first EOS (inserted by the engine for batch alignment).
    Those padding EOS tokens are ignored.

    Invalid conditions (return None):
      1. No EOS within the sequence
      2. First token before EOS is not PLUS or MINUS
      3. Sign is not followed by at least one NUM token
      4. Any token between sign and EOS is not a valid NUM token
      5. Illegal token (word, BOS, COEFF, PAD) appears before EOS
      6. Leading zero: ≥2 blocks and highest block is NUM_0
      7. Negative zero: MINUS followed by a single NUM_0
    """
    # ---- locate the first (real) EOS ----
    eos_idx = None
    for i, tid in enumerate(gen_ids):
        if tid == EOS_ID:
            eos_idx = i
            break
    if eos_idx is None:
        return None                         # (1) no EOS

    prefix = gen_ids[:eos_idx]
    if len(prefix) < 2:
        return None                         # need at least sign + one block

    # ---- illegal tokens before EOS ----
    for tid in prefix:
        if WORD_OFFSET <= tid < WORD_OFFSET + 6:
            return None                     # (5) word letter
        if tid in (BOS_ID, COEFF_ID, PAD_ID):
            return None                     # (5) BOS / COEFF / PAD

    # ---- sign ----
    sign_id = prefix[0]
    if sign_id == PLUS_ID:
        sign = 1
    elif sign_id == MINUS_ID:
        sign = -1
    else:
        return None                         # (2) bad sign

    # ---- blocks ----
    blocks = prefix[1:]
    if not blocks:
        return None                         # (3) no NUM after sign

    # Validate every block token
    for tid in blocks:
        ttype = token_type_of(tid)
        if ttype != 2:
            return None                     # (4) not a coeff-band token
        if tid < COEFF_OFFSET + COEFF_NUM_START:
            return None                     # (4) sign token in block position

    # Leading-zero check: if ≥2 blocks, highest can't be NUM_0
    if len(blocks) >= 2:
        first_val = blocks[0] - COEFF_OFFSET - COEFF_NUM_START
        if first_val == 0:
            return None                     # (6) leading zero

    # Negative zero check
    if sign == -1 and len(blocks) == 1:
        val = blocks[0] - COEFF_OFFSET - COEFF_NUM_START
        if val == 0:
            return None                     # (7) negative zero

    # ---- compute value ----
    abs_val = 0
    for tid in blocks:
        abs_val = abs_val * 1000 + (tid - COEFF_OFFSET - COEFF_NUM_START)

    return sign * abs_val


def _decode_all(gen_ids_batch: list[list[int]]) -> list[int | None]:
    """Decode a batch of generated token lists."""
    return [strict_decode_coefficient(ids) for ids in gen_ids_batch]


# ======================================================================
# Generation
# ======================================================================

@torch.no_grad()
def generate_coefficients(
    system,
    words: list[str],
    *,
    sequence_len: int = 32,
    gen_batch_size: int = 64,
) -> list[int | None]:
    """Free-generate coefficients for a list of words.

    Uses greedy decoding (temperature=0) with ``autoregressive_generate``.
    The prompt is ``[BOS] + word_letters + [COEFF]`` — the model never sees
    the ground-truth coefficient.

    Returns a list of ``int | None``, one per input word (None = invalid).
    """
    device = next(system.parameters()).device
    system.eval()

    predictions: list[int | None] = []
    n_words = len(words)

    for start in range(0, n_words, gen_batch_size):
        batch_words = words[start:start + gen_batch_size]
        B = len(batch_words)

        # Build prompt: [BOS][word][COEFF] — all same length (12 tokens)
        prompt_len = 1 + 10 + 1  # BOS + 10 letters + COEFF
        prompt_ids = torch.full((B, prompt_len), PAD_ID, dtype=torch.long, device=device)
        prompt_types = torch.full((B, prompt_len), 1, dtype=torch.long, device=device)

        for i, w in enumerate(batch_words):
            prompt_ids[i, 0] = BOS_ID
            for j, ch in enumerate(w):
                prompt_ids[i, 1 + j] = WORD_OFFSET + WORD_TOKEN_IDS[ch]
                prompt_types[i, 1 + j] = 0      # word type
            prompt_ids[i, 11] = COEFF_ID

        # Generate
        max_new = sequence_len - prompt_len
        gen_ids = autoregressive_generate(
            system,
            prompt_ids,
            prompt_types,
            max_new_tokens=max_new,
            gen_token_type=2,       # coefficient band (sign + blocks share type 2)
            stop_token=EOS_ID,
            temperature=0.0,        # greedy
            top_k=None,
            early_stop=True,
        )

        # Decode each sequence
        gen_lists = gen_ids.cpu().tolist()
        for i in range(B):
            pred = strict_decode_coefficient(gen_lists[i])
            predictions.append(pred)

    return predictions


# ======================================================================
# Metrics
# ======================================================================

@dataclass
class SampleMetrics:
    """Per-sample evaluation results."""
    n_samples: int = 0
    n_valid: int = 0
    n_invalid: int = 0
    n_exact_correct: int = 0
    n_magnitude_correct: int = 0
    n_sign_correct: int = 0

    @property
    def exact_accuracy(self) -> float:
        return self.n_exact_correct / self.n_samples if self.n_samples else 0.0

    @property
    def magnitude_accuracy(self) -> float:
        return self.n_magnitude_correct / self.n_samples if self.n_samples else 0.0

    @property
    def sign_accuracy(self) -> float:
        return self.n_sign_correct / self.n_samples if self.n_samples else 0.0

    @property
    def invalid_rate(self) -> float:
        return self.n_invalid / self.n_samples if self.n_samples else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "n_valid": self.n_valid,
            "n_invalid": self.n_invalid,
            "exact_accuracy": round(self.exact_accuracy, 6),
            "magnitude_accuracy": round(self.magnitude_accuracy, 6),
            "sign_accuracy": round(self.sign_accuracy, 6),
            "invalid_output_rate": round(self.invalid_rate, 6),
        }


@dataclass
class OrbitMetrics:
    """D3 orbit-level evaluation results."""
    n_orbits: int = 0
    n_eligible_orbits: int = 0        # ≥2 members
    n_singleton_orbits: int = 0
    n_consistent_orbits: int = 0      # all valid + same predicted coeff
    n_whole_correct_orbits: int = 0   # all valid + every pred == true

    @property
    def orbit_consistency(self) -> float:
        return (self.n_consistent_orbits / self.n_eligible_orbits
                if self.n_eligible_orbits else 0.0)

    @property
    def whole_orbit_accuracy(self) -> float:
        return (self.n_whole_correct_orbits / self.n_eligible_orbits
                if self.n_eligible_orbits else 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_orbits": self.n_orbits,
            "n_eligible_orbits": self.n_eligible_orbits,
            "n_singleton_orbits": self.n_singleton_orbits,
            "orbit_consistency": round(self.orbit_consistency, 6),
            "whole_orbit_accuracy": round(self.whole_orbit_accuracy, 6),
        }


def compute_sample_metrics(
    predictions: list[int | None],
    truths: list[int],
) -> SampleMetrics:
    """Compute sample-level metrics from predictions and ground-truth."""
    assert len(predictions) == len(truths), \
        f"Length mismatch: {len(predictions)} preds vs {len(truths)} truths"

    m = SampleMetrics()
    m.n_samples = len(predictions)

    for pred, true in zip(predictions, truths):
        if pred is None:
            m.n_invalid += 1
            # Invalid prediction → counted as wrong in all accuracy metrics
            continue

        m.n_valid += 1

        if pred == true:
            m.n_exact_correct += 1
        if abs(pred) == abs(true):
            m.n_magnitude_correct += 1
        if (pred >= 0 and true >= 0) or (pred < 0 and true < 0):
            m.n_sign_correct += 1

    return m


def compute_orbit_metrics(
    predictions: list[int | None],
    truths: list[int],
    words: list[str],
) -> OrbitMetrics:
    """Compute D3 orbit-level metrics.

    Groups samples by orbit_id, then computes:
      - orbit_consistency: all predictions in orbit are valid and identical
      - whole_orbit_accuracy: all predictions in orbit are valid and match truth

    Raises ValueError if any orbit has conflicting ground-truth coefficients.
    """
    assert len(predictions) == len(truths) == len(words)

    # Group by orbit_id
    orbit_groups: dict[str, list[int]] = defaultdict(list)
    for i, w in enumerate(words):
        oid = get_orbit_id(w)
        orbit_groups[oid].append(i)

    # Truth validation: every orbit must have consistent ground-truth
    for oid, indices in orbit_groups.items():
        truth_set = {truths[i] for i in indices}
        if len(truth_set) > 1:
            raise ValueError(
                f"Orbit {oid} has conflicting truth coefficients: {truth_set}. "
                f"Words: {[words[i] for i in indices]}"
            )

    m = OrbitMetrics()
    m.n_orbits = len(orbit_groups)

    for oid, indices in orbit_groups.items():
        if len(indices) < 2:
            m.n_singleton_orbits += 1
            continue

        m.n_eligible_orbits += 1

        orbit_preds = [predictions[i] for i in indices]
        orbit_truths = [truths[i] for i in indices]

        # All must be valid
        if any(p is None for p in orbit_preds):
            continue

        # Consistent: all predicted coefficients are identical
        unique_preds = set(orbit_preds)
        if len(unique_preds) == 1:
            m.n_consistent_orbits += 1

        # Whole-orbit correct: every prediction equals truth
        all_correct = all(p == t for p, t in zip(orbit_preds, orbit_truths))
        if all_correct:
            m.n_whole_correct_orbits += 1

    return m


# ======================================================================
# Main evaluation entry point
# ======================================================================

@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    sample: SampleMetrics = field(default_factory=SampleMetrics)
    orbit: OrbitMetrics = field(default_factory=OrbitMetrics)
    meta: dict[str, Any] = field(default_factory=dict)


def evaluate(
    system,
    samples: list[tuple[str, int]],
    *,
    split_name: str = "test",
    checkpoint: str | None = None,
    sequence_len: int = 32,
    gen_batch_size: int = 64,
    save_predictions: bool = False,
    predictions_dir: str | None = None,
) -> EvaluationResult:
    """Run a full free-generation evaluation.

    Args:
        system: A NanoInfra LMSystem (trunk + head).
        samples: List of (word, coefficient) pairs to evaluate.
        split_name: Label for the split being evaluated.
        checkpoint: Checkpoint identifier (optional).
        sequence_len: Total sequence length (prompt + generation).
        gen_batch_size: Batch size for generation.
        save_predictions: Whether to save per-sample predictions to disk.
        predictions_dir: Directory for prediction files.

    Returns:
        EvaluationResult with sample-level and orbit-level metrics.
    """
    t0 = time.time()
    words = [w for w, _ in samples]
    truths = [c for _, c in samples]

    # 1. Generate
    predictions = generate_coefficients(
        system, words,
        sequence_len=sequence_len,
        gen_batch_size=gen_batch_size,
    )

    # 2. Compute metrics
    sample_metrics = compute_sample_metrics(predictions, truths)
    orbit_metrics = compute_orbit_metrics(predictions, truths, words)

    elapsed = time.time() - t0

    # 3. Assemble result
    result = EvaluationResult(
        sample=sample_metrics,
        orbit=orbit_metrics,
        meta={
            "split": split_name,
            "checkpoint": checkpoint or "unknown",
            "n_samples": len(samples),
            "n_orbits": orbit_metrics.n_orbits,
            "decoding": {
                "method": "free_autoregressive_generation",
                "temperature": 0.0,
                "top_k": None,
                "gen_token_type": 2,
                "stop_token": "EOS",
                "sequence_len": sequence_len,
                "gen_batch_size": gen_batch_size,
            },
            "elapsed_seconds": round(elapsed, 2),
        },
    )

    # 4. Optionally save per-sample predictions
    if save_predictions and predictions_dir:
        pred_dir = Path(predictions_dir)
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_file = pred_dir / f"predictions_{split_name}.json"
        pred_data = [
            {"word": w, "true_coefficient": t, "predicted_coefficient": p,
             "valid": p is not None}
            for w, t, p in zip(words, truths, predictions)
        ]
        pred_file.write_text(json.dumps(pred_data, indent=2))
        result.meta["predictions_file"] = str(pred_file)

    return result


# ======================================================================
# Trainer-compatible periodic evaluator
# ======================================================================
class FreeGenEvaluator:
    """Periodic free-generation evaluator for use with core's Trainer.

    Caches a fixed eval set at construction and runs full free-generation
    evaluation when the Trainer calls ``evaluate()``.  Implements the
    duck-typed Evaluator interface (interval_steps / should_eval / evaluate).

    Usage in train.py::

        eval_pairs = get_split_coeffs(manifest, symb, "val", "orbit_grouped")
        eval_pairs = eval_pairs[:n_eval_samples]   # fixed subset
        evaluator = FreeGenEvaluator(
            eval_pairs,
            interval_steps=2000,
            sequence_len=32,
            gen_batch_size=64,
            split_name="val",
            save_dir=None,            # disable per-eval JSON saves
        )
        trainer = Trainer(..., evaluators=[evaluator])
    """

    def __init__(
        self,
        samples: list[tuple[str, int]],
        *,
        interval_steps: int = 2000,
        sequence_len: int = 32,
        gen_batch_size: int = 64,
        split_name: str = "val",
        save_dir: str | None = None,
    ):
        self.samples = samples
        self.interval_steps = interval_steps
        self.eval_at = None   # use interval-based scheduling
        self.sequence_len = sequence_len
        self.gen_batch_size = gen_batch_size
        self.split_name = split_name
        self.save_dir = Path(save_dir) if save_dir else None
        self.best_exact = 0.0
        self.best_ce = float("inf")

    def should_eval(self, step: int) -> bool:
        if self.eval_at is not None:
            return step in self.eval_at
        return step > 0 and step % self.interval_steps == 0

    def describe(self) -> str:
        return (f"free-generation eval on {len(self.samples)} {self.split_name} "
                f"samples, every {self.interval_steps} steps")

    @torch.no_grad()
    def evaluate(self, system, autocast_ctx, *, step: int | None = None) -> dict[str, float]:
        """Run free-generation evaluation.  Called by the Trainer.

        Args:
            system: LMSystem (trunk + head).
            autocast_ctx: torch.amp.autocast context manager.
            step: Current training step (used for directory naming).
        """
        result = evaluate(
            system,
            self.samples,
            split_name=self.split_name,
            sequence_len=self.sequence_len,
            gen_batch_size=self.gen_batch_size,
            save_predictions=False,   # periodic eval — don't save per-eval
        )

        metrics = {
            f"{self.split_name}/exact_accuracy": result.sample.exact_accuracy,
            f"{self.split_name}/magnitude_accuracy": result.sample.magnitude_accuracy,
            f"{self.split_name}/sign_accuracy": result.sample.sign_accuracy,
            f"{self.split_name}/invalid_rate": result.sample.invalid_rate,
            f"{self.split_name}/orbit_consistency": result.orbit.orbit_consistency,
            f"{self.split_name}/whole_orbit_accuracy": result.orbit.whole_orbit_accuracy,
            f"{self.split_name}/n_eligible_orbits": float(result.orbit.n_eligible_orbits),
        }

        # Track best
        if result.sample.exact_accuracy > self.best_exact:
            self.best_exact = result.sample.exact_accuracy
        metrics[f"{self.split_name}/best_exact_accuracy"] = self.best_exact

        # Optionally save JSON for this checkpoint
        if self.save_dir:
            tag = f"step_{step:06d}" if step is not None else "eval"
            step_dir = self.save_dir / tag
            step_dir.mkdir(parents=True, exist_ok=True)
            save_evaluation(result, step_dir / f"eval_{self.split_name}.json")

        return metrics


def evaluation_to_dict(result: EvaluationResult) -> dict[str, Any]:
    """Serialize an EvaluationResult to a JSON-safe dict."""
    return {
        "meta": result.meta,
        "sample_metrics": result.sample.to_dict(),
        "orbit_metrics": result.orbit.to_dict(),
    }


def save_evaluation(result: EvaluationResult, path: str | Path) -> None:
    """Save evaluation result as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evaluation_to_dict(result), indent=2))
