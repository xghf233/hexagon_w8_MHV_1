"""Greedy free generation and random-row metrics; no orbit grouping or label input."""

from contextlib import nullcontext
import hashlib

import numpy as np
import torch

from core.model.inference import autoregressive_generate
from .model import validate_attention
from .tokenizer import (
    EOS_ID, PAD_ID, PROMPT_LENGTH, SEQUENCE_LENGTH, build_layout,
    decode_generated, encode_prompt, encoding_metadata,
)


def decode_engine_output(ids: list[int]) -> int | None:
    """Normalize ONLY the engine's repeated-EOS batch padding, then strict decode."""
    ids = list(ids)
    if EOS_ID in ids:
        end = ids.index(EOS_ID)
        if any(token != EOS_ID for token in ids[end + 1:]):
            return None
        ids[end + 1:] = [PAD_ID] * (len(ids) - end - 1)
    try:
        return decode_generated(ids)
    except ValueError:
        return None


@torch.no_grad()
def generate_coefficients(system, words, *, sequence_len=SEQUENCE_LENGTH,
                          batch_size=64) -> list[int | None]:
    """Accept words ONLY. Returned None denotes an invalid generated sequence."""
    encoding_metadata(sequence_len)
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("Generation batch size must be a positive integer")
    validate_attention(system)
    if sequence_len != system.config.sequence_len:
        raise ValueError("Generation sequence length differs from the trained configuration")
    device = next(system.parameters()).device
    was_training = system.training
    predictions = []
    system.eval()
    try:
        for start in range(0, len(words), batch_size):
            prompts = torch.tensor([encode_prompt(word) for word in words[start:start + batch_size]],
                                   dtype=torch.long, device=device)
            types = build_layout().classify_token_types(prompts)
            context = (torch.autocast("cuda", dtype=torch.bfloat16)
                       if device.type == "cuda" else nullcontext())
            with context:
                generated = autoregressive_generate(
                    system, prompts, types, max_new_tokens=sequence_len - PROMPT_LENGTH,
                    gen_token_type=2, stop_token=EOS_ID, temperature=0.0,
                    top_k=None, early_stop=True,
                )
            predictions.extend(decode_engine_output(row) for row in generated.cpu().tolist())
    finally:
        system.train(was_training)
    return predictions


def metrics(predictions, truths) -> dict:
    if len(predictions) != len(truths) or not len(truths):
        raise ValueError("Metrics require equally sized, nonempty inputs")
    counts = {"n_samples": len(truths), "n_invalid": 0, "n_exact": 0, "n_magnitude": 0, "n_sign": 0}
    for pred, truth in zip(predictions, truths, strict=True):
        if pred is None:
            counts["n_invalid"] += 1
            continue
        counts["n_exact"] += int(pred == truth)
        counts["n_magnitude"] += int(abs(pred) == abs(truth))
        # Zero has neither positive nor negative sign (truths are nonzero).
        counts["n_sign"] += int((pred > 0) == (truth > 0) and pred != 0)
    n = counts["n_samples"]
    return {**counts, "exact_accuracy": counts["n_exact"] / n,
            "magnitude_accuracy": counts["n_magnitude"] / n,
            "sign_accuracy": counts["n_sign"] / n, "invalid_rate": counts["n_invalid"] / n}


def select_rows(dataset, count: int | None, seed: int) -> np.ndarray:
    """Fixed random subset of the stored split; None means the entire split."""
    if count is None:
        return dataset.indices.copy()
    if type(count) is not int or not 0 < count <= len(dataset):
        raise ValueError("Evaluation subset size is outside the split")
    selected = np.random.Generator(np.random.PCG64(seed)).choice(len(dataset), count, replace=False)
    return dataset.indices[selected]


def evaluate(system, dataset, rows: np.ndarray, *, batch_size=64, save_predictions=False) -> dict:
    rows = np.asarray(rows)
    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer) or not len(rows):
        raise ValueError("Evaluation rows must be a nonempty integer vector")
    if len(np.unique(rows)) != len(rows) or not np.all(np.isin(rows, dataset.indices)):
        raise ValueError("Evaluation rows must be unique members of the selected split")
    predictions = generate_coefficients(system, dataset.words[rows],
        sequence_len=system.config.sequence_len, batch_size=batch_size)
    truths = dataset.coefficients[rows].tolist()
    by_coefficient = {}
    by_magnitude = {}
    for group, key_function in ((by_coefficient, int), (by_magnitude, abs)):
        for value in sorted({key_function(t) for t in truths}):
            positions = [i for i, truth in enumerate(truths) if key_function(truth) == value]
            group[str(value)] = metrics([predictions[i] for i in positions], [truths[i] for i in positions])
    result = {"split": dataset.split, "dataset": dataset.identity(),
              "selection": "full" if len(rows) == len(dataset) else "fixed_subset",
              "row_ids_sha256": hashlib.sha256(rows.astype("<i4").tobytes()).hexdigest(),
              "decoding": "greedy_unconstrained_full_vocabulary", "sample_metrics": metrics(predictions, truths),
              "by_coefficient": by_coefficient, "by_magnitude": by_magnitude}
    if save_predictions:
        result["predictions"] = [{"row": int(row), "true": true, "predicted": pred}
                                 for row, true, pred in zip(rows, truths, predictions, strict=True)]
    return result
