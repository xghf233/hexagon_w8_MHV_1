"""Unconstrained free generation; row-weighted, exact-input-macro and novelty metrics."""

import hashlib
import numpy as np
import torch

from core.model.inference import autoregressive_generate
from .data_contract import require
from .model import validate_attention
from .tokenizer import (EOS_ID, PAD_ID, PROMPT_LENGTH, SEQUENCE_LENGTH, ANSWER_LENGTH,
                        build_layout, decode_generated, encode_prompt, encoding_metadata)


def decode_engine_output(ids):
    ids = list(ids)
    if EOS_ID in ids:
        end = ids.index(EOS_ID)
        # The engine pads completed rows with EOS. No other repairs are allowed.
        if any(t != EOS_ID for t in ids[end + 1:]):
            return None
        ids[end + 1:] = [PAD_ID] * (len(ids) - end - 1)
    try:
        return decode_generated(ids)
    except ValueError:
        return None


@torch.no_grad()
def generate_coefficients(system, conditions, y_types, *, sequence_len=SEQUENCE_LENGTH, batch_size=64):
    """ONLY two feature arrays; no word, position or coefficient-label argument."""
    encoding_metadata(sequence_len)
    require(type(batch_size) is int and batch_size > 0, "Invalid generation batch size")
    require(len(conditions) == len(y_types), "Feature array lengths differ")
    validate_attention(system)
    require(system.config.sequence_len == sequence_len, "Generation/model sequence length mismatch")
    was_training = system.training
    predictions = []
    system.eval()
    try:
        for start in range(0, len(conditions), batch_size):
            prompts = [encode_prompt(c, y) for c, y in zip(
                conditions[start:start + batch_size], y_types[start:start + batch_size], strict=True)]
            prompts = torch.tensor(prompts, dtype=torch.long, device=next(system.parameters()).device)
            require(prompts.shape[1] == PROMPT_LENGTH, "Wrong inference prefix length")
            types = build_layout().classify_token_types(prompts)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                generated = autoregressive_generate(
                    system, prompts, types, max_new_tokens=ANSWER_LENGTH, gen_token_type=2,
                    stop_token=EOS_ID, temperature=0.0, top_k=None, early_stop=True)
            predictions.extend(decode_engine_output(row) for row in generated.cpu().tolist())
    finally:
        system.train(was_training)
    return predictions


def metrics(predictions, truths):
    require(len(predictions) == len(truths), "Prediction/label lengths differ")
    counts = {key: 0 for key in ("n_invalid", "n_exact", "n_magnitude", "n_sign", "n_high", "n_low", "n_valid_zero")}
    for pred, true in zip(predictions, truths, strict=True):
        require(true != 0, "This evaluation protocol is nonzero-target only")
        if pred is None:
            counts["n_invalid"] += 1
            continue
        counts["n_exact"] += int(pred == true)
        counts["n_magnitude"] += int(abs(pred) == abs(true))
        counts["n_sign"] += int(pred != 0 and (pred > 0) == (true > 0))
        counts["n_high"] += int(abs(pred) // 100 == abs(true) // 100)
        counts["n_low"] += int(abs(pred) % 100 == abs(true) % 100)
        counts["n_valid_zero"] += int(pred == 0)
    n = len(truths)
    rates = {f"{key}_accuracy": counts[f"n_{key}"] / n if n else None
             for key in ("exact", "magnitude", "sign", "high", "low")}
    return {"n_samples": n, **counts, **rates,
            "invalid_rate": counts["n_invalid"] / n if n else None,
            "valid_zero_rate": counts["n_valid_zero"] / n if n else None}


def select_rows(dataset, count, seed):
    if count is None:
        return dataset.indices.copy()
    require(type(count) is int and 0 < count <= len(dataset), "Invalid evaluation subset size")
    positions = np.random.Generator(np.random.PCG64(seed)).choice(len(dataset), count, replace=False)
    return dataset.indices[positions]


def checked_rows(dataset, rows):
    rows = np.asarray(rows)
    require(rows.ndim == 1 and np.issubdtype(rows.dtype, np.integer) and len(rows) > 0,
            "Evaluation rows must be a nonempty integer vector")
    require(len(np.unique(rows)) == len(rows) and np.all(np.isin(rows, dataset.indices)),
            "Rows must be unique members of the selected split")
    return rows


def prediction_report(dataset, rows, predictions, *, save_predictions=False):
    rows = checked_rows(dataset, rows)
    truths = dataset.coefficients[rows].tolist()
    overall = metrics(predictions, truths)
    train = dataset.split_indices["train"]  # full frozen train, even for tiny diagnostics
    groups = dataset.input_group_ids[rows]
    families = dataset.family_ids[rows]
    seen = np.isin(groups, dataset.input_group_ids[train])
    seen_family = np.isin(families, dataset.family_ids[train])

    def selected(mask):
        positions = np.flatnonzero(mask)
        return metrics([predictions[i] for i in positions], [truths[i] for i in positions])

    group_metrics = [selected(groups == g) for g in np.unique(groups)]
    macro = {"n_groups": len(group_metrics), "n_samples": len(rows)}
    for key in ("exact_accuracy", "magnitude_accuracy", "sign_accuracy", "invalid_rate",
                "high_accuracy", "low_accuracy", "valid_zero_rate"):
        macro[key] = sum(m[key] for m in group_metrics) / len(group_metrics)
    target_array = np.asarray(truths)
    result = {
        "split": dataset.split, "dataset": dataset.identity(),
        "selection": "full" if len(rows) == len(dataset) else "fixed_subset",
        "row_ids_sha256": hashlib.sha256(rows.astype("<i4").tobytes()).hexdigest(),
        "sample_metrics": overall, "input_group_macro": macro,
        "novelty_reference": "complete frozen train split, not minibatches visited",
        "by_input_novelty": {"seen_input38": selected(seen), "unseen_input38": selected(~seen)},
        "by_family_novelty": {"seen_family": selected(seen_family), "unseen_family": selected(~seen_family)},
        "by_position": {f"{r + 1},{r + 2}": selected(dataset.y_positions[rows, 0] == r) for r in range(2, 7)},
        "by_y_types": {f"{a},{b}": selected(np.all(dataset.y_types[rows] == (a, b), axis=1))
                       for a in range(6, 9) for b in range(6, 9)},
        "by_coefficient": {str(c): selected(target_array == c) for c in sorted(set(truths))},
        "by_magnitude": {str(c): selected(np.abs(target_array) == c) for c in sorted(set(map(abs, truths)))},
        "rare_groups": {"high_zero": selected(np.abs(target_array) < 100),
                        "high_nonzero": selected(np.abs(target_array) >= 100)},
    }
    # Rarity is defined from TRAIN frequencies only; held-out labels never choose it.
    values, counts = np.unique(dataset.coefficients[train], return_counts=True)
    rare_values = values[counts <= 5]
    result["rare_groups"]["train_frequency_le_5"] = selected(np.isin(target_array, rare_values))
    result["rare_groups"]["unseen_target_value_in_train"] = selected(~np.isin(target_array, values))
    if save_predictions:
        result["predictions"] = [{"row": int(r), "input_group": int(g), "true": t, "predicted": p}
                                 for r, g, t, p in zip(rows, groups, truths, predictions, strict=True)]
    return result


def evaluate(system, dataset, rows, *, batch_size=64, save_predictions=False):
    rows = checked_rows(dataset, rows)
    predictions = generate_coefficients(system, dataset.conditions[rows], dataset.y_types[rows],
                                        sequence_len=system.config.sequence_len, batch_size=batch_size)
    report = prediction_report(dataset, rows, predictions, save_predictions=save_predictions)
    report["decoding"] = "greedy_full_vocabulary_base100_fixed2_no_repairs"
    return report
