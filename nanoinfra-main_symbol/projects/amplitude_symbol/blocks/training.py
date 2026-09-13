"""Reusable training orchestration for amplitude symbol experiments.

Each experiment imports ``run_training()`` and passes its own
``setup_model`` callback — everything else (data, split, eval, Trainer)
is shared.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

import torch

from core.model.gpt import GPT, GPTConfig
from core.training.model_setup import build_system
from core.training.trainer import Trainer, create_optimizers
from core.utils import print0

from ..dataset import AmplitudeDataSource, AmplitudeDataLoader
from ..evaluator import FreeGenEvaluator
from ..paths import resolve_data_path, resolve_manifest_path, resolve_upstream_root
from ..prepare_data import load_symbol, _get_git_commit
from ..split import generate_splits, get_split_coeffs, load_manifest
from ..tokenizer import N_TOKEN_TYPES, VOCAB_SIZE


def _output_dir(config: dict[str, Any]) -> Path:
    """Return the configured run directory without embedding a project name."""
    run_name = config.get("wandb", {}).get("name", "run")
    raw = config.get("output_dir") or f"outputs/{run_name}"
    return Path(os.path.expandvars(str(raw))).expanduser()


def run_training(
    config: dict[str, Any],
    *,
    setup_model: Callable[[Any, dict[str, Any]], None] | None = None,
) -> Any:
    """Standard training loop shared by all amplitude-symbol experiments.

    Args:
        config: Resolved Hydra config dict (OmegaConf.to_container).
        setup_model: Optional callback ``(system, config) -> None`` called
            AFTER model assembly but BEFORE optimizer creation.  Experiments
            use this to attach attention masks, swap heads, etc.

    Returns:
        The trained LMSystem.
    """

    assert torch.cuda.is_available(), "CUDA is required for training"
    torch.cuda.set_device(0)

    # ------------------------------------------------------------------
    # Model config
    # ------------------------------------------------------------------
    model_cfg = config["model"]
    gpt_config = GPTConfig(
        sequence_len=config["sequence_len"],
        vocab_size=VOCAB_SIZE,
        n_layer=model_cfg["n_layer"],
        n_embd=model_cfg["n_embd"],
        n_head=model_cfg["n_head"],
        n_kv_head=model_cfg["n_kv_head"],
        n_token_types=N_TOKEN_TYPES,
    )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    seq_len = config["sequence_len"]
    seed = config["seed"]

    data_cfg = config.get("data", {}) or {}
    data_path = resolve_data_path(data_cfg.get("path"))
    upstream_root = resolve_upstream_root(data_cfg.get("aiamplitudes_root"))
    manifest_path = resolve_manifest_path(data_cfg.get("manifest_path"))

    symb = load_symbol(data_path)

    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
    else:
        print0("  Generating splits (first run)...")
        commit = _get_git_commit(upstream_root)
        manifest = generate_splits(
            symb, seed=seed,
            data_path=str(data_path),
            aiamplitudes_commit=commit,
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))

    train_pairs = get_split_coeffs(manifest, symb, "train", "orbit_grouped")

    n_overfit = config.get("overfit_n_samples")
    if n_overfit is not None:
        train_pairs = train_pairs[:n_overfit]
        print0(f"  Overfit mode: {len(train_pairs)} samples")
    else:
        print0(f"  Training samples: {len(train_pairs)}")

    # ------------------------------------------------------------------
    # Data pipeline
    # ------------------------------------------------------------------
    source = AmplitudeDataSource(train_pairs, sequence_len=seq_len, seed=seed, device="cuda")
    loader = AmplitudeDataLoader(source, batch_size=config["device_batch_size"])

    # ------------------------------------------------------------------
    # Model assembly
    # ------------------------------------------------------------------
    use_compile = config.get("compile", False)
    print0(f"\n  Model: {gpt_config.n_layer} layers, {gpt_config.n_embd} dim, "
           f"{gpt_config.n_head} heads, {VOCAB_SIZE} vocab, "
           f"{N_TOKEN_TYPES} token types"
           f"{' (compiled)' if use_compile else ''}")

    setup = build_system(GPT, gpt_config, use_compile=use_compile, seed=seed)
    system = setup["system"]
    system.attention_config = {"mode": "causal"}

    n_params = sum(p.numel() for p in system.parameters())
    print0(f"  Parameters: {n_params:,}")

    # --- experiment hook ---
    if setup_model is not None:
        setup_model(system, config)

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------
    world_size = setup["world_size"]
    optimizers = create_optimizers(system, config["optimizer"], world_size)

    # ------------------------------------------------------------------
    # Evaluators
    # ------------------------------------------------------------------
    evaluators = []
    eval_cfg = config.get("evaluation", {})
    if eval_cfg.get("enabled", False):
        eval_pairs = get_split_coeffs(manifest, symb,
                                       eval_cfg.get("split", "val"),
                                       "orbit_grouped")
        n_eval = min(eval_cfg.get("n_samples", 256), len(eval_pairs))
        eval_pairs = eval_pairs[:n_eval]

        fev = FreeGenEvaluator(
            eval_pairs,
            interval_steps=eval_cfg.get("interval_steps", 2000),
            sequence_len=seq_len,
            gen_batch_size=eval_cfg.get("gen_batch_size", 64),
            split_name=eval_cfg.get("split", "val"),
            save_dir=str(_output_dir(config)),
        )
        evaluators.append(fev)
        print0(f"  Periodic eval: every {fev.interval_steps} steps on {n_eval} samples")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer = Trainer(
        system=system,
        optimizers=optimizers,
        dataloader=loader,
        config=config,
        rank=setup["rank"],
        world_size=world_size,
        evaluators=evaluators,
    )
    trainer.train()

    print0(f"\n{'=' * 72}")
    print0(f"  Training complete")
    print0(f"{'=' * 72}")

    # ------------------------------------------------------------------
    # Post-training
    # ------------------------------------------------------------------
    if config.get("verify", False):
        print0("\n  Running overfit verification...")
        _verify_overfit_fn(system, train_pairs, sequence_len=seq_len)

    if eval_cfg.get("enabled", False):
        _run_evaluation_fn(system, manifest, symb, config, eval_cfg)

    return system


# ======================================================================
# Post-training helpers (thin wrappers kept here so experiments share them)
# ======================================================================

@torch.no_grad()
def _verify_overfit_fn(system, samples, sequence_len=32):
    from core.model.inference import autoregressive_generate
    from ..tokenizer import BOS_ID, COEFF_ID, EOS_ID, WORD_OFFSET, WORD_TOKEN_IDS
    from ..evaluator import strict_decode_coefficient

    system.eval()
    device = next(system.parameters()).device

    correct = 0
    for word, true_coeff in samples:
        prompt_ids = [BOS_ID]
        prompt_types = [1]
        for ch in word:
            prompt_ids.append(WORD_OFFSET + WORD_TOKEN_IDS[ch])
            prompt_types.append(0)
        prompt_ids.append(COEFF_ID)
        prompt_types.append(1)

        prompt = torch.tensor([prompt_ids], device=device)
        ptypes = torch.tensor([prompt_types], device=device)

        max_new = 32 - len(prompt_ids)
        gen_ids = autoregressive_generate(
            system, prompt, ptypes,
            max_new_tokens=max_new,
            gen_token_type=2,
            stop_token=EOS_ID,
            temperature=0.0,
            top_k=None,
            early_stop=True,
        )

        pred_coeff = strict_decode_coefficient(gen_ids[0].tolist())
        if pred_coeff == true_coeff:
            correct += 1

    acc = correct / len(samples) if samples else 0.0
    print0(f"  Overfit verification: {correct}/{len(samples)} exact ({acc*100:.1f}%)")
    return acc, correct, len(samples)


def _run_evaluation_fn(system, manifest, symb, config, eval_cfg):
    from ..evaluator import evaluate, save_evaluation

    split_name = eval_cfg.get("split", "val")
    n_eval_samples = min(eval_cfg.get("n_samples", 256),
                         manifest["orbit_grouped"][split_name]["n_samples"])
    eval_pairs = get_split_coeffs(manifest, symb, split_name, "orbit_grouped")
    eval_pairs = eval_pairs[:n_eval_samples]

    print0(f"\n  Evaluating on {split_name} split ({len(eval_pairs)} samples)...")

    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = evaluate(
        system, eval_pairs,
        split_name=split_name,
        sequence_len=config["sequence_len"],
        gen_batch_size=eval_cfg.get("gen_batch_size", 64),
        save_predictions=eval_cfg.get("save_predictions", False),
        predictions_dir=str(output_dir) if eval_cfg.get("save_predictions", False) else None,
    )

    eval_path = output_dir / f"evaluation_{split_name}.json"
    save_evaluation(result, eval_path)
    print0(f"  Evaluation saved to {eval_path}")

    sm = result.sample
    om = result.orbit
    print0(f"  Sample: exact={sm.exact_accuracy:.4f}  mag={sm.magnitude_accuracy:.4f}  "
           f"sign={sm.sign_accuracy:.4f}  invalid={sm.invalid_rate:.4f}")
    print0(f"  Orbit:  consistency={om.orbit_consistency:.4f}  "
           f"whole={om.whole_orbit_accuracy:.4f}  "
           f"(eligible={om.n_eligible_orbits}, singleton={om.n_singleton_orbits})")
