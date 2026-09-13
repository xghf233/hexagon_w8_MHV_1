"""First execution ONLY: CUDA contracts + continuous20 versus resume10+10.

Host source parsing, integer preencoding and RNG bookkeeping are necessary GPU
workflow operations. This module provides no CPU/MPS model path or final test.
"""

import argparse
from copy import copy
import gc
import inspect
import json
from pathlib import Path
import platform
import time

from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
import numpy as np
import torch

from core.model.inference import autoregressive_generate
from core.model.kv_cache import KVCache
from core.training.lr_schedulers import get_lr_multiplier
from .checkpoint import capture_rng, code_hashes, restore_rng
from .convert_data import prepare_data
from .data_contract import N_ROWS, SPLIT_COUNTS, outside_repository, separate_path, require, write_json
from .dataset import HexagonDataset, HexagonDataLoader
from .eval_checkpoint import load_for_evaluation
from .evaluator import decode_engine_output
from .model import build_gpu_system, attach_attention, validate_attention, EXPECTED_PARAMETERS
from .runtime import require_cuda_server
from .tokenizer import (AssembledSequence, assemble_batch, assemble_sequence, next_token_batch,
                        encode_prompt, encode_coefficient, decode_coefficient, decode_generated,
                        build_layout, encoding_metadata, BOS_ID, QUERY_ID, CONTEXT_ID, ANSWER_ID,
                        PLUS_ID, MINUS_ID, NUM_OFFSET, EOS_ID, PAD_ID, ANSWER_LENGTH,
                        PROMPT_LENGTH, SEQUENCE_LENGTH, VOCAB_SIZE)
from .train import run_training, validate_config

BATCH_KEYS = ("idx", "targets", "token_types", "target_types", "loss_weights")


def recipe(name, data_dir, metadata_sha256, output_dir):
    with initialize_config_module(config_module=f"{__package__}.configs", version_base=None):
        config = compose(config_name=name)
    config.data_dir = str(data_dir)
    config.metadata_sha256 = metadata_sha256
    config.output_dir = str(output_dir)
    result = OmegaConf.to_container(config, resolve=True)
    validate_config(result)
    return result


def ensure(condition, message):
    require(bool(condition), message)


def must_reject(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except (ValueError, TypeError):
        return
    raise ValueError(f"Invalid input accepted by {function.__name__}")


def equal_batch(left, right):
    for key in BATCH_KEYS:
        ensure(left[key].device.type == right[key].device.type == "cuda", "Batch comparison must use CUDA")
        ensure(torch.equal(left[key], right[key]), f"Resumed batch differs: {key}")
    ensure(left["state_dict"] == right["state_dict"], "Next batch sampler states differ")


def encoding_checks(train, device):
    assembled = assemble_batch(train.conditions, train.y_types, train.coefficients)
    ensure(assembled.tokens.shape == (N_ROWS, SEQUENCE_LENGTH), "Wrong full pool encoding shape")
    for start in range(0, N_ROWS, 1024):
        end = min(start + 1024, N_ROWS)
        sequence = AssembledSequence(assembled.tokens[start:end], assembled.token_types[start:end],
                                     assembled.loss_weights[start:end])
        batch = {k: v.to(device) for k, v in next_token_batch(sequence).items()}
        raw = sequence.tokens.to(device)
        truth = torch.tensor(np.array(train.coefficients[start:end]), dtype=torch.long, device=device)
        conditions = torch.tensor(np.array(train.conditions[start:end]), dtype=torch.long, device=device)
        ys = torch.tensor(np.array(train.y_types[start:end]), dtype=torch.long, device=device)
        ensure(batch["idx"].shape == (end - start, 127), "Wrong shifted length")
        expected = torch.zeros_like(batch["targets"], dtype=torch.bool)
        expected[:, 113:117] = True
        ensure(torch.equal(batch["targets"] != -1, expected), "Only shifted 113..116 may be supervised")
        ensure(torch.equal(batch["loss_weights"], expected.float()), "Exactly four unit loss weights required")
        ensure(torch.all(raw[:, 0] == BOS_ID) and torch.all(raw[:, 1] == QUERY_ID)
               and torch.all(raw[:, 4] == CONTEXT_ID) and torch.all(raw[:, 113] == ANSWER_ID),
               "Prompt delimiters changed")
        ensure(torch.equal(raw[:, 2:4], ys), "Ordered y query changed")
        triples = raw[:, 5:113].reshape(-1, 36, 3)
        ensure(torch.equal(triples[:, :, 0], torch.where(conditions < 0, MINUS_ID, PLUS_ID)), "Condition signs mismatch")
        ensure(torch.equal(triples[:, :, 1] - NUM_OFFSET, conditions.abs() // 100), "Condition high blocks mismatch")
        ensure(torch.equal(triples[:, :, 2] - NUM_OFFSET, conditions.abs() % 100), "Condition low blocks mismatch")
        ensure(torch.all(raw[:, 118:] == PAD_ID), "Expected ten raw PAD slots")
        ensure(torch.equal(batch["targets"][:, 113], torch.where(truth < 0, MINUS_ID, PLUS_ID)), "Target signs mismatch")
        ensure(torch.equal(batch["targets"][:, 114] - NUM_OFFSET, truth.abs() // 100), "Target high blocks mismatch")
        ensure(torch.equal(batch["targets"][:, 115] - NUM_OFFSET, truth.abs() % 100), "Target low blocks mismatch")
        ensure(torch.all(batch["targets"][:, 116] == EOS_ID), "Wrong EOS target")
        ensure(torch.all(batch["target_types"][:, 113:116] == 2)
               and torch.all(batch["target_types"][:, 116] == 1), "Target token types mismatch")
        ensure(torch.all(batch["token_types"][:, 2:4] == 0)
               and torch.all(batch["token_types"][:, 5:113] == 2), "Feature token types mismatch")
        ensure(not torch.any(raw[:, :PROMPT_LENGTH] < 6), "Ordinary background letters leaked into input")
    for c in (-9999, -1920, -240, -100, -1, 0, 1, 99, 100, 216, 1920, 9999):
        ensure(decode_coefficient(encode_coefficient(c)) == c, "Integer encoding round-trip failed")
    for invalid in ([PLUS_ID, NUM_OFFSET, EOS_ID], [MINUS_ID, NUM_OFFSET, NUM_OFFSET, EOS_ID],
                    [PLUS_ID, NUM_OFFSET, NUM_OFFSET + 1, PAD_ID],
                    [PLUS_ID, NUM_OFFSET, NUM_OFFSET, EOS_ID, PLUS_ID]):
        must_reject(decode_generated, invalid)
    for value in (-10000, 10000, 1.0, True):
        must_reject(encode_coefficient, value)
    ensure(decode_engine_output([EOS_ID] * 4) is None, "Early EOS is not a valid coefficient")
    ensure(decode_engine_output([PLUS_ID, NUM_OFFSET, NUM_OFFSET, EOS_ID]) == 0, "Canonical zero must be valid")
    ensure(set(inspect.signature(encode_prompt).parameters) == {"conditions", "y_types"}, "Prompt API is too broad")
    must_reject(encode_prompt, [0] * 35, [6, 7])
    must_reject(encode_prompt, [0] * 36, [5, 7])
    must_reject(encode_prompt, [0] * 36, [6, 7], y_positions=[2, 3])
    row = int(train.indices[0])
    scalar = assemble_sequence(train.conditions[row], train.y_types[row], int(train.coefficients[row]))
    for key in ("tokens", "token_types", "loss_weights"):
        ensure(torch.equal(getattr(scalar, key).to(device), getattr(assembled, key)[row].to(device)),
               f"Scalar/vector encoding disagreement: {key}")
    # Invariance fixture: changing every provenance field cannot alter the model path.
    fixture = copy(train)
    fixture.indices = train.indices[:8]
    original = fixture.preencode().tokens.to(device)
    fixture.words = np.zeros_like(train.words)
    fixture.y_positions = np.zeros_like(train.y_positions)
    fixture.source_row_ids = np.zeros_like(train.source_row_ids)
    fixture.family_ids = np.zeros_like(train.family_ids)
    fixture.input_group_ids = np.zeros_like(train.input_group_ids)
    ensure(torch.equal(original, fixture.preencode().tokens.to(device)), "Provenance entered model encoding")
    # Synthetic ordered nonzero slots exercise transpose/order even if a real table is symmetric.
    synthetic = assemble_sequence(list(range(36)), [8, 6], -101).tokens.to(device)
    ensure(torch.equal(synthetic[7:113:3] - NUM_OFFSET, torch.arange(36, device=device)), "C36 slot order changed")
    ensure(torch.equal(synthetic[2:4], torch.tensor([8, 6], device=device)), "Y order was sorted or symmetrized")
    return {"rows_checked_on_cuda": N_ROWS, "scalar_batch_equal": True,
            "provenance_invariance": True, "strict_decoder_negative_cases": True}


@torch.no_grad()
def generation_trace(system, prompts, *, cached):
    """Reference/diagnostic paths match the engine's post-EOS batch padding."""
    tokens = prompts.clone()
    types = build_layout().classify_token_types(tokens)
    cache = KVCache.for_model(system.config, len(tokens), PROMPT_LENGTH + ANSWER_LENGTH) if cached else None
    finished = torch.zeros(len(tokens), dtype=torch.bool, device=tokens.device)
    output, logits_trace = [], []
    for step in range(ANSWER_LENGTH):
        hidden = system.trunk(tokens, token_types=types, kv_cache=cache)
        logits = system.head(hidden[:, -1:]).squeeze(1)
        logits_trace.append(logits.float())
        nxt = logits.argmax(-1, keepdim=True)
        finished |= nxt[:, 0] == EOS_ID
        nxt[finished] = EOS_ID
        output.append(nxt)
        if cached:
            tokens, types = nxt, torch.full_like(nxt, 2)
        else:
            tokens = torch.cat((tokens, nxt), dim=1)
            types = torch.cat((types, torch.full_like(nxt, 2)), dim=1)
    return torch.cat(output, dim=1), torch.stack(logits_trace, dim=1)


def cuda_contract_checks(config, train):
    device = require_cuda_server()
    report = encoding_checks(train, device)
    loader = HexagonDataLoader(train, 8, seed=config["seed"], device=device)
    first = next(loader)
    state = loader.state_dict()
    expected = next(loader)
    restored = HexagonDataLoader(train, 8, seed=config["seed"], device=device)
    restored.set_state(state)
    equal_batch(expected, next(restored))
    # Also cross a shuffle epoch boundary, not just the first few rows.
    state["rows_consumed"] = (len(train) // 8) * 8
    loader.set_state(state)
    restored.set_state(state)
    equal_batch(next(loader), next(restored))
    for key, bad in (("batch_size", 9), ("seed", config["seed"] + 1)):
        broken = {**state, "contract": {**state["contract"], key: bad}}
        must_reject(restored.set_state, broken)
    pilot_schedule = {"type": "linear", "max_steps": 5000, "warmup_steps": 200,
                      "warmdown_ratio": 0.2, "final_lr_frac": 0.0}
    for step, value in ((0, 0.005), (199, 1), (4000, 1), (4999, 0.001), (5000, 0)):
        ensure(abs(get_lr_multiplier(step, pilot_schedule) - value) < 1e-12, "Pilot scheduler boundary mismatch")
    system = build_gpu_system(config)
    ensure(all(p.device.type == "cuda" for p in system.parameters()), "CPU model parameter found")
    # A zero residual branch could make a broken mask look correct. Discard this fixture after diagnostics.
    with torch.no_grad():
        for block in system.trunk.blocks:
            block.attn.c_proj.weight.normal_(std=0.02)
            block.mlp.c_proj.weight.normal_(std=0.02)
        system.head.lm_head.weight.normal_(std=0.02)
    system.eval()
    x, types = first["idx"], first["token_types"]
    changed = x.clone()
    changed[:, 114] = torch.where(changed[:, 114] == PLUS_ID, MINUS_ID, PLUS_ID)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        h1 = system.trunk(x, token_types=types)
        h2 = system.trunk(changed, token_types=types)
        torch.testing.assert_close(h1[:, :114], h2[:, :114], rtol=0, atol=0)
        from projects.amplitude_symbol.blocks.attention import attach_word_bidirectional_attention
        attach_word_bidirectional_attention(system, SEQUENCE_LENGTH, word_start=0, word_end=115)
        bad1 = system.trunk(x, token_types=types)
        bad2 = system.trunk(changed, token_types=types)
        ensure(not torch.equal(bad1[:, :114], bad2[:, :114]), "Broken-mask negative control was ineffective")
        attach_attention(system)
        validate_attention(system)
        pad_changed, pad_types = x.clone(), types.clone()
        pad_changed[:, 118:], pad_types[:, 118:] = NUM_OFFSET + 99, 2
        hpad = system.trunk(pad_changed, token_types=pad_types)
        torch.testing.assert_close(h1[:, :118], hpad[:, :118], rtol=0, atol=0)
        logits = system.head(h1)
        ensure(logits.shape == (8, 127, VOCAB_SIZE) and torch.isfinite(logits).all(), "Invalid logits")
        prompt = x[:, :PROMPT_LENGTH]
        engine = autoregressive_generate(system, prompt, types[:, :PROMPT_LENGTH],
                                         max_new_tokens=4, gen_token_type=2, stop_token=EOS_ID,
                                         temperature=0.0, early_stop=False)
        cached, cached_logits = generation_trace(system, prompt, cached=True)
        reference, full_logits = generation_trace(system, prompt, cached=False)
        ensure(torch.equal(engine, cached) and torch.equal(cached, reference),
               "Full-prefix/KV-cache greedy outputs differ")
        # bf16 kernels at different query lengths may round differently. Require
        # strict greedy equality AND bounded logits (~1-2 bf16 ULP near magnitude 1).
        torch.testing.assert_close(cached_logits, full_logits, rtol=1e-2, atol=1e-2)
        max_logit_delta = float((cached_logits - full_logits).abs().max())
    system.train()
    system.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = system.loss(first)
    ensure(torch.isfinite(loss), "Nonfinite diagnostic loss")
    reference_loss = torch.nn.functional.cross_entropy(
        logits[:, 113:117].reshape(-1, VOCAB_SIZE), first["targets"][:, 113:117].reshape(-1))
    # Also covers optional fused CE; bf16 kernel differences are bounded, not exact.
    torch.testing.assert_close(loss.detach(), reference_loss, rtol=1e-2, atol=1e-2)
    loss.backward()
    gradients = [p.grad for p in system.parameters() if p.grad is not None]
    ensure(gradients and all(torch.isfinite(g).all() for g in gradients), "Missing/nonfinite gradients")
    report.update({
        "status": "passed", "parameters": EXPECTED_PARAMETERS, "encoding": encoding_metadata(),
        "loss_shift": "[113,117)", "attention": "prefix [0,114) bidirectional; answer causal",
        "no_answer_or_PAD_leakage": True, "bad_mask_negative_control": True,
        "finite_forward_backward": True, "diagnostic_loss": float(loss.detach()),
        "four_target_mean_loss_checked": True, "loss_rtol": 1e-2, "loss_atol": 1e-2,
        "cache_full_generation_equal": True, "cache_full_logits_max_abs_difference": max_logit_delta,
        "cache_logit_rtol": 1e-2, "cache_logit_atol": 1e-2,
        "loader_resume_and_epoch_boundary_equal": True, "pilot_lr_boundaries_checked": True})
    return report


def compare_state_trees(left, right, *, path="state", tolerance=False):
    """Optimizer tensor comparisons happen on CUDA, including scalar step tensors."""
    if isinstance(left, torch.Tensor):
        ensure(isinstance(right, torch.Tensor) and left.dtype == right.dtype and left.shape == right.shape,
               f"Tensor schema differs: {path}")
        a, b = left.to("cuda"), right.to("cuda")
        torch.testing.assert_close(a, b, rtol=1e-5 if tolerance else 0, atol=1e-6 if tolerance else 0)
        return float((a.double() - b.double()).abs().max()) if a.numel() else 0.0
    ensure(type(left) is type(right), f"State type differs: {path}")
    if isinstance(left, dict):
        ensure(left.keys() == right.keys(), f"State keys differ: {path}")
        return max((compare_state_trees(left[k], right[k], path=f"{path}.{k}", tolerance=tolerance)
                    for k in left), default=0.0)
    if isinstance(left, (tuple, list)):
        ensure(len(left) == len(right), f"State length differs: {path}")
        return max((compare_state_trees(a, b, path=f"{path}[{i}]", tolerance=tolerance)
                    for i, (a, b) in enumerate(zip(left, right, strict=True))), default=0.0)
    ensure(left == right, f"State scalar differs: {path}")
    return 0.0


def compare_final_checkpoints(continuous, resumed, train, config):
    from core.model.checkpoint_manager import load_checkpoint_dcp
    from core.training.optim import build_optimizers
    left, lm = load_for_evaluation(continuous, train)
    right, rm = load_for_evaluation(resumed, train)
    ensure(all(p.device.type == "cuda" for model in (left, right) for p in model.parameters()), "CPU model found")
    delta = compare_state_trees(left.state_dict(), right.state_dict(), tolerance=True)
    left_opts = build_optimizers(left, config["optimizer"])
    right_opts = build_optimizers(right, config["optimizer"])
    load_checkpoint_dcp(str(continuous), left, load_optimizer=True, optimizers=left_opts)
    load_checkpoint_dcp(str(resumed), right, load_optimizer=True, optimizers=right_opts)
    validate_attention(left)
    validate_attention(right)
    ensure(all(opt.state for opt in left_opts + right_opts), "Optimizer states were not restored")
    opt_delta = compare_state_trees([opt.state_dict() for opt in left_opts],
                                   [opt.state_dict() for opt in right_opts], tolerance=True)
    ensure(lm["dataloader_state"] == rm["dataloader_state"], "Final sampler states differ")
    ensure(lm["rng_state"] == rm["rng_state"], "Final RNG states differ")
    restore_rng(lm["rng_state"])
    ensure(capture_rng() == lm["rng_state"], "RNG restoration did not round-trip")
    loaders = [HexagonDataLoader(train, config["device_batch_size"], seed=config["seed"], device="cuda")
               for _ in range(2)]
    for loader, meta in zip(loaders, (lm, rm), strict=True):
        loader.set_state(meta["dataloader_state"])
    equal_batch(next(loaders[0]), next(loaders[1]))
    return {"resume_parameters_max_abs_difference": delta, "resume_optimizer_max_abs_difference": opt_delta,
            "resume_rtol": 1e-5, "resume_atol": 1e-6, "resume_rng_equal": True,
            "resume_actual_next_batch_equal": True, "resume_optimizer_state_equal_within_tolerance": True}


def run(args):
    device = require_cuda_server()
    output, data_dir = outside_repository(args.output), outside_repository(args.data_dir)
    separate_path(output, data_dir)
    require(not output.exists(), "Choose a NEW smoke directory")
    if args.source:
        separate_path(output, args.source.expanduser().resolve().parent)
    if data_dir.exists():
        require(not args.source and args.metadata_sha256, "Existing data requires trusted digest, no --source")
        digest = args.metadata_sha256
    else:
        require(args.source and not args.metadata_sha256, "New data requires --source and no supplied digest")
        digest = prepare_data(args.source, data_dir)["metadata_sha256"]
    train = HexagonDataset(data_dir, expected_metadata_sha256=digest)
    separate_path(output, Path(train.metadata["source"]["path"]).parent)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write_json(output / "environment.json", {
        "python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device), "device": str(device),
        "metadata_sha256": digest, "code_sha256": code_hashes(), "standalone_cpu_tests": False})
    config = recipe("smoke", data_dir, digest, output / "continuous")
    write_json(output / "cuda_contract_checks.json", cuda_contract_checks(config, train))
    gc.collect()
    torch.cuda.empty_cache()
    continuous = run_training(config)
    gc.collect()
    torch.cuda.empty_cache()
    prefix_config = recipe("smoke", data_dir, digest, output / "prefix")
    prefix_config["stop_after_steps"] = 10
    prefix = run_training(prefix_config)
    gc.collect()
    torch.cuda.empty_cache()
    resumed_config = recipe("smoke", data_dir, digest, output / "resumed")
    resumed_config["resume_from"] = prefix["last_checkpoint"]
    resumed = run_training(resumed_config)
    gc.collect()
    torch.cuda.empty_cache()
    comparison = compare_final_checkpoints(continuous["last_checkpoint"], resumed["last_checkpoint"], train, config)
    left_val = json.loads((output / "continuous/val_subset_000020.json").read_text(encoding="utf-8"))
    right_val = json.loads((output / "resumed/val_subset_000020.json").read_text(encoding="utf-8"))
    ensure(left_val == right_val, "Continuous/resumed generation reports differ")
    report = {
        "status": "passed_gpu_smoke", "elapsed_seconds": time.monotonic() - started,
        "parameters": EXPECTED_PARAMETERS, "split_counts": SPLIT_COUNTS, "metadata_sha256": digest,
        "model_updates": 40, "continuous_steps": continuous["completed_steps"],
        "resumed_steps": resumed["completed_steps"], **comparison,
        "test_evaluated": False, "tiny_overfit_completed": False, "pilot_completed": False,
        "standalone_cpu_tests": False, "host_data_io_checks": True}
    write_json(output / "gpu_smoke_report.json", report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--metadata-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        print(json.dumps({"status": "failed_gpu_smoke", "error": str(exc)}, ensure_ascii=False))
        raise  # preserve every existing artifact; never relax checks or auto-run the pilot


if __name__ == "__main__":
    main()
