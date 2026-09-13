"""One GPU-only smoke workflow: host data I/O, CUDA checks, 20 vs 10+10 updates.

No standalone CPU test suite, no CPU model fallback, no test-set evaluation.
All paths must be new, outside the code repository and immutable source directory.
"""

import argparse
import gc
import json
from pathlib import Path
import platform
import time

from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
import numpy as np
import torch

from core.model.inference import autoregressive_generate
from .checkpoint import code_hashes, write_json
from .convert_data import prepare_data
from .data_contract import outside_repository
from .dataset import HexagonDataset, HexagonDataLoader
from .eval_checkpoint import load_for_evaluation
from .model import build_gpu_system, attach_attention, validate_attention, EXPECTED_PARAMETERS
from .runtime import require_cuda_server
from .tokenizer import (assemble_batch, next_token_batch, encode_coefficient, decode_coefficient,
                        decode_generated, build_layout, encoding_metadata,
                        PLUS_ID, MINUS_ID, NUM_OFFSET, EOS_ID, PAD_ID, ANSWER_LENGTH)
from .train import run_training, validate_config


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
    if not bool(condition):
        raise ValueError(message)


@torch.no_grad()
def full_forward_generation(system, prompts):
    """CUDA reference for the cached greedy engine, with identical generated types."""
    tokens = prompts.clone()
    types = build_layout().classify_token_types(tokens)
    finished = torch.zeros(tokens.shape[0], dtype=torch.bool, device=tokens.device)
    output = []
    for step in range(ANSWER_LENGTH):
        hidden = system.trunk(tokens, token_types=types)
        nxt = system.head(hidden[:, -1:]).squeeze(1).argmax(-1, keepdim=True)
        finished |= nxt[:, 0] == EOS_ID
        nxt[finished] = EOS_ID
        output.append(nxt)
        if step + 1 < ANSWER_LENGTH:
            tokens = torch.cat((tokens, nxt), dim=1)
            types = torch.cat((types, torch.full_like(nxt, 2)), dim=1)
    return torch.cat(output, dim=1)


def cuda_contract_checks(config, train):
    device = require_cuda_server()
    # Formatting is prepared on the host; all tensor arithmetic comparisons are CUDA.
    assembled = assemble_batch(train.words, train.coefficients)
    batch = {k: v.to(device) for k, v in next_token_batch(assembled).items()}
    truth = torch.tensor(np.array(train.coefficients), dtype=torch.long, device=device)
    ensure(batch["idx"].shape == (11208, 15), "Incorrect full 0y batch shape")
    expected_mask = torch.zeros_like(batch["targets"], dtype=torch.bool)
    expected_mask[:, 9:13] = True
    ensure(torch.equal(batch["targets"] != -1, expected_mask), "Loss shift must supervise indices 9..12")
    ensure(torch.equal(batch["loss_weights"], expected_mask.float()), "Four equally weighted answers required")
    ensure(torch.all(batch["idx"][:, 0] == 9) and torch.all(batch["idx"][:, 9] == 10), "Prompt delimiters mismatch")
    source_words = torch.tensor(np.array(train.words), dtype=torch.long, device=device)
    ensure(torch.equal(batch["idx"][:, 1:9], source_words), "Native word IDs changed during encoding")
    ensure(torch.all(batch["idx"][:, 14] == PAD_ID), "Trailing input slot must be PAD")
    ensure(torch.all(batch["token_types"][:, 1:9] == 0), "Word type mismatch")
    ensure(torch.equal(batch["targets"][:, 9], torch.where(truth < 0, MINUS_ID, PLUS_ID)), "Sign mismatch")
    ensure(torch.equal(batch["targets"][:, 10] - NUM_OFFSET, truth.abs() // 100), "High-block mismatch")
    ensure(torch.equal(batch["targets"][:, 11] - NUM_OFFSET, truth.abs() % 100), "Low-block mismatch")
    ensure(torch.all(batch["targets"][:, 12] == EOS_ID), "EOS target mismatch")
    ensure(torch.all(batch["target_types"][:, 9:12] == 2), "Number target types mismatch")
    ensure(torch.all(batch["target_types"][:, 12] == 1), "EOS must have control target type")
    ensure(int(torch.sum(truth == 1920)) == 6, "All six 1920 records must survive")
    # Pure syntax checks are part of this CUDA-gated workflow, never a CPU model run.
    for c in (-9999, -1920, -156, -100, -1, 0, 1, 99, 100, 1920, 9999):
        ensure(decode_coefficient(encode_coefficient(c)) == c, "Fixed-base100 round trip")
    for bad in ([PLUS_ID, NUM_OFFSET + 1, EOS_ID],
                [MINUS_ID, NUM_OFFSET, NUM_OFFSET, EOS_ID],
                [PLUS_ID, NUM_OFFSET, NUM_OFFSET + 1, PAD_ID]):
        try:
            decode_generated(bad)
        except ValueError:
            pass
        else:
            raise ValueError("Invalid generated syntax was accepted")
    for c in (-10000, 10000):
        try:
            encode_coefficient(c)
        except ValueError:
            pass
        else:
            raise ValueError("Out-of-range fixed2 coefficient was accepted")

    loader = HexagonDataLoader(train, 8, device=device)
    first = next(loader)
    state = loader.state_dict()
    expected_next = next(loader)
    restored = HexagonDataLoader(train, 8, device=device)
    restored.set_state(state)
    actual_next = next(restored)
    for key in ("idx", "targets", "token_types", "target_types", "loss_weights"):
        ensure(torch.equal(expected_next[key], actual_next[key]), f"Loader resume mismatch: {key}")
    system = build_gpu_system(config)
    validate_attention(system)
    ensure(all(p.device.type == "cuda" for p in system.parameters()), "CPU model parameters found")
    ensure(sum(p.numel() for p in system.parameters()) == EXPECTED_PARAMETERS, "Parameter count mismatch")
    # Exercise attention at nonzero branch weights: zero-initialized residual projections
    # otherwise hide incorrect masks. This diagnostic fixture is never used for training.
    with torch.no_grad():
        for block in system.trunk.blocks:
            block.attn.c_proj.weight.normal_(std=0.02)
            block.mlp.c_proj.weight.normal_(std=0.02)
        system.head.lm_head.weight.normal_(std=0.02)
    system.eval()
    x, types = first["idx"], first["token_types"]
    changed = x.clone()
    changed[:, 10] = torch.where(changed[:, 10] == PLUS_ID, MINUS_ID, PLUS_ID)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        h1 = system.trunk(x, token_types=types)
        h2 = system.trunk(changed, token_types=types)
        torch.testing.assert_close(h1[:, :10], h2[:, :10], rtol=0, atol=0)
        from projects.amplitude_symbol.blocks.attention import attach_word_bidirectional_attention
        attach_word_bidirectional_attention(system, 16, word_start=1, word_end=11)
        bad1 = system.trunk(x, token_types=types)
        bad2 = system.trunk(changed, token_types=types)
        ensure(not torch.equal(bad1[:, :10], bad2[:, :10]), "Leakage negative control did not exercise attention")
        attach_attention(system)
        validate_attention(system)
        logits = system.head(system.trunk(x, token_types=types))
        ensure(logits.shape == (8, 15, 115) and torch.isfinite(logits).all(), "Invalid logits")
        prompt = x[:4, :10]
        cached = autoregressive_generate(system, prompt, types[:4, :10], max_new_tokens=4,
                                         gen_token_type=2, stop_token=EOS_ID, temperature=0.0,
                                         early_stop=False)
        reference = full_forward_generation(system, prompt)
        ensure(torch.equal(cached, reference), "Cached and full-forward greedy output differ")
    system.train()
    system.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = system.loss(first)
    ensure(torch.isfinite(loss), "Nonfinite diagnostic loss")
    loss.backward()
    gradients = [p.grad for p in system.parameters() if p.grad is not None]
    ensure(gradients and all(torch.isfinite(g).all() for g in gradients), "Missing/nonfinite CUDA gradients")
    result = {"status": "passed", "device": str(device), "rows_checked_on_cuda": 11208,
              "parameters": EXPECTED_PARAMETERS, "encoding": encoding_metadata(),
              "loss_shift": "targets[:,9:13]", "mask": "[1,9)",
              "no_label_leakage": True, "leakage_negative_control": True,
              "cache_full_generation_equal": True, "loader_resume_equal": True,
              "finite_forward_backward": True, "diagnostic_loss": float(loss.detach())}
    del system, batch, assembled, loader, restored
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run(args):
    device = require_cuda_server()
    output = outside_repository(args.output)
    data_dir = outside_repository(args.data_dir)
    if output.exists() or output == data_dir or data_dir in output.parents or output in data_dir.parents:
        raise ValueError("Use separate new run/data directories outside the repository")
    if args.source and (args.source.expanduser().resolve().parent == output
                        or args.source.expanduser().resolve().parent in output.parents):
        raise ValueError("Run output must be outside the immutable source directory")
    if data_dir.exists():
        if args.source or not args.metadata_sha256:
            raise ValueError("Existing data: pass --metadata-sha256, omit --source")
        digest = args.metadata_sha256
    else:
        if not args.source or args.metadata_sha256:
            raise ValueError("New data: pass --source, omit --metadata-sha256")
        digest = prepare_data(args.source, data_dir)["metadata_sha256"]
    train = HexagonDataset(data_dir, "train", expected_metadata_sha256=digest)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write_json(output / "environment.json", {
        "python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device), "device": str(device),
        "metadata_sha256": digest, "code_sha256": code_hashes(), "standalone_cpu_tests": False})
    config = recipe("smoke", data_dir, digest, output / "continuous")
    write_json(output / "cuda_contract_checks.json", cuda_contract_checks(config, train))
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
    left, left_meta = load_for_evaluation(continuous["last_checkpoint"], train)
    right, right_meta = load_for_evaluation(resumed["last_checkpoint"], train)
    a, b = left.state_dict(), right.state_dict()
    ensure(set(a) == set(b), "Checkpoint tensor keys differ")
    max_delta = 0.0
    for key in a:
        ensure(a[key].device.type == "cuda" and b[key].device.type == "cuda", "CPU comparison forbidden")
        torch.testing.assert_close(a[key], b[key], rtol=1e-5, atol=1e-6)
        max_delta = max(max_delta, float((a[key].float() - b[key].float()).abs().max()))
    ensure(left_meta["dataloader_state"] == right_meta["dataloader_state"], "Final loader state differs")
    ensure(left_meta["rng_state"] == right_meta["rng_state"], "Final RNG states differ")
    left_val = json.loads((output / "continuous" / "val_subset_000020.json").read_text())
    right_val = json.loads((output / "resumed" / "val_subset_000020.json").read_text())
    ensure(left_val["sample_metrics"] == right_val["sample_metrics"], "Resumed generation metrics differ")
    report = {"status": "passed_gpu_smoke", "elapsed_seconds": time.monotonic() - started,
              "parameters": EXPECTED_PARAMETERS, "split_counts": {"train": 8966, "val": 1120, "test": 1122},
              "metadata_sha256": digest, "model_updates": 40,
              "continuous_steps": continuous["completed_steps"], "resumed_steps": resumed["completed_steps"],
              "resume_parameters_max_abs_difference": max_delta, "resume_rtol": 1e-5, "resume_atol": 1e-6,
              "test_evaluated": False, "formal_training_completed": False,
              "standalone_cpu_tests": False, "host_data_io_checks": True}
    write_json(output / "gpu_smoke_report.json", report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Published x32 gzip; only when creating NEW training data")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--metadata-sha256", help="Required for an existing immutable training dataset")
    parser.add_argument("--output", type=Path, required=True, help="NEW GPU smoke run directory")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        print(json.dumps({"status": "failed_gpu_smoke", "error": str(exc)}, ensure_ascii=False))
        raise


if __name__ == "__main__":
    main()
