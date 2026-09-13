"""Single-GPU heptagon runner; no local execution is implied by this module.

Reuse core's GPT assembly, AdamW groups, schedule and DCP implementation. This
project loop owns sample accounting, full-vs-subset evaluation, best selection
and the complete checkpoint envelope; it does not modify the shared Trainer.
"""

import json
import math
import os
from pathlib import Path
import random
import time

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

from core.training.lr_schedulers import get_lr_multiplier
from .checkpoint import capture_rng, restore_rng, resume, run_contract, save, write_json
from .dataset import HeptagonDataset, HeptagonDataLoader
from .evaluator import evaluate, metrics, select_rows
from .model import build_gpu_system, model_config


def validate_config(config: dict) -> None:
    allowed = {"data_dir", "output_dir", "resume_from", "stop_after_steps", "sequence_len",
               "model", "seed", "compile", "head_softcap", "device_batch_size", "total_batch_size",
               "max_steps", "optimizer", "evaluation", "checkpoint", "logging",
               "overfit_n_samples", "require_overfit_exact"}
    # Hydra injects a top-level `hydra` node into every composed config; it is
    # framework bookkeeping, not recipe, and must not fail the strict check.
    if (set(config) ^ allowed) - {"hydra"}:
        raise ValueError(f"Missing/unsupported configuration keys: {(set(config) ^ allowed) - {'hydra'}}")
    model_config(config)
    for key in ("device_batch_size", "max_steps"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be positive; automatic step calculation is disabled")
    if config["sequence_len"] != 16:
        raise ValueError("The first w6 recipe uses sequence_len=16")
    if config["total_batch_size"] != config["device_batch_size"] * config["sequence_len"]:
        raise ValueError("Only one GPU and gradient accumulation=1 are supported")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be a uint32 integer")
    for key in ("compile", "require_overfit_exact"):
        if type(config[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    for key in ("overfit_n_samples", "stop_after_steps"):
        if config[key] is not None and (type(config[key]) is not int or config[key] <= 0):
            raise ValueError(f"{key} must be null or a positive integer")
    if config["require_overfit_exact"] and config["overfit_n_samples"] is None:
        raise ValueError("Overfit acceptance requires an explicit small training subset")
    optimizer = config["optimizer"]
    if set(optimizer) != {"type", "lr_max", "embedding_lr", "unembedding_lr", "weight_decay",
                          "betas", "fused", "max_grad_norm", "scheduler"}:
        raise ValueError("Unexpected optimizer configuration")
    if optimizer["type"] != "adamw" or optimizer["fused"] is not True:
        raise ValueError("Core optimizer recipe is fused AdamW")
    for key in ("lr_max", "embedding_lr", "unembedding_lr", "max_grad_norm"):
        if not math.isfinite(optimizer[key]) or optimizer[key] <= 0:
            raise ValueError(f"Invalid optimizer {key}")
    if not math.isfinite(config["head_softcap"]) or config["head_softcap"] <= 0:
        raise ValueError("Invalid head softcap")
    if not math.isfinite(optimizer["weight_decay"]) or optimizer["weight_decay"] < 0:
        raise ValueError("Invalid weight decay")
    if len(optimizer["betas"]) != 2 or any(not 0 <= beta < 1 for beta in optimizer["betas"]):
        raise ValueError("Invalid AdamW betas")
    schedule = optimizer["scheduler"]
    if set(schedule) != {"type", "warmup_steps", "warmdown_ratio", "final_lr_frac"}:
        raise ValueError("Unexpected scheduler configuration")
    if schedule["type"] != "linear" or not 0 <= schedule["warmdown_ratio"] < 1:
        raise ValueError("Expected linear warmup/constant/warmdown recipe")
    warmup = schedule["warmup_steps"]
    if type(warmup) is not int or not 0 <= warmup <= config["max_steps"] - round(
            schedule["warmdown_ratio"] * config["max_steps"]):
        raise ValueError("Warmup and warmdown exceed the training budget")
    if not 0 <= schedule["final_lr_frac"] <= 1:
        raise ValueError("Invalid final LR fraction")
    ev = config["evaluation"]
    if set(ev) != {"subset_size", "subset_seed", "interval_steps", "full_interval_steps",
                   "full_validation", "gen_batch_size", "train_diagnostic_size"}:
        raise ValueError("Unexpected evaluation protocol")
    for key in ("subset_size", "interval_steps", "full_interval_steps", "gen_batch_size", "train_diagnostic_size"):
        if type(ev[key]) is not int or ev[key] <= 0:
            raise ValueError(f"Invalid evaluation {key}")
    if type(ev["subset_seed"]) is not int or ev["subset_seed"] < 0 or type(ev["full_validation"]) is not bool:
        raise ValueError("Invalid evaluation seed/full_validation")
    if set(config["checkpoint"]) != {"save_every"} or type(config["checkpoint"]["save_every"]) is not int or config["checkpoint"]["save_every"] <= 0:
        raise ValueError("Invalid checkpoint interval")
    if set(config["logging"]) != {"log_every"} or type(config["logging"]["log_every"]) is not int or config["logging"]["log_every"] <= 0:
        raise ValueError("Invalid logging interval")


def fresh_output(path: str, data_dir: Path) -> Path:
    """Never place run artifacts in either edition or inside the immutable dataset."""
    output = Path(path).expanduser().resolve()
    repo = Path(__file__).resolve().parents[3]
    if output == repo or repo in output.parents or output == data_dir or data_dir in output.parents:
        raise ValueError("Output must be outside the repository and the dataset directory")
    if output.exists():
        raise FileExistsError("Choose a new run directory, including for resumed runs")
    return output


def run_training(config: dict) -> dict:
    validate_config(config)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or "RANK" in os.environ:
        raise ValueError("Launch with plain python on one GPU")
    train = HeptagonDataset(config["data_dir"], "train")
    if config["overfit_n_samples"] is not None:
        train = train.training_subset(config["overfit_n_samples"], config["seed"])
    val = HeptagonDataset(config["data_dir"], "val")
    ev = config["evaluation"]
    val_subset = select_rows(val, ev["subset_size"], ev["subset_seed"])
    train_rows = select_rows(train, min(ev["train_diagnostic_size"], len(train)), ev["subset_seed"])
    output = fresh_output(config["output_dir"], train.root)
    system = build_gpu_system(config)
    from core.training.optim import build_optimizers
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    loader = HeptagonDataLoader(train, config["device_batch_size"], seed=config["seed"], device="cuda")
    optimizers = build_optimizers(system, config["optimizer"])
    contract = run_contract(config, train, head_type=type(system.head).__name__)
    completed, best = 0, None
    if config["resume_from"]:
        meta = resume(Path(config["resume_from"]).expanduser().resolve(strict=True),
                      system, optimizers, loader, contract)
        completed, best = meta["completed_steps"], meta["best"]
    initial_steps = completed
    end_step = config["max_steps"]
    if config["stop_after_steps"] is not None:
        end_step = min(end_step, completed + config["stop_after_steps"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "checkpoints").mkdir()
    n_params = sum(parameter.numel() for parameter in system.parameters())
    write_json(output / "run.json", {
        "config": config, "contract": contract, "n_parameters": n_params,
        "initial_completed_steps": initial_steps, "planned_end_step": end_step,
        "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
        "effective_optimizer_groups": [
            {"lr": group["initial_lr"], "eps": group["eps"], "weight_decay": group["weight_decay"]}
            for opt in optimizers for group in opt.param_groups],
        "samples_per_step": loader.batch_size, "actual_input_tokens_per_step": loader.batch_size * 15,
        "supervised_tokens_per_step": loader.batch_size * 3,
    })
    np.savez(output / "evaluation_rows.npz", val_subset=val_subset, train_diagnostic=train_rows)
    # Choose the constant predictor using train only; never evaluate on test here.
    values, frequencies = np.unique(train.coefficients[train.indices], return_counts=True)
    majority = int(values[np.argmax(frequencies)])
    write_json(output / "baseline.json", {"predictor": majority, "selected_from": "train",
        "val_full": metrics([majority] * len(val), val.coefficients[val.indices].tolist())})
    schedule = {**config["optimizer"]["scheduler"], "max_steps": config["max_steps"]}
    system.train()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    train_exact = None
    history_path = output / "history.jsonl"

    def log(record):
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)

    while completed < end_step:
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * get_lr_multiplier(completed, schedule)
        batch = next(loader)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = system.loss(batch)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Nonfinite training loss before update {completed + 1}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(system.parameters(), config["optimizer"]["max_grad_norm"],
                                                   error_if_nonfinite=True)
        for optimizer in optimizers:
            optimizer.step()
        completed += 1
        last = completed == end_step
        if completed % config["logging"]["log_every"] == 0 or last or completed == initial_steps + 1:
            log({"event": "train", "completed_steps": completed, "loss": float(loss.detach()),
                 "grad_norm": float(grad_norm), "rows_consumed": completed * loader.batch_size,
                 "train_passes": completed * loader.batch_size / len(train),
                 "input_tokens": completed * loader.batch_size * 15,
                 "supervised_tokens": completed * loader.batch_size * 3,
                 "lr_groups": [group["lr"] for opt in optimizers for group in opt.param_groups],
                 "session_elapsed_seconds": time.perf_counter() - started,
                 "peak_cuda_bytes": torch.cuda.max_memory_allocated()})
        full = ev["full_validation"] and (last or completed % ev["full_interval_steps"] == 0)
        do_eval = full or last or completed % ev["interval_steps"] == 0
        improved = False
        checkpoint_path = output / "checkpoints" / f"step_{completed:06d}"
        if do_eval:
            rng = capture_rng()
            try:
                rows = val.indices if full else val_subset
                result = evaluate(system, val, rows, batch_size=ev["gen_batch_size"])
                tag = "val_full" if full else "val_subset"
                write_json(output / f"{tag}_{completed:06d}.json", result)
                log({"event": tag, "completed_steps": completed, **result["sample_metrics"]})
                if full and (best is None or result["sample_metrics"]["exact_accuracy"] > best["exact_accuracy"]):
                    improved = True
                    best = {"exact_accuracy": result["sample_metrics"]["exact_accuracy"],
                            "completed_steps": completed, "checkpoint": str(checkpoint_path),
                            "selection": "val_full"}
                diagnostic = evaluate(system, train, train_rows, batch_size=ev["gen_batch_size"])
                train_exact = diagnostic["sample_metrics"]["exact_accuracy"]
                write_json(output / f"train_diagnostic_{completed:06d}.json", diagnostic)
                log({"event": "train_diagnostic", "completed_steps": completed, **diagnostic["sample_metrics"]})
            finally:
                restore_rng(rng)
        if improved or last or completed % config["checkpoint"]["save_every"] == 0:
            save(checkpoint_path, system, optimizers, loader, contract, completed, best)
            log({"event": "checkpoint", "completed_steps": completed, "path": str(checkpoint_path),
                 "best": best})
    summary = {"completed_steps": completed, "rows_consumed": loader.state_dict()["rows_consumed"],
               "train_passes": loader.state_dict()["rows_consumed"] / len(train),
               "n_parameters": n_params, "best": best, "last_checkpoint": str(checkpoint_path),
               "train_diagnostic_exact": train_exact, "elapsed_seconds": time.perf_counter() - started,
               "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "test_evaluated": False}
    if config["require_overfit_exact"] and (len(train_rows) != len(train) or train_exact != 1.0):
        summary["acceptance"] = "failed_tiny_overfit"
        write_json(output / "summary.json", summary)
        raise RuntimeError("Tiny overfit did not reach exact=1 on the entire training subset")
    summary["acceptance"] = "passed_tiny_overfit" if config["require_overfit_exact"] else "not_requested"
    write_json(output / "summary.json", summary)
    return summary


@hydra.main(version_base=None, config_path="configs", config_name="w6_random")
def main(cfg: DictConfig) -> None:
    run_training(OmegaConf.to_container(cfg, resolve=True))


if __name__ == "__main__":
    main()
