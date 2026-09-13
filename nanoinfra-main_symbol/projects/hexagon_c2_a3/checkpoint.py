"""Project checkpoint envelope: exact recipe, sampler and RNG state, atomic publish."""

from dataclasses import asdict
import json
import platform
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from .model import ATTENTION_CONFIG, model_config, validate_attention
from .tokenizer import encoding_metadata
from .data_contract import dependency_hashes, write_json
from .runtime import require_cuda_server

CHECKPOINT_VERSION = "hexagon-m2-adjacent38-prefix128-training-v1"

code_hashes = dependency_hashes


def run_contract(config: dict, dataset, *, head_type: str) -> dict:
    return {"version": CHECKPOINT_VERSION, "dataset": dataset.identity(),
            "encoding": encoding_metadata(config["sequence_len"]),
            "attention": dict(ATTENTION_CONFIG), "model_config": asdict(model_config(config)),
            "recipe": {key: config[key] for key in (
                "seed", "compile", "head_softcap", "device_batch_size", "max_steps", "metadata_sha256",
                "optimizer", "evaluation", "overfit_n_samples", "require_overfit_exact",
                "total_batch_size", "checkpoint", "logging")},
            "head_type": head_type, "precision": "bf16", "code_sha256": code_hashes(),
            "task": "hexagon_w8_adjacent_2y_nonzero_m2_38", "coefficient_scale": 32, "split_type": "random_row",
            "torch_version": str(torch.__version__), "numpy_version": np.__version__,
            "python_version": platform.python_version(), "cuda_version": torch.version.cuda,
            "matmul_precision": torch.get_float32_matmul_precision()}


def capture_rng() -> dict:
    legacy = np.random.get_state()
    python_state = random.getstate()
    # JSON-canonical containers make live-versus-reloaded equality meaningful.
    return {"python": [python_state[0], list(python_state[1]), python_state[2]],
            "numpy": [legacy[0], legacy[1].tolist(), legacy[2], legacy[3], legacy[4]],
            "torch_cpu": torch.get_rng_state().tolist(),
            "torch_cuda": [torch.cuda.get_rng_state(0).tolist()] if torch.cuda.is_initialized() else []}


def restore_rng(state: dict) -> None:
    def tuples(value):
        return tuple(tuples(item) for item in value) if isinstance(value, (list, tuple)) else value
    random.setstate(tuples(state["python"]))
    legacy = state["numpy"]
    np.random.set_state((legacy[0], np.array(legacy[1], dtype=np.uint32), *legacy[2:]))
    torch.set_rng_state(torch.tensor(state["torch_cpu"], dtype=torch.uint8, device="cpu"))
    if state["torch_cuda"]:
        if not torch.cuda.is_available() or len(state["torch_cuda"]) != 1:
            raise ValueError("Expected RNG state for the single training GPU")
        torch.cuda.set_rng_state(torch.tensor(state["torch_cuda"][0], dtype=torch.uint8, device="cpu"), 0)


def read_metadata(directory: str | Path) -> dict:
    directory = Path(directory).expanduser().resolve(strict=True)
    with (directory / "meta.json").open(encoding="utf-8") as stream:
        meta = json.load(stream)
    if meta.get("m2_contract", {}).get("version") != CHECKPOINT_VERSION:
        raise ValueError("Not a supported M2/38 checkpoint")
    if meta.get("attention_config") != ATTENTION_CONFIG:
        raise ValueError("Checkpoint attention configuration mismatch")
    completed = meta.get("completed_steps")
    if type(completed) is not int or completed <= 0 or meta.get("step") != completed - 1:
        raise ValueError("Invalid checkpoint step numbering")
    contract = meta["m2_contract"]
    if meta.get("model_config") != contract["model_config"]:
        raise ValueError("Checkpoint model blueprint disagrees with recipe")
    return meta


def save(directory: Path, system, optimizers, loader, contract: dict,
         completed_steps: int, best: dict | None) -> None:
    from core.model.checkpoint_manager import save_checkpoint_dcp
    require_cuda_server()
    validate_attention(system)
    if completed_steps != loader.state_dict()["rows_consumed"] // loader.batch_size:
        raise ValueError("Optimizer-step/sampler accounting differs at save")
    if not all(bool(torch.isfinite(p).all()) for p in system.parameters()):
        raise FloatingPointError("Refusing to save nonfinite model parameters")
    if directory.exists():
        raise FileExistsError(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}-", dir=directory.parent))
    rng = capture_rng()
    try:
        save_checkpoint_dcp(str(staging), system, optimizers, {
            "step": completed_steps - 1, "completed_steps": completed_steps,
            "m2_contract": contract, "rng_state": rng, "best": best,
        }, dataloader_state=loader.state_dict())
        read_metadata(staging)
        if directory.exists():
            raise FileExistsError(directory)
        staging.rename(directory)
    finally:
        # Saving itself must not alter the subsequent training RNG stream.
        restore_rng(rng)


def resume(directory, system, optimizers, loader, contract: dict) -> dict:
    from core.model.checkpoint_manager import load_checkpoint_dcp
    require_cuda_server()
    meta = read_metadata(directory)
    if meta["m2_contract"] != contract:
        raise ValueError("Resume contract mismatch: data/code/model/recipe/environment changed")
    if meta["completed_steps"] >= contract["recipe"]["max_steps"]:
        raise ValueError("Checkpoint has already exhausted the configured training budget")
    if meta["dataloader_state"]["rows_consumed"] != meta["completed_steps"] * loader.batch_size:
        raise ValueError("Checkpoint optimizer steps and consumed rows disagree")
    # Validate sampler configuration before mutating model/optimizer parameters.
    loader.set_state(meta["dataloader_state"])
    load_checkpoint_dcp(str(directory), system, load_optimizer=True, optimizers=optimizers)
    validate_attention(system)
    restore_rng(meta["rng_state"])
    return meta
