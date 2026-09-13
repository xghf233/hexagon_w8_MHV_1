"""Project checkpoint envelope: exact recipe, sampler and RNG state, atomic publish."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from .model import ATTENTION_CONFIG, model_config, validate_attention
from .tokenizer import encoding_metadata

CHECKPOINT_VERSION = "heptagon-training-v1"


def write_json(path: Path, value: dict) -> None:
    """Write once; existing reports are never overwritten."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def code_hashes() -> dict:
    project = Path(__file__).resolve().parent
    edition = project.parents[1]
    paths = [project / name for name in (
        "alphabet.py", "tokenizer.py", "dataset.py", "model.py", "evaluator.py", "checkpoint.py",
        "train.py", "eval_checkpoint.py")]
    paths += [edition / "core" / name for name in (
        "model/gpt.py", "model/heads.py", "model/system.py", "model/kv_cache.py", "model/inference.py",
        "model/checkpoint_manager.py", "training/model_setup.py", "training/optim.py",
        "training/lr_schedulers.py", "data/supervision.py", "tokenization/vocab_layout.py")]
    paths.append(project.parent / "amplitude_symbol/blocks/attention.py")
    return {str(path.relative_to(edition)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def run_contract(config: dict, dataset, *, head_type: str) -> dict:
    return {"version": CHECKPOINT_VERSION, "dataset": dataset.identity(),
            "encoding": encoding_metadata(config["sequence_len"]),
            "attention": dict(ATTENTION_CONFIG), "model_config": asdict(model_config(config)),
            "recipe": {key: config[key] for key in (
                "seed", "compile", "head_softcap", "device_batch_size", "max_steps",
                "optimizer", "evaluation", "overfit_n_samples", "require_overfit_exact")},
            "head_type": head_type, "precision": "bf16", "code_sha256": code_hashes(),
            "torch_version": str(torch.__version__), "numpy_version": np.__version__}


def capture_rng() -> dict:
    legacy = np.random.get_state()
    return {"python": random.getstate(),
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
    if meta.get("heptagon_contract", {}).get("version") != CHECKPOINT_VERSION:
        raise ValueError("Not a supported heptagon checkpoint")
    if meta.get("attention_config") != ATTENTION_CONFIG:
        raise ValueError("Checkpoint attention configuration mismatch")
    completed = meta.get("completed_steps")
    if type(completed) is not int or completed <= 0 or meta.get("step") != completed - 1:
        raise ValueError("Invalid checkpoint step numbering")
    contract = meta["heptagon_contract"]
    if meta.get("model_config") != contract["model_config"]:
        raise ValueError("Checkpoint model blueprint disagrees with recipe")
    return meta


def save(directory: Path, system, optimizers, loader, contract: dict,
         completed_steps: int, best: dict | None) -> None:
    from core.model.checkpoint_manager import save_checkpoint_dcp
    validate_attention(system)
    if directory.exists():
        raise FileExistsError(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}-", dir=directory.parent))
    rng = capture_rng()
    try:
        save_checkpoint_dcp(str(staging), system, optimizers, {
            "step": completed_steps - 1, "completed_steps": completed_steps,
            "heptagon_contract": contract, "rng_state": rng, "best": best,
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
    meta = read_metadata(directory)
    if meta["heptagon_contract"] != contract:
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
