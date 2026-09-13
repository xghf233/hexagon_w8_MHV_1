"""Explicit immutable-checkpoint evaluation. Final test is a separate opt-in."""

import argparse
from pathlib import Path
import platform
import numpy as np
import torch

from .baselines import TrainBaselines
from .checkpoint import code_hashes, read_metadata
from .data_contract import outside_repository, separate_path, require, write_json
from .dataset import HexagonDataset
from .evaluator import evaluate
from .model import ATTENTION_CONFIG, build_gpu_system, validate_attention
from .runtime import require_cuda_server
from .tokenizer import encoding_metadata


def load_for_evaluation(checkpoint, dataset):
    require_cuda_server()
    meta = read_metadata(checkpoint)
    contract = meta["m2_contract"]
    require(contract["dataset"]["metadata_sha256"] == dataset.metadata_sha256, "Checkpoint/data mismatch")
    require(contract["code_sha256"] == code_hashes(), "Checkpoint dependency code changed")
    require(contract["torch_version"] == str(torch.__version__) and contract["numpy_version"] == np.__version__
            and contract["python_version"] == platform.python_version()
            and contract["cuda_version"] == torch.version.cuda, "Checkpoint environment mismatch")
    config = {**contract["recipe"], "sequence_len": contract["encoding"]["sequence_len"],
              "model": {key: contract["model_config"][key]
                        for key in ("n_layer", "n_embd", "n_head", "n_kv_head")}}
    require(contract["encoding"] == encoding_metadata(config["sequence_len"])
            and contract["attention"] == ATTENTION_CONFIG, "Tokenizer/attention mismatch")
    system = build_gpu_system(config)
    require(type(system.head).__name__ == contract["head_type"], "Loss-head implementation changed")
    require(contract["matmul_precision"] == torch.get_float32_matmul_precision(), "Matmul policy changed")
    from core.model.checkpoint_manager import load_model_only
    load_model_only(str(checkpoint), system)
    validate_attention(system)
    return system, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--metadata-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--gen-batch-size", type=int, default=64)
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        parser.error("Final test requires --allow-test after freezing model and protocol")
    if args.gen_batch_size <= 0:
        parser.error("--gen-batch-size must be positive")
    require_cuda_server()
    dataset = HexagonDataset(args.data_dir, args.split, expected_metadata_sha256=args.metadata_sha256)
    output = outside_repository(args.output)
    separate_path(output, dataset.root)
    separate_path(output, Path(dataset.metadata["source"]["path"]).parent)
    separate_path(output, args.checkpoint)
    require(not output.exists(), "Choose a NEW report path")
    system, meta = load_for_evaluation(args.checkpoint, dataset)
    result = evaluate(system, dataset, dataset.indices, batch_size=args.gen_batch_size,
                      save_predictions=args.save_predictions)
    train = HexagonDataset(args.data_dir, "train", expected_metadata_sha256=args.metadata_sha256)
    result["baselines"] = TrainBaselines(train).evaluate(dataset, dataset.indices)
    result["checkpoint"] = str(args.checkpoint.resolve())
    result["completed_steps"] = meta["completed_steps"]
    result["test_opt_in"] = args.allow_test
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(result["sample_metrics"])


if __name__ == "__main__":
    main()
