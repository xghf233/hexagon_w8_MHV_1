"""Explicit checkpoint evaluation; test split requires a separate opt-in flag."""

import argparse
from pathlib import Path

from .checkpoint import code_hashes, read_metadata, write_json
from .dataset import HeptagonDataset
from .evaluator import evaluate, select_rows
from .model import ATTENTION_CONFIG, build_gpu_system, validate_attention
from .tokenizer import encoding_metadata


def load_for_evaluation(checkpoint, dataset):
    # Inspect all project metadata before loading tensor/optimizer state.
    meta = read_metadata(checkpoint)
    contract = meta["heptagon_contract"]
    if contract["dataset"]["metadata_sha256"] != dataset.metadata_sha256:
        raise ValueError("Checkpoint and evaluation dataset differ")
    if contract["code_sha256"] != code_hashes():
        raise ValueError("Checkpoint code version differs; reconcile code before evaluating")
    config = {**contract["recipe"], "sequence_len": contract["encoding"]["sequence_len"],
              "model": {key: contract["model_config"][key]
                        for key in ("n_layer", "n_embd", "n_head", "n_kv_head")}}
    if contract["encoding"] != encoding_metadata(config["sequence_len"]) or contract["attention"] != ATTENTION_CONFIG:
        raise ValueError("Tokenizer/attention contract mismatch")
    config["compile"] = False  # inference uses the raw trunk's cached decode path
    system = build_gpu_system(config)
    from core.model.checkpoint_manager import load_model_only
    load_model_only(str(checkpoint), system)
    validate_attention(system)
    return system, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--gen-batch-size", type=int, default=64)
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        parser.error("Test evaluation requires --allow-test after locking the protocol")
    if args.gen_batch_size <= 0:
        parser.error("--gen-batch-size must be positive")
    dataset = HeptagonDataset(args.data_dir, args.split)
    output = args.output.expanduser().resolve()
    repo = Path(__file__).resolve().parents[3]
    if output.exists() or repo == output or repo in output.parents or dataset.root in output.parents:
        raise ValueError("Output must be a new report outside Git and the dataset directory")
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    if checkpoint == output or checkpoint in output.parents:
        raise ValueError("Do not write reports inside an immutable checkpoint")
    system, meta = load_for_evaluation(checkpoint, dataset)
    result = evaluate(system, dataset, select_rows(dataset, None, 0),
                      batch_size=args.gen_batch_size, save_predictions=args.save_predictions)
    result["checkpoint"] = str(checkpoint)
    result["completed_steps"] = meta["completed_steps"]
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(result["sample_metrics"])


if __name__ == "__main__":
    main()
