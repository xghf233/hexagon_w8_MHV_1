"""Heptagon assembly and checked runtime attention (not stored in tensor state)."""

import os

import torch

from core.model.gpt import GPT, GPTConfig
from projects.amplitude_symbol.blocks.attention import attach_word_bidirectional_attention, word_bidirectional_mask
from .tokenizer import N_TOKEN_TYPES, VOCAB_SIZE, WORD_START, WORD_END, encoding_metadata

ATTENTION_CONFIG = {"mode": "word_bidirectional", "word_start": WORD_START, "word_end": WORD_END}


def model_config(config: dict) -> GPTConfig:
    encoding_metadata(config["sequence_len"])
    geometry = config["model"]
    if set(geometry) != {"n_layer", "n_embd", "n_head", "n_kv_head"}:
        raise ValueError("Model geometry must not override the tokenizer's vocabulary")
    for value in geometry.values():
        if type(value) is not int or value <= 0:
            raise ValueError("Model dimensions must be positive integers")
    width, heads, kv = geometry["n_embd"], geometry["n_head"], geometry["n_kv_head"]
    if width % heads or heads % kv or (width // heads) % 2:
        raise ValueError("Invalid head geometry or odd rotary head dimension")
    return GPTConfig(sequence_len=config["sequence_len"], vocab_size=VOCAB_SIZE,
                     n_token_types=N_TOKEN_TYPES, **geometry)


def attach_attention(system) -> None:
    # Reuse the existing general mask factory, NEVER its ten-letter defaults.
    attach_word_bidirectional_attention(system, system.config.sequence_len,
                                        word_start=WORD_START, word_end=WORD_END)


def validate_attention(system) -> None:
    if getattr(system, "attention_config", None) != ATTENTION_CONFIG:
        raise ValueError("Expected heptagon word-bidirectional attention over [1,7)")
    expected = word_bidirectional_mask(system.config.sequence_len, WORD_START, WORD_END)
    for block in system.trunk.blocks:
        mask = getattr(block.attn, "_word_bidi_mask", None)
        if mask is None or mask.shape != (system.config.sequence_len,) * 2:
            raise ValueError("Runtime attention mask is missing or has wrong shape")
        if not torch.equal(mask.cpu(), expected.cpu()):
            raise ValueError("Runtime attention mask content differs from heptagon protocol")


def build_gpu_system(config: dict):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or "RANK" in os.environ:
        raise ValueError("Use plain python on one GPU, not a distributed launcher")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This training recipe requires a bf16-capable CUDA GPU")
    # Import only for GPU execution; config and mask unit tests can stay CPU-only.
    from core.training.model_setup import build_system
    torch.cuda.set_device(0)
    setup = build_system(GPT, model_config(config), use_compile=False,
                         head_softcap=config["head_softcap"], seed=config["seed"])
    system = setup["system"]
    attach_attention(system)
    validate_attention(system)
    # Compile AFTER installing the runtime behavior.
    if config["compile"]:
        system.set_compiled_trunk(torch.compile(system.trunk, dynamic=True))
    return system
