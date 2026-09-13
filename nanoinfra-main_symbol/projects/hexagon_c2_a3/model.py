"""Task-only GPT assembly; complete prefix-LM attention is reinstalled on every load."""

import torch
from core.model.gpt import GPT, GPTConfig
from projects.amplitude_symbol.blocks.attention import word_bidirectional_mask
from .runtime import require_cuda_server
from .tokenizer import N_TOKEN_TYPES, VOCAB_SIZE, PROMPT_LENGTH, SEQUENCE_LENGTH, encoding_metadata

EXPECTED_PARAMETERS = 3_206_400
MODEL_GEOMETRY = {"n_layer": 4, "n_embd": 256, "n_head": 4, "n_kv_head": 4}
ATTENTION_CONFIG = {"mode": "prefix_lm", "prefix_length": PROMPT_LENGTH, "sequence_len": SEQUENCE_LENGTH}


def model_config(config):
    encoding_metadata(config["sequence_len"])
    if config["model"] != MODEL_GEOMETRY or any(type(v) is not int for v in config["model"].values()):
        raise ValueError("M2 geometry is locked to 4/256/4/4")
    return GPTConfig(sequence_len=SEQUENCE_LENGTH, vocab_size=VOCAB_SIZE,
                     n_token_types=N_TOKEN_TYPES, **MODEL_GEOMETRY)


def attach_attention(system):
    if system.config.sequence_len != SEQUENCE_LENGTH:
        raise ValueError("Wrong model sequence length")
    # The shared factory's historical name does not dictate this task's semantics.
    mask = word_bidirectional_mask(SEQUENCE_LENGTH, 0, PROMPT_LENGTH)
    mask = mask.to(next(system.parameters()).device)
    for block in system.trunk.blocks:
        block.attn._word_bidi_mask = mask
    system.attention_config = dict(ATTENTION_CONFIG)


def validate_attention(system):
    device = next(system.parameters()).device
    if device.type != "cuda":
        raise RuntimeError("Model checks require CUDA")
    if system.config.sequence_len != SEQUENCE_LENGTH or getattr(system, "attention_config", None) != ATTENTION_CONFIG:
        raise ValueError("Expected M2 prefix-LM [0,114)")
    # Independent formula checks the factory result, not just its metadata.
    q = torch.arange(SEQUENCE_LENGTH, device=device)[:, None]
    k = torch.arange(SEQUENCE_LENGTH, device=device)[None, :]
    expected = torch.where((q >= k) | ((q < PROMPT_LENGTH) & (k < PROMPT_LENGTH)), 0.0, -torch.inf)
    for block in system.trunk.blocks:
        mask = getattr(block.attn, "_word_bidi_mask", None)
        if mask is None or mask.device != device or not torch.equal(mask, expected):
            raise ValueError("Missing or incorrect runtime prefix mask")


def build_gpu_system(config):
    require_cuda_server()
    if config["compile"] is not False:
        raise ValueError("Compiled training is outside the first M2 experiment")
    from core.training.model_setup import build_system
    system = build_system(GPT, model_config(config), use_compile=False,
                          head_softcap=config["head_softcap"], seed=config["seed"])["system"]
    attach_attention(system)
    validate_attention(system)
    if sum(p.numel() for p in system.parameters()) != EXPECTED_PARAMETERS:
        raise ValueError("Unexpected parameter count; expected 3,206,400")
    return system
