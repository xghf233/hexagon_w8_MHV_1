"""Tests for the Symbol project's runtime attention configuration."""

from types import SimpleNamespace

import pytest
import torch

from projects.amplitude_symbol.blocks.attention import (
    attach_word_bidirectional_attention,
    word_bidirectional_mask,
)


def test_word_region_is_bidirectional_and_coefficient_region_is_causal():
    mask = word_bidirectional_mask(sequence_len=16, word_start=1, word_end=11)

    assert mask[1, 10] == 0
    assert torch.isneginf(mask[1, 11])
    assert mask[12, 11] == 0
    assert torch.isneginf(mask[11, 12])


def test_attach_records_checkpoint_metadata():
    blocks = [SimpleNamespace(attn=SimpleNamespace()) for _ in range(2)]
    system = SimpleNamespace(trunk=SimpleNamespace(blocks=blocks))

    attach_word_bidirectional_attention(system, sequence_len=32)

    assert system.attention_config == {
        "mode": "word_bidirectional",
        "word_start": 1,
        "word_end": 11,
    }
    assert all(block.attn._word_bidi_mask.shape == (32, 32) for block in blocks)


def test_attach_rejects_invalid_word_interval():
    system = SimpleNamespace(trunk=SimpleNamespace(blocks=[]))
    with pytest.raises(ValueError, match="word_start"):
        attach_word_bidirectional_attention(
            system,
            sequence_len=8,
            word_start=1,
            word_end=11,
        )
