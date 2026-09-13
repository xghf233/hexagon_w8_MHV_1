"""Attention mask factories for amplitude symbol experiments.

Each function takes a sequence length and returns a ``[T, T]`` float mask
(0.0 = attend, -inf = mask) suitable for ``F.scaled_dot_product_attention``.
"""

from __future__ import annotations

import torch


WORD_START = 1
WORD_END = 11


def causal_mask(sequence_len: int) -> torch.Tensor:
    """Standard causal (lower-triangular) attention mask."""
    q_idx = torch.arange(sequence_len)[:, None]
    k_idx = torch.arange(sequence_len)[None, :]
    mask = torch.where(q_idx >= k_idx, 0.0, float("-inf"))
    return mask


def word_bidirectional_mask(
    sequence_len: int = 32,
    word_start: int = WORD_START,
    word_end: int = WORD_END,
) -> torch.Tensor:
    """Attention mask where word-region positions see each other bidirectionally.

    Positions in ``[word_start, word_end)`` can attend to each other fully,
    while all other positions use standard causal attention (q >= k).

    For the amplitude symbol task:
      - Position  0: BOS          (causal)
      - Positions 1–10: word letters  (bidirectional within the block)
      - Position 11:   COEFF separator (causal, sees all word positions)
      - Positions 12+:  coefficient + EOS + PAD  (standard causal)
    """
    mask = torch.full((sequence_len, sequence_len), float("-inf"))

    q_idx = torch.arange(sequence_len)[:, None]   # [T, 1]
    k_idx = torch.arange(sequence_len)[None, :]   # [1, T]

    # Standard causal: q >= k
    mask[q_idx >= k_idx] = 0.0

    # Word region: full bidirectional
    q_in_word = (q_idx >= word_start) & (q_idx < word_end)
    k_in_word = (k_idx >= word_start) & (k_idx < word_end)
    mask[q_in_word & k_in_word] = 0.0

    return mask


def attach_word_bidirectional_attention(
    system,
    sequence_len: int,
    *,
    word_start: int = WORD_START,
    word_end: int = WORD_END,
) -> None:
    """Attach the runtime mask and its checkpoint metadata to ``system``."""
    if not 0 <= word_start < word_end <= sequence_len:
        raise ValueError(
            "Expected 0 <= word_start < word_end <= sequence_len, got "
            f"{word_start}, {word_end}, {sequence_len}"
        )
    mask = word_bidirectional_mask(
        sequence_len=sequence_len,
        word_start=word_start,
        word_end=word_end,
    )
    for block in system.trunk.blocks:
        block.attn._word_bidi_mask = mask
    system.attention_config = {
        "mode": "word_bidirectional",
        "word_start": word_start,
        "word_end": word_end,
    }
