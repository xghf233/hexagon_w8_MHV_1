"""Experiment: word-bidirectional attention for D3 orbit coefficient prediction.

Positions 1–10 (word letters) can see each other fully — the model no longer
reads word letters in causal left-to-right order.  Coefficient positions
(11+) remain standard causal.

Thin glue: imports ``run_training`` from blocks and attaches the word-bidi
mask via a ``setup_model`` callback.  Everything else (data, split, eval)
is shared with the baseline.

Usage:
    python experiments/word_bidi/train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

_NANOINFRA_ROOT = Path(__file__).resolve().parents[4]
if str(_NANOINFRA_ROOT) not in sys.path:
    sys.path.insert(0, str(_NANOINFRA_ROOT))

from core.utils import print0
from projects.amplitude_symbol.blocks.attention import (
    WORD_END,
    WORD_START,
    attach_word_bidirectional_attention,
)
from projects.amplitude_symbol.blocks.training import run_training


def _attach_word_bidi_mask(system, config):
    """Attach word-bidirectional attention mask to every attention layer."""
    seq_len = config["sequence_len"]
    attach_word_bidirectional_attention(system, seq_len)
    print0(
        "  Word-bidirectional attention mask attached "
        f"(positions {WORD_START}-{WORD_END - 1})"
    )
    return system


@hydra.main(
    version_base=None,
    config_path=".",
    config_name="word_bidi",
)
def main(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)

    print0("=" * 72)
    print0("  WORD-BIDIRECTIONAL ATTENTION EXPERIMENT")
    print0("  word positions 1-10: full bidirectional")
    print0("  all other positions: standard causal")
    print0("=" * 72)

    run_training(config, setup_model=_attach_word_bidi_mask)


if __name__ == "__main__":
    main()
