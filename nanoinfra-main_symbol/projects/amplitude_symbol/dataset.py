"""DataSource and DataLoader for amplitude symbol coefficient prediction.

Follows the nano_motion/train_t2m.py pattern:
    DataSource (yields individual padded samples)
        → AmplitudeDataLoader (batches, applies NextTokenPrediction shift,
          masks unsupervised positions)
        → Trainer-compatible {idx, targets, token_types} batches
"""

from __future__ import annotations

import random
from typing import Any, Iterator

import torch

from .tokenizer import (
    IGNORE_INDEX,
    assemble_sequence,
)


class AmplitudeDataSource:
    """Infinite iterator yielding individual symbol samples.

    Each yielded sample is an AssembledSequence with .tokens, .token_types,
    and .loss_weights, all padded to sequence_len.

    The source is shuffled once per epoch. Yields indefinitely.

    IMPORTANT: Tensors are placed on `device` by the source — NanoInfra's
    Trainer expects the dataloader to handle GPU placement.
    """

    def __init__(
        self,
        samples: list[tuple[str, int]],
        sequence_len: int = 32,
        seed: int = 42,
        shuffle: bool = True,
        device: str | torch.device = "cpu",
    ):
        """
        Args:
            samples: List of (word, coefficient) pairs.
            sequence_len: Fixed sequence length for all samples.
            seed: Random seed for shuffling.
            shuffle: Whether to shuffle samples each epoch.
            device: Device to place tensors on ("cuda" or "cpu").
        """
        self.samples = samples
        self.sequence_len = sequence_len
        self.rng = random.Random(seed)
        self.shuffle = shuffle
        self.device = torch.device(device) if isinstance(device, str) else device
        self._order = list(range(len(samples)))
        self._pos = 0
        if shuffle:
            self.rng.shuffle(self._order)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        return self._generate()

    def _generate(self) -> Iterator[dict[str, torch.Tensor]]:
        while True:
            for idx in self._order:
                word, coeff = self.samples[idx]
                seq = assemble_sequence(word, coeff, self.sequence_len)
                yield {
                    "tokens": seq.tokens.to(self.device),
                    "token_types": seq.token_types.to(self.device),
                    "loss_weights": seq.loss_weights.to(self.device),
                }
                self._pos += 1
            # Start new epoch — reshuffle
            if self.shuffle:
                self.rng.shuffle(self._order)
            self._pos = 0

    def get_state(self) -> dict[str, Any] | None:
        """Checkpointable state: row position and order."""
        return {
            "pos": self._pos,
            "order": list(self._order),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore from checkpoint."""
        self._pos = state["pos"]
        self._order = state["order"]


class AmplitudeDataLoader:
    """Batches DataSource samples and produces Trainer-compatible dicts.

    Applies the NextTokenPrediction convention:
      - idx:         tokens[:, :-1]  — input positions
      - targets:     tokens[:, 1:]   — shifted targets, with unsupervised
                      positions set to IGNORE_INDEX
      - token_types: types[:, :-1]   — input-aligned token types

    This is the single entry point the Trainer sees — it receives {idx,
    targets, token_types} and nothing else.
    """

    def __init__(
        self,
        source: AmplitudeDataSource,
        batch_size: int,
    ):
        self.source = source
        self.batch_size = batch_size
        self._it = iter(source)
        self._n = 0  # cumulative rows served (for checkpointing)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        rows = [next(self._it) for _ in range(self.batch_size)]
        self._n += len(rows)

        toks = torch.stack([r["tokens"] for r in rows])           # [B, L]
        lw = torch.stack([r["loss_weights"] for r in rows])       # [B, L]
        types = torch.stack([r["token_types"] for r in rows])     # [B, L]

        # Shift-by-1: input=positions 0..L-2, target=positions 1..L-1
        # Unsupervised target positions → IGNORE_INDEX
        targets = torch.where(
            lw[:, 1:] > 0,
            toks[:, 1:],
            torch.full_like(toks[:, 1:], IGNORE_INDEX),
        )

        return {
            "idx": toks[:, :-1],
            "targets": targets,
            "token_types": types[:, :-1],
        }

    def state_dict(self) -> dict[str, Any]:
        return {"rows": self._n}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._it = iter(self.source)
        # Fast-forward
        to_skip = state.get("rows", 0) // self.batch_size
        for _ in range(to_skip):
            next(self)
        self._n = state.get("rows", 0)
