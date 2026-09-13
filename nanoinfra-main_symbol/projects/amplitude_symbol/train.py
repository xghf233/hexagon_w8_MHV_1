"""Training entry point — baseline (causal attention, no D3 constraints).

Thin Hydra glue that delegates to ``blocks.training.run_training``.

Usage:
    python -m projects.amplitude_symbol.train --config-name=baseline_orbit
    python -m projects.amplitude_symbol.train --config-name=smoke
    python -m projects.amplitude_symbol.train --config-name=tiny_overfit
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

_NANOINFRA_ROOT = Path(__file__).resolve().parents[2]
if str(_NANOINFRA_ROOT) not in sys.path:
    sys.path.insert(0, str(_NANOINFRA_ROOT))

from core.utils import print0
from .blocks.training import run_training


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="tiny_overfit",
)
def main(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)

    cfg_name = getattr(cfg, "_metadata", None)
    cfg_label = cfg_name.object_type if cfg_name else "unknown"
    print0("=" * 72)
    print0(f"  AMPLITUDE SYMBOL TRAINING  [{cfg_label}]")
    print0("=" * 72)

    # Baseline: standard causal attention, no model modifications
    run_training(config)


if __name__ == "__main__":
    main()
