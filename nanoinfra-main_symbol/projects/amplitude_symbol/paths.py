"""Portable path resolution for the amplitude-symbol project.

Explicit configuration wins over environment variables, which win over the
historical NanoInfra workspace layout. Resolving a path does not create or read it;
callers remain responsible for validating the resource they need.
"""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parents[1]

DATA_PATH_ENV = "NANOINFRA_SYMBOL_DATA_PATH"
UPSTREAM_ROOT_ENV = "NANOINFRA_AIAMPLITUDES_ROOT"
MANIFEST_PATH_ENV = "NANOINFRA_SYMBOL_MANIFEST_PATH"

DEFAULT_DATA_PATH = (
    WORKSPACE_ROOT / "data" / "aiamplitudes" / "raw" / "EZ_symb_new_norm"
)
DEFAULT_UPSTREAM_ROOT = WORKSPACE_ROOT / "upstream" / "AIAmplitudes_common_public"
DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parent / "MANIFEST.json"


def _resolve(explicit: str | Path | None, env_name: str, default: Path) -> Path:
    raw = explicit if explicit not in (None, "") else os.environ.get(env_name)
    if raw in (None, ""):
        return default
    return Path(os.path.expandvars(str(raw))).expanduser()


def resolve_data_path(explicit: str | Path | None = None) -> Path:
    """Resolve the raw five-loop symbol file path."""
    return _resolve(explicit, DATA_PATH_ENV, DEFAULT_DATA_PATH)


def resolve_upstream_root(explicit: str | Path | None = None) -> Path:
    """Resolve the ``AIAmplitudes_common_public`` checkout root."""
    return _resolve(explicit, UPSTREAM_ROOT_ENV, DEFAULT_UPSTREAM_ROOT)


def resolve_manifest_path(explicit: str | Path | None = None) -> Path:
    """Resolve the generated split-manifest path."""
    return _resolve(explicit, MANIFEST_PATH_ENV, DEFAULT_MANIFEST_PATH)
