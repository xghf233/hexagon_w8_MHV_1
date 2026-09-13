# NanoInfra Symbol

This copy now targets **Hexagon four-loop weight-8 0y coefficient prediction**.
Start with [the new Hexagon module](projects/hexagon_symbol_w8_0y/README.md) and
[the GPU-only handoff](../SERVER_HANDOFF.md). The new code is written but has NOT
been executed. The Heptagon/amplitude material below is retained snapshot context,
not this project's execution instructions or validation record; historical reports
and results were deliberately excluded from the copy.

Specialized NanoInfra workspace for scattering-amplitude symbol coefficient
prediction. This edition contains two scientific projects sharing one framework
core. This is the active research repository for the Symbol edition; the historical standard
edition is not included.

The existing five-loop `amplitude_symbol` project uses ten-letter words and D3
orbit-grouped splits. The new three-loop `heptagon_symbol` project uses six-letter
words from a 42-letter alphabet and random-row splits. Both use the existing GPT
mechanisms; the heptagon project has its own training orchestration and encoding.
Their split protocols and accuracy results must not be treated as equivalent.

## Repository contents

| Path | Purpose |
| --- | --- |
| [`core/`](core/) | GPT trunk, training loop, checkpointing, inference, and evaluation mechanisms |
| [`projects/amplitude_symbol/`](projects/amplitude_symbol/) | Symbol data pipeline, tokenizer, experiments, tests, reports, and curated results |
| [`projects/heptagon_symbol/`](projects/heptagon_symbol/) | Three-loop MHV conversion, random-row data pipeline, training/evaluation code, and imported server results |

Generated data, checkpoints, and run outputs are kept outside this repository;
artifact directories are not required to exist in the publication copy.

For current heptagon work, start with its [project README](projects/heptagon_symbol/README.md),
[progress record](projects/heptagon_symbol/PROGRESS.md)
and [server runbook](projects/heptagon_symbol/SERVER_RUNBOOK.md).

As of 2026-09-07, the server's Hydra configuration fix and first-run reports have
been imported into this repository. Supplied records show 73,008 updates,
13,657,600 parameters, and full val/test exact accuracy of 1.0 (46,725 rows each,
random-row split). All 20 source hashes in both retained checkpoint contracts
match this edition. Models and complete outputs remain outside Git.
See the [import audit](projects/heptagon_symbol/reports/SERVER_IMPORT_2026-09-07.md)
for provenance and limits; no model or unit tests were rerun locally.

## Environment

The Python package requires Python 3.12 or newer. Training and checkpoint evaluation
are intended for the server CUDA environment. The following installation example
is server-only and requires approval; use a confirmed existing environment when
available. Run from this edition directory.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

The five-loop `amplitude_symbol` project additionally requires the official
`AIAmplitudes_common_public` code and the five-loop `EZ_symb_new_norm` data file.
Install that checkout into the remote environment or add it to `PYTHONPATH`.
Their locations are external to this Git repository and can be supplied through
Hydra configuration or these environment variables:

- `NANOINFRA_SYMBOL_DATA_PATH`
- `NANOINFRA_AIAMPLITUDES_ROOT`
- `NANOINFRA_SYMBOL_MANIFEST_PATH`

If none are set, the resolver retains the historical remote-workspace layout as a
compatibility fallback.

The heptagon runtime instead reads the converted `words.npy`, `coefficients.npy`,
`splits.npz`, `metadata.json`, and `audit.json` via an explicit `data_dir`. It does
not require Wolfram, the original WXF, or the AIAmplitudes checkout at runtime.
Environment installation and all tests/training remain server-side operations,
subject to the repository's execution approval rules.

## Artifact policy

Git contains source code, configurations, small reports, compact summary JSON, and
selected figures. It does not contain raw datasets, generated split manifests,
checkpoint tensors, full output trees, logs, training-history JSONL, or per-sample
prediction dumps.

The two retained full-validation summaries report:

| Run | Exact accuracy | Whole-orbit accuracy |
| --- | ---: | ---: |
| Word-bidi 500K | 99.0033% | 98.9313% |
| Word-bidi 150K | 98.9351% | 98.7722% |

These are recorded observations under the saved evaluation protocol, not an
isolated causal comparison of architectural and training changes. Detailed reports
and provenance links are under
[`projects/amplitude_symbol/reports/`](projects/amplitude_symbol/reports/README.md).

## License

MIT — see [LICENSE](LICENSE).
