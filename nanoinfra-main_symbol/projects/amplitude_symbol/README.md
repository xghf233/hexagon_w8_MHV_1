# Amplitude Symbol Prediction

This project trains decoder-only transformers to predict integer coefficients of
the five-loop scattering-amplitude symbol. Data splits are grouped by complete
dihedral (D3) orbits so that related words do not cross train, validation, and test
boundaries.

The current main experiment uses word-bidirectional attention: the ten input-symbol
positions can attend to one another, while coefficient generation remains causal.
The repository contains the implementation and compact evidence needed to review
the experiment. Raw data, checkpoints, per-sample predictions, and complete run
directories remain outside Git.

## Current validation results

These are saved observations from full deterministic free-generation evaluation on
26,388 validation samples spanning 4,398 D3 orbits. They do not isolate attention,
batch size, compilation, and training length as independent causal factors.

| Run | Steps | Exact | Magnitude | Sign | Orbit consistency | Whole-orbit accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Word-bidi 500K | 499,999 | 99.0033% | 99.0791% | 99.8370% | 99.4998% | 98.9313% |
| Word-bidi 150K | 149,999 | 98.9351% | 99.1436% | 99.6551% | 99.4316% | 98.7722% |

Machine-readable summaries are in [`results/`](results/README.md).

## Layout

| Path | Purpose |
| --- | --- |
| `blocks/` | Symbol-specific attention and shared training assembly |
| `configs/` | Baseline, smoke, and overfit configurations |
| `experiments/word_bidi/` | Word-bidirectional training entry point and configurations |
| `tests/` | Unit tests for data, tokenizer, splits, and evaluation |
| `docs/` | Project plan and historical progress record |
| `reports/` | Reviewed experiment reports and selected figures |
| `results/` | Small final evaluation and error-analysis summaries |

## Entry points

Run commands from the `nanoinfra-main_symbol` root in the configured remote GPU
environment.

```bash
# Causal-attention baseline
python -m projects.amplitude_symbol.train --config-name=baseline_orbit

# Main word-bidirectional experiment (500K config by default)
python -m projects.amplitude_symbol.experiments.word_bidi.train

# Use the 150K configuration
python -m projects.amplitude_symbol.experiments.word_bidi.train \
  --config-name=word_bidi_150k

# Evaluate an external checkpoint
python -m projects.amplitude_symbol.eval_checkpoint \
  --checkpoint models/amplitude_symbol/word_bidi/step_499999 \
  --attention-mode word_bidirectional --split val --save-predictions
```

External resources can be supplied with `data.path`, `data.aiamplitudes_root`, and
`data.manifest_path` in Hydra configuration, or with
`NANOINFRA_SYMBOL_DATA_PATH`, `NANOINFRA_AIAMPLITUDES_ROOT`, and
`NANOINFRA_SYMBOL_MANIFEST_PATH`. Running training regenerates the ignored
`MANIFEST.json` from the data and split seed when it is absent.

New checkpoints record their runtime attention mode and word interval. The two
retained legacy checkpoints predate that metadata, so evaluation must state
`--attention-mode word_bidirectional` explicitly as shown above.

## Documentation

- [Project plan](docs/PLAN.md)
- [Historical progress record](docs/PROGRESS.md)
- [Baseline report](reports/BASELINE_REPORT.md)
- [Word-bidirectional 500K report](reports/WORD_BIDI_REPORT.md)
- [Word-bidirectional 150K report](reports/WORD_BIDI_150K_REPORT.md)
- [500K versus 150K error analysis](reports/ERROR_ANALYSIS.md)

See [the reports index](reports/README.md) for the selected figures and evidence
boundary.
