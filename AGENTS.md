# Hexagon local preparation and GPU-only execution rules

This directory is the new Hexagon four-loop, weight-8, 0y nonzero coefficient
project. The user renamed the previous hexagon_w6_MHV directory to
hexagon_w8_MHV because this training project starts at weight 8.
Current implementation lives in `nanoinfra-main_symbol/projects/hexagon_symbol_w8_0y/`.
Read README.md, STATIC_REVIEW.md and SERVER_HANDOFF.md before continuing this task.

## Current scope

- The user authorized local code/configuration/documentation preparation for a
  GPU smoke-ready candidate. The current code has NOT been executed or validated.
- Use random-row PCG64 seed=42, C4=32*c4, base100 fixed TWO blocks,
  model 4/256/4/4, and explicit word attention [1,9). Do not change these silently.
- Do not run Python checks, CPU tests, smoke tests, model inference, training or
  benchmarks on the Mac. Do not create/activate/install a local Python environment.
- The user explicitly excluded a standalone CPU verification phase anywhere.
  All model checks and compute must use the approved Linux bf16 CUDA GPU server;
  host file I/O, preprocessing, format guards, logging and RNG handling remain
  necessary parts of that GPU workflow. Never fall back to CPU or MPS models.
- Server details will be supplied later. Preparing commands is not permission
  to connect to the GPU server, upload data, install dependencies or start
  remote computation now. GitHub code publication is a separate operation.
- The user authorized this directory rename and the initial Git commit/push to
  https://github.com/xghf233/hexagon_w8_MHV_1. Do not change that destination,
  repository visibility or the sibling Heptagon repositories.

## Preserve existing work

- Do not modify the sibling Heptagon projects or HZQ-git as an implicit sync.
- Do not modify existing tools/ or source Symbol_Data artifacts: their exact
  extraction/scaling provenance is hash-locked. Do not rescale x32 data again.
- Keep the copied core/ and shared amplitude attention unchanged unless a
  concrete framework-level change is separately explained and authorized.
- Generated arrays/checkpoints/reports stay outside this code directory and
  outside immutable source directories. New runs use new output directories.
- Never delete old runs/checkpoints automatically, overwrite source data, commit
  secrets, or copy .venv/data/logs as part of a source-code snapshot.
- Inspect Git status before subsequent work. Future commits/pushes need their
  own explicit authorization; preserve unrelated dirty-worktree changes.

## Reporting and next execution

- Written/static-reviewed/executed/passed are distinct states. Do not adopt
  copied historical Heptagon results or CPU runbook steps as Hexagon evidence.
- First execution is gpu_smoke only: data preparation, CUDA contracts, continuous
  20 updates and a separate 10+10 resume comparison. It does not run final test.
- Tiny-overfit, compiled smoke, formal training, extra seeds and final test are
  separate later phases, not automatically authorized by a smoke request.
- On failure preserve artifacts and report the cause; do not weaken hashes,
  alter splits or loosen numerical tolerances merely to make a report pass.
