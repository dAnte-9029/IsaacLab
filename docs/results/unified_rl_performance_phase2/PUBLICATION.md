# Phase 2 publication scope

Published: implementation, focused tests, Chinese report, reproduction/comparison scripts,
summary and consistency JSON, per-run parameters/identity/completion/timings,
small runner summaries/contracts, original benchmark harness snapshots, and final test logs.

Intentionally local only (preserved in this result directory):

- `*/run/evidence/`, `*/evidence/`: complete SQLite databases and any WAL/SHM files.
- `*/replay.npz`: full deterministic action/force/state replay arrays.
- `evaluation_inputs/*.pt`: evaluation-only frozen snapshot copies; source/budget metadata is published.
- `*/run/assets/`, `*/assets/`: generated simulator assets.
- `*/run/source_snapshot/`, `*/source_snapshot/`: per-run source copies.
- `*/resources.jsonl`, other step/event JSONL, console/replay logs and intermediate diagnostic logs.

Raw evidence remains available locally; a Git checkout alone does not contain it.
The report describes the pre-publication validation state. No formal training was run.
Use `scripts/flapping_px4/run_unified_phase2.py` with the documented existing baseline/plant
worktrees and Conda environment to regenerate full evidence in a new directory.
