# Phase 4 published results

Published: `summary.json` (all cases, resource summaries, evidence validation and replay parity), `ipc.json`, `thermal_summary.json`, and `topology.json`.

Raw per-case directories remain local in this worktree and are excluded from Git: SQLite databases/WAL, source snapshots, assets, resource time series, console logs, replay NPZ arrays, and `ipc_source/w0/real_rollout.pt`. These occupy approximately 3.7 GiB. No raw artifact was deleted for publication. Training checkpoints had already been deleted by the benchmark; the remaining rollout payload is benchmark-only and must not be used for formal recovery.

Paths in JSON refer to the original local run. Recomputing the summary requires the local raw directories; a fresh checkout can reproduce new runs with the command in `../../unified_rl_performance_phase4_zh.md`, using the documented Isaac Sim environment and external plant assets.
