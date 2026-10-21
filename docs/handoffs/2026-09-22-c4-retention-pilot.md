# C4 retention pilot — 2026-09-22

## Objective and current status

Train one zero-wind policy for feasible continuous paths while preserving old task
capabilities. The user authorized implementing and executing the literature-derived
bounded plan. Work is in `.worktrees/native-multibody-rl`, branch
`feat/native-multibody-rl`, base HEAD `41ea8565`. Existing dirty work is preserved;
no commit or push was requested. Implementation and runtime preflight are complete.
The formal sequential experiment completed normally: all four arms and all scheduled
evaluations finished, but no checkpoint passed all six suites. See
`../audits/2026-09-22-c4-retention-diagnosis.md` for results, fixed-state action/memory
analysis and next experiment recommendations. No additional training was launched.

The September 10 handoff was stale: the original 201-iteration C4 run completed.
Development tracking was 96/96 at 25, 5/96 at 50 and 0/96 at 75--200 despite survival.
This establishes regression, not its exclusive cause. The new diagnostics examine
task-gradient interference, tail-channel interference, policy KL and critic errors.

## Frozen comparison

See `../decisions/ADR-2026-09-22-c4-retention-curriculum.md` for assumptions and sources.

| Arm | C4 sampler | Frozen behavior memory |
| --- | --- | --- |
| A | Existing fixed distribution | None |
| B | Four-bin adaptive distribution | None |
| C | Existing fixed distribution | Actor mean MSE, coefficient 1.0 |
| D | Four-bin adaptive distribution | Actor mean MSE, coefficient 1.0 |

All arms use seed 0, 101 iterations, 256 CPU-native environments, 48 rollout steps,
16 minibatches, identical six-family task weights and fresh optimizer state from
the full-C3 `model_175.pt` source. This source passed all legacy suites once but lacks
an adjacent passing pair and is research-only. The controller observation/action,
reward, physics and default task contracts are unchanged.

The curriculum changes only C4 geometry sampling inside the existing mild bounds;
it does not change outer-family PPO target weights. Each bin retains at least 10%
probability. Updates require completed episodes in every bin and fresh evidence;
the maximum probability change is 0.05 per update. Bin IDs 100--103 occupy a separate
diagnostic namespace from old hard-slice IDs and are not actor inputs.

The memory was collected from new deterministic teacher training rollouts with seed
982451653, 256 environments and 2,400 steps (614,400 transitions). It contains 4,096
observation/action-mean pairs per family, 24,576 total. Only completed successful
episodes enter the bounded reservoirs. Memory replay contributes an auxiliary actor
mean loss; it is not off-policy PPO, full CLEAR, critic replay or variance protection.
Recovery states were not deliberately injected. The coefficient is a declared pilot
choice, not a validated optimum.

## Execution and artifacts

Root for paths below: `/home/zn/IsaacLab/.worktrees/native-multibody-rl`.

- Service: `c4-retention-pilot-20260922.service` (user systemd, sequential workers).
- Experiment: `logs/continual_c4/20260922_pilot/`.
- Memory: `logs/continual_c4/20260922_memory/memory.pt`, with provenance in `summary.json`.
- Preflight logs and preexisting tracked diff: `logs/continual_c4/20260922_implementation/`.
- Pilot `contract.json`, `source_manifest.json`, `source_snapshot/` and
  `working_tree.patch` freeze inputs and implemented sources, including untracked code.
- Each arm has `training/status.json`, `training/summary.json`, checkpoint files,
  `update_diagnostics.jsonl`, and episode metrics; B/D also log curriculum evolution.
- At checkpoints 50/75/100, evaluation runs C4 development plus C3c/C3b/C3a/C2c/C1
  in fresh processes. An adjacent passing pair is only a development candidate.

Inspect durable progress with:

```bash
systemctl --user status c4-retention-pilot-20260922.service --no-pager
cat logs/continual_c4/20260922_pilot/status.json
cat logs/continual_c4/20260922_pilot/summary.json
```

The pipeline stops on infrastructure errors or changed project sources. Do not edit
project Python/C++/TOML while it is running. A failed scientific gate is recorded and
does not suppress the remaining retention suites or comparison arms. The total planned
training budget is 4,964,352 transitions, excluding memory collection and evaluation.
There is no automatic holdout access, distribution expansion, wind addition or promotion.
After completion, inspect per-suite errors and candidate pairs before choosing a method;
seed 0 alone cannot establish reproducibility or general arbitrary-path capability.

## Validation evidence

- 128 pure/contract tests passed (`unit_tests_final.log`, 15.09 s).
- Fresh CPU-native curriculum-label and partial-reset isolation test passed
  (`native_runtime.log`); remaining rows retain their paths and observations stay finite.
- B and D completed five-iteration real training smokes and one-iteration fresh-process
  resumes. D actor displacement was 0.016640 for the smoke and 0.008835 after resume;
  memory auxiliary calls advanced from 8 to 12. Optimizer state has 19 entries.
- The short smokes end before a full episode, so their zero curriculum updates are
  expected. Completed-episode adaptation still needs confirmation in the formal B/D run.
- Resume restores optimizer, scheduling, private sampler/memory RNG and diagnostic
  counters; simulation episodes restart, so it is not bitwise trajectory continuation.
- `git diff --check` passed. No promotion, holdout or wind result is claimed.

## Main implementation locations

- `path_tracking/c4_curriculum.py`: bounded adaptive sampler and serialized history.
- `px4_like/c4_retention.py`: terminal-safe training tracker, memory auxiliary and diagnostics.
- `analysis/c4_training.py`: opt-in variants, collection and resume integration.
- `analysis/c4_retention_pilot.py`: frozen sequential comparison and result aggregation.
- `scripts/flapping_px4/run_c4_retention_pilot.py`: durable-run CLI.
- `tests/test_c4_retention.py`, `tests/test_native_cpu_c4_retention_runtime_isaac.py`.
