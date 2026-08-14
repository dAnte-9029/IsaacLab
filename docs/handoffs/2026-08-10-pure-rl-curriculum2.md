# PureRL Curriculum 2 Longitudinal Handoff

> 2026-08-12 update: C2b is promoted through `model_475.pt` from
> `2026-08-12_10-45-45_pure_rl_c2b_seed0`. The measured PureRL launcher defaults below use the validated
> sample-equivalent 256/16/500/25 training configuration.
> The untrained C2c envelope was subsequently increased to 4--12 degrees under
> `ADR-2026-08-12-pure-rl-maneuver-envelope-v2.md`; C2a/C2b evidence is unchanged.

## Scope and current status

PureRL C2 longitudinal support is implemented on branch `feat/native-multibody-rl`. C2a and C2b have completed
promotion. `model_475.pt` from `2026-08-12_10-45-45_pure_rl_c2b_seed0` is the accepted C2c source. C1 remains
the retention baseline. C2 adds explicit C2a, C2b, and C2c task IDs while preserving the
direct four-channel action, 555-value actor observation, 60 Hz policy rate, 480 Hz CPU PhysX, measured multibody
plant, native holonomic wing mechanism, no-wind default, and all C1 reward weights and termination thresholds.

The C2 reward does not specify a target speed. It rewards signed velocity along the active three-dimensional path tangent and penalizes velocity along the lateral and vertical path normals. The path contains a level entry, a constant climb or descent, and an infinite level recovery.

## Frozen stage distributions

| Stage | Absolute slope | Level / climb / descent | Fixed promotion grid |
| --- | --- | --- | --- |
| C2a | 1.5--4 deg | 50 / 25 / 25 percent | 80 cases at 0, +/-2, +/-4 deg |
| C2b | 2--6 deg | 30 / 35 / 35 percent | 112 cases, adding +/-6 deg |
| C2c | 4--12 deg | 25 / 37.5 / 37.5 percent | 112 cases at 0, +/-4, +/-8, +/-12 deg |

Every training episode samples a 15--20 m entry and a 20--30 m slope segment. Evaluation uses 17.5 m and 25 m. Each fixed grid combines its signed slopes with four headings and four initial flap phases. C2a/C2b retain the 32-case +/-10-degree diagnostic; C2c uses a 32-case +/-15-degree diagnostic. Diagnostics cannot select or promote a checkpoint.

## Qualification evidence

The focused non-Isaac suite passed 166 tests. Fresh processes also passed:

```text
tests/test_native_cpu_pure_rl_runtime_isaac.py: 1 passed in 13.63 s
tests/test_native_cpu_pure_rl_longitudinal_runtime_isaac.py: 1 passed in 7.10 s
```

The C2 runtime test used six CPU environments and verified active native constraints, deterministic level/climb/descent schedules, observation shape `(6, 555)`, preview altitude signs, orthonormal tangent/normal bases, finite rewards and telemetry, exact level-case C1 reward decomposition, partial reset isolation, and repeated reset completion.

Headless GLFW and final Isaac plugin-unload warnings appeared in both the existing C1 and new C2 processes. They did not affect pytest completion. Same-process environment reload remains unsupported; each authority run must start in a fresh process.

On 2026-08-12, fresh current-contract evaluation completed the 112-case C2b fixed grid for `model_450.pt`,
`model_475.pt`, and `model_499.pt`; all three passed. The sample-equivalent adjacent `model_450.pt` and
`model_475.pt` checkpoints also passed the 16-case C1 retention suite with 100 percent success and zero
termination. The promotion helper returned `model_475.pt`. Full metrics and artifact paths are recorded in
`docs/audits/2026-08-12-pure-rl-c2b-promotion.md`.

## Required checkpoint lineage

Each stage is a new run initialized with policy weights only. Do not restore the prior optimizer or iteration counter.

| Target | Required source stage | Warm-start mode |
| --- | --- | --- |
| C2a | `c1_straight` | selected C1 checkpoint |
| C2b | `c2a` | promoted C2a checkpoint |
| C2c | `c2b` | promoted C2b checkpoint |

`train_and_watch.py` records the exact source checkpoint path and SHA-256 in `curriculum_source.json`. The `--load-run` and `--checkpoint` values must resolve to the same file passed through `--source-checkpoint-path` under `logs/rsl_rl/flapping_bot_straight_flight/`.

## Launch commands

Run from this worktree after replacing the uppercase placeholders with an existing run directory name and exact
checkpoint filename/path. The measured tasks automatically force CPU simulation, CPU policy optimization, 256
training environments, 16 minibatches, 500 iterations, a 25-iteration save interval, and the native extension.
They also default to sequential checkpoint evaluation. The explicit `--train-only` flags below make the authority
boundary independent of launcher-default changes.

```bash
cd /home/zn/IsaacLab/.worktrees/native-multibody-rl
source /home/zn/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
```

C2a from the selected C1 checkpoint:

```bash
TERM=xterm ./isaaclab.sh -p scripts/flapping_rl/train_and_watch.py \
  --task Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C2a-Direct-v0 \
  --run-name pure_rl_c2a_seed0 \
  --native-cpu \
  --num-envs 256 \
  --agent-num-mini-batches 16 \
  --max-iterations 500 \
  --save-interval 25 \
  --train-only \
  --seed 0 \
  --load_weights_only \
  --load_run C1_RUN_DIRECTORY \
  --checkpoint C1_CHECKPOINT.pt \
  --source-stage c1_straight \
  --source-checkpoint-path /home/zn/IsaacLab/.worktrees/native-multibody-rl/logs/rsl_rl/flapping_bot_straight_flight/C1_RUN_DIRECTORY/C1_CHECKPOINT.pt \
  --headless
```

C2b after C2a promotion:

```bash
TERM=xterm ./isaaclab.sh -p scripts/flapping_rl/train_and_watch.py \
  --task Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C2b-Direct-v0 \
  --run-name pure_rl_c2b_seed0 \
  --native-cpu \
  --num-envs 256 \
  --agent-num-mini-batches 16 \
  --max-iterations 500 \
  --save-interval 25 \
  --train-only \
  --seed 0 \
  --load_weights_only \
  --load_run C2A_RUN_DIRECTORY \
  --checkpoint C2A_PROMOTED_CHECKPOINT.pt \
  --source-stage c2a \
  --source-checkpoint-path /home/zn/IsaacLab/.worktrees/native-multibody-rl/logs/rsl_rl/flapping_bot_straight_flight/C2A_RUN_DIRECTORY/C2A_PROMOTED_CHECKPOINT.pt \
  --headless
```

C2c from the accepted C2b checkpoint:

```bash
TERM=xterm ./isaaclab.sh -p scripts/flapping_rl/train_and_watch.py \
  --task Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C2c-Direct-v0 \
  --run-name pure_rl_c2c_seed0 \
  --native-cpu \
  --num-envs 256 \
  --agent-num-mini-batches 16 \
  --max-iterations 500 \
  --save-interval 25 \
  --train-only \
  --seed 0 \
  --load_weights_only \
  --load_run 2026-08-12_10-45-45_pure_rl_c2b_seed0 \
  --checkpoint model_475.pt \
  --source-stage c2b \
  --source-checkpoint-path /home/zn/IsaacLab/.worktrees/native-multibody-rl/logs/rsl_rl/flapping_bot_straight_flight/2026-08-12_10-45-45_pure_rl_c2b_seed0/model_475.pt \
  --headless
```

## Evaluation and promotion procedure

The launcher saves checkpoints every 25 PPO iterations under the 256-environment default. Measured PureRL
authority runs do not start a concurrent watcher; evaluate the saved checkpoints sequentially in fresh processes.
If `--concurrent-eval` is deliberately enabled for a non-authority experiment, C2a/C2b/C2c automatically select
the corresponding 80/112/112-case fixed suite. Sample-equivalent promotion eligibility begins after 614,400
transitions, which maps to iteration 50 for 256 environments.

For every candidate checkpoint at iteration 50 or later under the 256-environment default:

1. Require a complete C2 suite row with finite metrics.
2. Evaluate the same checkpoint on `c1_straight` and attach a matching C1 retention row.
3. Run `evaluate_longitudinal_promotion()` from `scripts/flapping_rl/pure_rl_longitudinal_promotion.py` with the ordered C2 rows, matching C1 rows, and the selected source-C1 baseline.
4. Require two adjacent passing checkpoints exactly 25 iterations apart.
5. Start the next stage from the later checkpoint returned by the promotion helper.

The C2 hard gates are overall survival at least 0.95, climb and descent success each at least 0.90, recovery reached at least 0.95, mean absolute cross-track and height error each at most 0.50 m, p95 absolute height error at most 1.50 m, reverse-motion fraction at most 0.01, and finite metrics. C1 retention additionally requires success at least 0.95, termination at most 0.05, score drop at most five, and mean errors within `max(2 * source baseline, 0.25 m)`.

Concurrent watcher output, when explicitly enabled, records C2 selection as retention-incomplete until matching
C1 evidence is supplied. Therefore, do not treat a watcher-side C2 score or a stage diagnostic result as
promotion by itself.

## Deferred work

- C2c convergence and stage duration remain unverified.
- C1 retention evaluation is still required for each C2c promotion candidate.
- Wind, C3 lateral/composite geometry, direct-GPU screening, and reward-weight tuning are outside this handoff.
- CPU-native results are authoritative; no GPU plant equivalence is claimed.
