# PureRL Curriculum 2 Longitudinal Handoff

## Scope and current status

PureRL C2 longitudinal support is implemented on branch `feat/native-multibody-rl` but no C2 PPO run has been started. C1 remains the baseline task. C2 adds explicit C2a, C2b, and C2c task IDs while preserving the direct four-channel action, 555-value actor observation, 60 Hz policy rate, 480 Hz CPU PhysX, measured multibody plant, native holonomic wing mechanism, no-wind default, and all C1 reward weights and termination thresholds.

The C2 reward does not specify a target speed. It rewards signed velocity along the active three-dimensional path tangent and penalizes velocity along the lateral and vertical path normals. The path contains a level entry, a constant climb or descent, and an infinite level recovery.

## Frozen stage distributions

| Stage | Absolute slope | Level / climb / descent | Fixed promotion grid |
| --- | --- | --- | --- |
| C2a | 1.5--4 deg | 50 / 25 / 25 percent | 80 cases at 0, +/-2, +/-4 deg |
| C2b | 2--6 deg | 30 / 35 / 35 percent | 112 cases, adding +/-6 deg |
| C2c | 2--8 deg | 25 / 37.5 / 37.5 percent | 144 cases, adding +/-8 deg |

Every training episode samples a 15--20 m entry and a 20--30 m slope segment. Evaluation uses 17.5 m and 25 m. Each fixed grid combines its signed slopes with four headings and four initial flap phases. The separate 32-case +/-10 degree grid is diagnostic-only and must not select or promote a checkpoint.

## Qualification evidence

The focused non-Isaac suite passed 166 tests. Fresh processes also passed:

```text
tests/test_native_cpu_pure_rl_runtime_isaac.py: 1 passed in 13.63 s
tests/test_native_cpu_pure_rl_longitudinal_runtime_isaac.py: 1 passed in 7.10 s
```

The C2 runtime test used six CPU environments and verified active native constraints, deterministic level/climb/descent schedules, observation shape `(6, 555)`, preview altitude signs, orthonormal tangent/normal bases, finite rewards and telemetry, exact level-case C1 reward decomposition, partial reset isolation, and repeated reset completion.

Headless GLFW and final Isaac plugin-unload warnings appeared in both the existing C1 and new C2 processes. They did not affect pytest completion. Same-process environment reload remains unsupported; each authority run must start in a fresh process.

## Required checkpoint lineage

Each stage is a new run initialized with policy weights only. Do not restore the prior optimizer or iteration counter.

| Target | Required source stage | Warm-start mode |
| --- | --- | --- |
| C2a | `c1_straight` | selected C1 checkpoint |
| C2b | `c2a` | promoted C2a checkpoint |
| C2c | `c2b` | promoted C2b checkpoint |

`train_and_watch.py` records the exact source checkpoint path and SHA-256 in `curriculum_source.json`. The `--load-run` and `--checkpoint` values must resolve to the same file passed through `--source-checkpoint-path` under `logs/rsl_rl/flapping_bot_straight_flight/`.

## Launch commands

Run from this worktree after replacing the uppercase placeholders with an existing run directory name and exact checkpoint filename/path. The measured tasks automatically force CPU simulation, CPU policy optimization, 64 training environments, and the native extension. The explicit flags below make the authority boundary visible.

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
  --num-envs 64 \
  --max-iterations 2000 \
  --save-interval 100 \
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
  --num-envs 64 \
  --max-iterations 2000 \
  --save-interval 100 \
  --seed 0 \
  --load_weights_only \
  --load_run C2A_RUN_DIRECTORY \
  --checkpoint C2A_PROMOTED_CHECKPOINT.pt \
  --source-stage c2a \
  --source-checkpoint-path /home/zn/IsaacLab/.worktrees/native-multibody-rl/logs/rsl_rl/flapping_bot_straight_flight/C2A_RUN_DIRECTORY/C2A_PROMOTED_CHECKPOINT.pt \
  --headless
```

C2c after C2b promotion:

```bash
TERM=xterm ./isaaclab.sh -p scripts/flapping_rl/train_and_watch.py \
  --task Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C2c-Direct-v0 \
  --run-name pure_rl_c2c_seed0 \
  --native-cpu \
  --num-envs 64 \
  --max-iterations 2000 \
  --save-interval 100 \
  --seed 0 \
  --load_weights_only \
  --load_run C2B_RUN_DIRECTORY \
  --checkpoint C2B_PROMOTED_CHECKPOINT.pt \
  --source-stage c2b \
  --source-checkpoint-path /home/zn/IsaacLab/.worktrees/native-multibody-rl/logs/rsl_rl/flapping_bot_straight_flight/C2B_RUN_DIRECTORY/C2B_PROMOTED_CHECKPOINT.pt \
  --headless
```

## Evaluation and promotion procedure

The launcher saves and watches checkpoints every 100 PPO iterations. For C2a/C2b/C2c it automatically selects the corresponding 80/112/144-case fixed suite. Evaluation before iteration 200 is evidence but is not promotion-eligible.

For every candidate checkpoint at iteration 200 or later:

1. Require a complete C2 suite row with finite metrics.
2. Evaluate the same checkpoint on `c1_straight` and attach a matching C1 retention row.
3. Run `evaluate_longitudinal_promotion()` from `scripts/flapping_rl/pure_rl_longitudinal_promotion.py` with the ordered C2 rows, matching C1 rows, and the selected source-C1 baseline.
4. Require two adjacent passing checkpoints exactly 100 iterations apart.
5. Start the next stage from the later checkpoint returned by the promotion helper.

The C2 hard gates are overall survival at least 0.95, climb and descent success each at least 0.90, recovery reached at least 0.95, mean absolute cross-track and height error each at most 0.50 m, p95 absolute height error at most 1.50 m, reverse-motion fraction at most 0.01, and finite metrics. C1 retention additionally requires success at least 0.95, termination at most 0.05, score drop at most five, and mean errors within `max(2 * source baseline, 0.25 m)`.

The watcher intentionally records C2 selection as retention-incomplete until matching C1 evidence is supplied. Therefore, do not treat a watcher-side C2 score or a diagnostic +/-10 degree result as promotion by itself.

## Deferred work

- PPO convergence and stage durations are unverified.
- C1 retention evaluation still needs to be run for each candidate checkpoint.
- Wind, C3 lateral/composite geometry, direct-GPU screening, and reward-weight tuning are outside this handoff.
- CPU-native results are authoritative; no GPU plant equivalence is claimed.
