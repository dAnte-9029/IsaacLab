# PureRL Curriculum 3 Spatial Handoff

## Scope and status

PureRL C3 spatial support is implemented on branch `feat/native-multibody-rl`. The implementation adds project-local C3a, C3b, and C3c task registration without modifying `source/isaaclab_tasks/`. It preserves the measured CPU-native multibody plant, native holonomic wing mechanism, four direct actions, 60 Hz policy rate over 480 Hz physics, normalized 555-value actor observation, and no-wind authority boundary.

C3 engineering and runtime validation are complete. C2c is promoted through adjacent `model_525.pt` and
`model_550.pt` from `2026-08-12_17-12-48_pure_rl_c2c_seed0_resume500_to550`. The first C3a run learned the turn
grid but lost strong positive-slope C2c retention. C3a remains unpromoted.
The untrained C3 envelope now follows `ADR-2026-08-12-pure-rl-maneuver-envelope-v2.md`: C3b sequential vertical
events use 4--12 degrees, C3c coupled events use 3--10 degrees, and every C3 evaluation contract is v2.
C3c sampled vertical events alternate climb and descent direction so 2--4 event paths remain above ground from
the 10 m reset altitude; its deterministic gate uses one coupled vertical turn followed by one level turn.

A subsequent bounded actor-only distillation experiment copied the loaded C2c actor after weights-only warm start
and constrained its action mean only on C1/C2c rehearsal observations. Coefficient 0.05 preserved 100 percent C3a
success and improved iteration-99 C2c survival/climb/`+12` degree success to 95.54 percent, 89.58 percent, and
11/16. Five `+12` degree tilt terminations and a C1 tail-limit fraction of 11.64 percent remain, so C3a is still
unpromoted. Evidence is in `docs/audits/2026-08-13-pure-rl-c3a-actor-distillation.md`.

The follow-up default-disabled retention-aware scheduler changes future on-policy C1/C2c/C3a reset allocation
from completed-episode signals. A fresh 200-iteration seed-0 run with distillation coefficient 0.05 recovered the
C1 tail-limit gate by iteration 150 and preserved 100 percent C3a success. C2c peaked at iteration 175 with 93.75
percent survival, 85.42 percent climb success, and 9/16 `+12` degree success, then regressed at iteration 199. No
checkpoint passed C2c; `model_175.pt` is diagnostic only and C3a remains unpromoted. Evidence is in
`docs/audits/2026-08-14-pure-rl-c3a-retention-aware-sampling.md`.

## Implemented behavior

- C3a samples isolated level turns with 15 percent C1, 35 percent exact C2c, and 50 percent turn episodes. Exact
  C2c rehearsal reuses the C2c sampler and query; climb and descent are weighted 2:1 after a diagnostic localized
  retention failure to positive 12 degree tilt terminations.
- C3b adds seven sequential templates: same-direction turns, S-turns, turn then climb, turn then descent, climb then turn, descent then turn, and loiter. Curvature and slope do not overlap in current-stage C3b samples.
- C3c adds coupled turn with climb or descent events under the frozen elliptical demand bound.
- Paths are approximately 300 m long, sampled every 0.25 m, stored as batched Tensors, and queried through a bounded local projection window.
- The actor continues to receive five body-frame preview points with the existing 0.12--0.60 s horizon and remains exactly 555 values.
- Reward does not provide a target roll. During a turn it removes the zero-roll preference through 25 degrees, applies a quadratic soft penalty from 25 to 35 degrees, and adds a distinct termination at 35 degrees.
- Partial reset replaces only selected path rows, clears only their projection progress, and aligns reset yaw and initial velocity with the new path tangent.

## Task and evaluation contracts

| Stage | Task suffix | Fixed grid | Required retained evaluations |
| --- | --- | ---: | --- |
| C3a | `MeasuredPureRL-C3a-Direct-v0` | 96 | C1 and C2c |
| C3b | `MeasuredPureRL-C3b-Direct-v0` | 176 | C1, C2c, and C3a |
| C3c | `MeasuredPureRL-C3c-Direct-v0` | 96 | C1, C2c, C3a, and C3b |

The fixed grids use held-out geometry and deterministic heading/flap-phase schedules. Every current-stage grid requires at least 95 percent survival and event completion, at least 90 percent overall success, mean horizontal and vertical errors at most 0.5 m, P95 errors at most 1.5 m, reverse motion at most 1 percent, P95 absolute roll at most 25 degrees, zero roll-limit terminations, finite metrics, and complete stage-specific slices.

Promotion begins after 614,400 transitions and is evaluated every 307,200 transitions. With the 256-environment, 48-step default this maps to iteration 50 and a 25-iteration interval. Two adjacent current-stage passes are required. Every earlier suite must also pass its frozen gate, and its success rate may not fall by more than five percentage points from the corresponding promoted source baseline. Missing, duplicate, wrong-contract, wrong-checkpoint, non-finite, or incomplete evidence fails closed.

## Validation evidence

The 2026-08-13 rehearsal correction and bounded continuation are recorded in
`docs/audits/2026-08-13-pure-rl-c3a-c2c-rehearsal.md`. The corrected implementation passed 146 focused tests and
the CPU-native C3 runtime gate. Continued checkpoints 575 and 600 retained C1 and C3a, but each achieved only
66.67 percent C2c climb success, so neither is promotion eligible.

The final focused command passed:

```text
219 passed in 2.67s
```

The fresh-process CPU-native C3 runtime gate passed:

```text
tests/test_native_cpu_pure_rl_spatial_runtime_isaac.py: 1 passed, 3 warnings in 7.34s
```

The runtime gate used six C3c environments and verified CPU device ownership, disabled physics replication, active native constraints, exact fixed path schedules, finite `(6, 555)` observations, finite path and query Tensors, orthonormal route frames, C3a/C3b deterministic path generation, eight zero-action policy steps, spatial reward/termination telemetry, isolated partial resets, and the 25/30/35 degree helper behavior. The warnings were existing headless GLFW, Gym re-registration, USD schema, and final plugin-unload messages; pytest completed successfully.

Static compilation and diff checks are part of the final closeout. No PPO run was launched during implementation.

## Required checkpoint lineage

| Target | Required source stage | Retention authority |
| --- | --- | --- |
| C3a | promoted C2c | C1 and C2c |
| C3b | promoted C3a | C1, C2c, and C3a |
| C3c | promoted C3b | C1, C2c, C3a, and C3b |

Each target is a new run initialized with policy weights only. Do not restore the prior optimizer or iteration counter. `train_and_watch.py` records the source checkpoint path and SHA-256 in `curriculum_source.json` and rejects a missing or incorrect source stage.

## C3a warm-start command

Run only after replacing the placeholders with a formally promoted C2c run and checkpoint:

```bash
cd /home/zn/IsaacLab/.worktrees/native-multibody-rl
source /home/zn/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

TERM=xterm ./isaaclab.sh -p scripts/flapping_rl/train_and_watch.py \
  --task Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C3a-Direct-v0 \
  --run-name pure_rl_c3a_seed0 \
  --native-cpu \
  --num-envs 256 \
  --agent-num-mini-batches 16 \
  --max-iterations 500 \
  --save-interval 25 \
  --train-only \
  --seed 0 \
  --load_weights_only \
  --load_run C2C_RUN_DIRECTORY \
  --checkpoint C2C_PROMOTED_CHECKPOINT.pt \
  --source-stage c2c \
  --source-checkpoint-path /home/zn/IsaacLab/.worktrees/native-multibody-rl/logs/rsl_rl/flapping_bot_straight_flight/C2C_RUN_DIRECTORY/C2C_PROMOTED_CHECKPOINT.pt \
  --headless
```

Measured PureRL training defaults to no concurrent watcher; evaluate saved C3a checkpoints sequentially in fresh
processes. If `--concurrent-eval` is deliberately enabled for a non-authority experiment, the watcher selects
`pure_rl_spatial_c3a_v2` with 96 environments and 96 episodes unless explicitly overridden. Watcher output alone
does not establish promotion because separate C1 and C2c retention evidence must be supplied to
`evaluate_spatial_promotion()`.

## Remaining limitations

- C3a must be restarted weights-only from promoted C2c `model_550.pt` under the corrected rehearsal contract.
- The scheduler's online strong-climb signal does not yet match held-out post-recovery tilt failures closely enough.
- C3a convergence and sample requirements under the corrected curriculum remain unknown.
- No C3 checkpoint has passed the complete C1/C2c retention matrix.
- Same-process full environment reload remains unsupported; use fresh processes for runtime authority tests.
- Wind, direct-GPU plant equivalence, controller gain changes, reward-weight tuning, and real-flight qualification remain outside this handoff.
