# PureRL curriculum-1 checkpoint evaluation

## Material Passport

- Date: 2026-08-06
- Origin skill: `academic-research-suite`, experiment-agent
- Branch: `feat/native-multibody-rl`
- Baseline commit: `e707e816`
- Task: `Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-Direct-v0`
- Source run: `2026-08-06_18-57-36_curriculum1_default_smoke`
- Evaluation contract: `pure_rl_curriculum1_v1`
- Evaluation suite: `pure_rl_curriculum1_nowind_v1`
- Runtime: CPU native measured-wing multibody plant, 480 Hz physics and 60 Hz policy
- Verification status: completed in one fresh process
- Claim boundary: evaluator integration and two-iteration smoke checkpoint quality only

## Objective

Replace the legacy target-speed watcher for the canonical MeasuredPureRL task
with a route-relative curriculum-1 evaluator and apply its pre-registered
success gate to the two existing launch-smoke checkpoints. No new training,
reward change, PPO change or plant change was authorized or performed.

## Fixed evaluation contract

Each checkpoint was evaluated in 16 parallel environments. The environments
covered all 16 combinations of four route headings and four initial flap
phases. Each environment contributed exactly one episode. The watcher checked
the realized reset tensors against the registered grid before stepping.

The success gate required at least 80 percent timeouts, at most 20 percent
terminations, mean absolute route-relative cross-track and height error no
larger than 0.5 m, positive mean progress, frequency-limit occupancy no larger
than 0.25 and tail-limit occupancy no larger than 0.10. The evaluator contains
no target-speed error.

## Command

```bash
source /home/zn/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export TERM=xterm-256color
export PYTHONPATH="$PWD/source/flapping_bot${PYTHONPATH:+:$PYTHONPATH}"
./isaaclab.sh -p scripts/flapping_rl/watch_and_eval.py \
  --task Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-Direct-v0 \
  --log_dir logs/rsl_rl/flapping_bot_straight_flight/2026-08-06_18-57-36_curriculum1_default_smoke \
  --device cpu --eval_suite pure_rl_curriculum1_nowind_v1 \
  --num_envs 16 --episodes 16 --poll_s 5 --once \
  --kit_args "--portable-root logs/portable/pure_rl_curriculum1_eval_20260806_v1 \
    --ext-folder $PWD/source/flapping_bot/native_extensions \
    --enable omni.flapping_bot.holonomic_constraint" \
  --headless
```

## Results

| Metric | `model_0.pt` | `model_1.pt` |
|---|---:|---:|
| Episodes / heading-phase pairs | 16 / 16 | 16 / 16 |
| Success gate | fail | fail |
| Timeout rate | 0.0000 | 0.0000 |
| Termination rate | 1.0000 | 1.0000 |
| Mean episode duration | 1.4323 s | 1.6073 s |
| Mean along-track progress | 11.6974 m | 13.0964 m |
| Mean along-track velocity | 8.2046 m/s | 8.1769 m/s |
| Reverse-motion fraction | 0.0000 | 0.0000 |
| Mean absolute cross-track error | 0.2784 m | 0.3957 m |
| Mean absolute height error | 0.6966 m | 0.6945 m |
| Mean maximum tilt | 65.9370 deg | 65.9267 deg |
| Mean body angular rate | 2.1757 rad/s | 2.0605 rad/s |
| Mean actual flap frequency | 2.6042 Hz | 2.5310 Hz |
| Frequency-limit fraction | 0.0000 | 0.0000 |
| Tail-limit fraction | 0.0000 | 0.0000 |
| Mean normalized action delta | 0.00853 | 0.00939 |
| Diagnostic score | 33.7545 | 31.7290 |

`model_0.pt` termination causes were 56.25 percent tilt, 6.25 percent
cross-track and 43.75 percent height error. `model_1.pt` recorded 62.5 percent
tilt, 6.25 percent cross-track and 37.5 percent height error. Cause rates can
overlap on a terminal step; each checkpoint had one episode where two limits
were crossed together. Neither checkpoint had a ground termination.

Both checkpoints fail the primary survival gate and the mean height-error
gate. They do not represent stable straight-flight controllers. The zero
frequency-limit and tail-limit fractions show that failure was not caused by
continuous occupancy of the configured action limits.

The selection contract names `model_1.pt` as the best available fallback
because both checkpoints fail the gate and tie on timeout and termination,
after which its longer mean episode duration has priority. Its lower diagnostic
score does not override the survival-oriented ordering. This selection is not
a success claim.

## Artifact checks

The versioned rows were appended to `eval/summary.csv`; legacy rows remain but
have an empty evaluation-contract field and were excluded from selection.
`eval/model_0.json`, `eval/model_1.json` and `eval/best_checkpoint.json` were
rewritten with the new contract. Both JSON files contain a case row and suite
row, all numeric PureRL metrics are finite, each reports 16 episodes and 16
heading-phase pairs, and neither contains `mean_abs_vx_err`.

## Verification

- PureRL-related non-simulator suite: `117 passed`.
- CPU native Isaac runtime test: `1 passed`; evaluation caches, rewards,
  observations, mechanism state and telemetry remained finite.
- Relevant Python files passed `py_compile`.
- `git diff --check` passed before the formal evaluation.

Isaac Sim emitted existing headless-window, generated-schema, CPU `powersave`
governor and PCIe-width warnings. They did not stop the runtime gate or formal
evaluation. Because the Isaac Lab wrapper does not reliably communicate a
child Python failure, completion was verified from artifact timestamps,
versioned CSV rows and JSON content rather than wrapper status alone.

## Conclusion

The evaluation engineering loop is closed: the canonical task now has a
route-relative, speed-free, deterministic and versioned curriculum-1 gate.
The existing two-iteration smoke remains valid as a launch and optimizer test,
but it does not pass the new flight-quality gate. The next experiment should
therefore be a bounded multi-seed curriculum-1 PPO learnability run with this
evaluator frozen in advance.
