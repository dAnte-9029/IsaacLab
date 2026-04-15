# Estimated Teacher Tuning Handoff

## Commit Baseline

- Working baseline commit: `17e3a40c`
- Commit message: `feat: add dual-teacher imu state-source flow`

Start new work from this commit.

## What Is Already Done

### 1. Dual-teacher runtime contract is explicit

The repo now supports explicit runtime separation between:

- `teacher_state_source`: `truth` or `estimated`
- `policy_state_source`: `truth` or `estimated`
- `imu_source`: `synthetic` or `isaacsim`

Key files:

- [state_source_contract.py](/home/zn/IsaacLab/source/flapping_bot/flapping_bot/direct/flapping_bot/state_source_contract.py)
- [straight_flight_env.py](/home/zn/IsaacLab/source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py)
- [path_tracking_env.py](/home/zn/IsaacLab/source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py)

### 2. Non-RL controller scripts support estimated teacher with either synthetic or live Isaac IMU

Key files:

- [fly_straight_line.py](/home/zn/IsaacLab/scripts/flapping_px4/fly_straight_line.py)
- [fly_loiter.py](/home/zn/IsaacLab/scripts/flapping_px4/fly_loiter.py)
- [imu_provider.py](/home/zn/IsaacLab/source/flapping_bot/flapping_bot/px4_like/imu_provider.py)
- [isaacsim_imu_adapter.py](/home/zn/IsaacLab/source/flapping_bot/flapping_bot/px4_like/isaacsim_imu_adapter.py)
- [state_estimation.py](/home/zn/IsaacLab/source/flapping_bot/flapping_bot/px4_like/state_estimation.py)

### 3. Live Isaac IMU late-binding bug is fixed

Root cause:

- Isaac Lab `Imu` uses lazy play-time initialization.
- These controller scripts create the IMU after the env is already running.
- Without manual init, `sensor.reset()` crashed because `_timestamp` did not exist yet.

Current fix:

- In `fly_straight_line.py` and `fly_loiter.py`, if a freshly created IMU sensor is not initialized, call `_initialize_impl()` once and mark `_is_initialized = True` before `reset()` / `update()`.

This is a runtime lifecycle fix, not a controller tuning fix.

### 4. RL entrypoints no longer default path-tracking eval to truth teacher

Key files:

- [train_and_watch.py](/home/zn/IsaacLab/scripts/flapping_rl/train_and_watch.py)
- [eval_suites.py](/home/zn/IsaacLab/scripts/flapping_rl/eval_suites.py)
- [path_tracking_eval_common.py](/home/zn/IsaacLab/scripts/flapping_rl/path_tracking_eval_common.py)

Current policy:

- RL launch defaults inject:
  - `teacher_state_source=estimated`
  - `policy_state_source=estimated`
  - `imu_source=synthetic`
- Truth suites are still available as forensic baselines.

## What The New Agent Should Do

The next task is **not** more IMU plumbing. The next task is **estimated-teacher controller tuning**.

### Required high-level rule

- Keep `truth teacher` behavior fixed as the forensic upper-bound baseline.
- Tune `estimated teacher` separately.
- Prefer separate parameter sets over duplicating controller logic.

### Practical tuning sequence

1. Freeze `truth teacher`
2. Tune `estimated + synthetic` first
3. Use `estimated + isaacsim_live` only as the final convergence check

Reason:

- `estimated + synthetic` is more repeatable.
- If that path is still poor, live Isaac IMU only adds phase/noise confounders.

## Current Evidence

### Straight-line smoke results

- truth:
  - [summary.json](/home/zn/IsaacLab/logs/flapping_px4/straight_line/20260412_165617/summary.json)
  - mean abs track error: `0.058 m`
- estimated + synthetic:
  - [summary.json](/home/zn/IsaacLab/logs/flapping_px4/straight_line/20260412_165649/summary.json)
  - mean abs track error: `0.253 m`
  - mean estimated XY pos error: `0.916 m`
  - mean estimated vel XYZ error: `0.557 m/s`
  - mean estimated yaw error: `0.752 deg`
- estimated + isaacsim_live:
  - [summary.json](/home/zn/IsaacLab/logs/flapping_px4/straight_line/20260412_165322/summary.json)
  - mean abs track error: `3.372 m`
  - mean estimated XY pos error: `1.100 m`
  - mean estimated vel XYZ error: `0.643 m/s`
  - mean estimated yaw error: `1.741 deg`

Interpretation:

- estimated path works
- live IMU path runs without NaN/sign breakage
- but straight-line quality with live IMU is still materially worse than synthetic

### Loiter smoke results

- truth:
  - [summary.json](/home/zn/IsaacLab/logs/flapping_px4/loiter/20260412_165724/summary.json)
  - mean abs track error: `0.244 m`
  - mean abs height error: `2.608 m`
- estimated + synthetic:
  - [summary.json](/home/zn/IsaacLab/logs/flapping_px4/loiter/20260412_165759/summary.json)
  - mean abs track error: `0.839 m`
  - mean abs height error: `2.790 m`
  - mean estimated XY pos error: `1.056 m`
  - mean estimated vel XYZ error: `1.042 m/s`
  - mean estimated yaw error: `1.540 deg`
- estimated + isaacsim_live:
  - [summary.json](/home/zn/IsaacLab/logs/flapping_px4/loiter/20260412_165420/summary.json)
  - mean abs track error: `0.618 m`
  - mean abs height error: `1.491 m`
  - mean estimated XY pos error: `1.175 m`
  - mean estimated vel XYZ error: `0.926 m/s`
  - mean estimated yaw error: `0.966 deg`

Interpretation:

- estimated controller is viable
- the remaining issue looks more like tuning / phase behavior than gross sign or frame mismatch

## Tuning Objective

Do **not** touch RL reward, actor, student-teacher training mechanics, or DeLaurier backend.

Focus only on:

- estimated-state teacher controller behavior
- estimator-facing controller bandwidth / damping choices
- estimated-state TECS sensitivity to noisy / lagged altitude, velocity, yaw, wind
- inner-loop smoothing or rate limiting only if clearly needed for estimated path

## Recommended Files To Inspect First

- [straight_line_controller.py](/home/zn/IsaacLab/source/flapping_bot/flapping_bot/px4_like/straight_line_controller.py)
- [loiter_controller.py](/home/zn/IsaacLab/source/flapping_bot/flapping_bot/px4_like/loiter_controller.py)
- [path_tracking_controller.py](/home/zn/IsaacLab/source/flapping_bot/flapping_bot/px4_like/path_tracking_controller.py)
- [tecs.py](/home/zn/IsaacLab/source/flapping_bot/flapping_bot/px4_like/tecs.py)
- [state_estimation.py](/home/zn/IsaacLab/source/flapping_bot/flapping_bot/px4_like/state_estimation.py)
- [fly_straight_line.py](/home/zn/IsaacLab/scripts/flapping_px4/fly_straight_line.py)
- [fly_loiter.py](/home/zn/IsaacLab/scripts/flapping_px4/fly_loiter.py)

## Concrete Constraints For The New Agent

- `truth teacher` should remain the forensic baseline and should not be retuned as the main path.
- Do not fork a second controller implementation unless absolutely necessary.
- Prefer one controller code path with separate `truth` vs `estimated` tuning surfaces.
- First target is `estimated + synthetic`.
- Second target is to see whether those changes transfer to `estimated + isaacsim_live`.

## Suggested Verification Commands

### Syntax and tests

```bash
python -m py_compile source/flapping_bot/flapping_bot/px4_like/straight_line_controller.py
python -m py_compile source/flapping_bot/flapping_bot/px4_like/loiter_controller.py
python -m py_compile source/flapping_bot/flapping_bot/px4_like/path_tracking_controller.py
python -m py_compile source/flapping_bot/flapping_bot/px4_like/tecs.py
./isaaclab.sh -p -m pytest tests/test_train_and_watch.py tests/test_eval_suites.py tests/test_fly_controller_state_source_contract.py tests/test_imu_provider.py tests/test_sensor_state_estimator.py tests/test_rl_teacher_guidance.py tests/test_path_tracking_env_contract.py -q
```

### Non-RL tuning comparisons

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_straight_line.py --state_source truth --teacher_state_source truth --imu_source synthetic --steps 1200 --headless
./isaaclab.sh -p scripts/flapping_px4/fly_straight_line.py --state_source estimated --teacher_state_source estimated --imu_source synthetic --steps 1200 --headless
./isaaclab.sh -p scripts/flapping_px4/fly_straight_line.py --state_source estimated --teacher_state_source estimated --imu_source isaacsim --steps 1200 --headless

./isaaclab.sh -p scripts/flapping_px4/fly_loiter.py --state_source truth --teacher_state_source truth --imu_source synthetic --steps 1200 --min_loiter_turns 0.5 --metrics_warmup_s 2.0 --headless
./isaaclab.sh -p scripts/flapping_px4/fly_loiter.py --state_source estimated --teacher_state_source estimated --imu_source synthetic --steps 1200 --min_loiter_turns 0.5 --metrics_warmup_s 2.0 --headless
./isaaclab.sh -p scripts/flapping_px4/fly_loiter.py --state_source estimated --teacher_state_source estimated --imu_source isaacsim --steps 1200 --min_loiter_turns 0.5 --metrics_warmup_s 2.0 --headless
```

## Deliverable Expected From The New Agent

The new agent should produce:

1. a minimal estimated-teacher tuning patch
2. fresh before/after numeric comparison
3. explicit statement that `truth teacher` was kept fixed
4. recommendation on whether `estimated + synthetic` is ready
5. recommendation on whether `estimated + isaacsim_live` is close enough or still needs extra estimator work
