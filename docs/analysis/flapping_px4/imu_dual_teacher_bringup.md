# Isaac Sim IMU Dual-Teacher Bring-up

## Scope

This bring-up records the repo-local estimated-state path for the non-RL PX4-like controller scripts.

- `teacher_state_source`: explicit `truth` / `estimated`
- `policy_state_source`: explicit runtime contract
- `imu_source`: explicit `synthetic` / `isaacsim`

The goal is to make `teacher_estimated` mean estimated-state control without silent truth-wind leakage, and to make `imu_source=isaacsim` use a real Isaac Lab IMU sensor instead of a truth-derived placeholder.

## Key Implementation Points

### Runtime contract

- Added repo-local contract helper:
  - [state_source_contract.py](/home/zn/IsaacLab/source/flapping_bot/flapping_bot/direct/flapping_bot/state_source_contract.py)
- Wired explicit fields through direct envs and controller-only scripts:
  - [straight_flight_env.py](/home/zn/IsaacLab/source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py)
  - [path_tracking_env.py](/home/zn/IsaacLab/source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py)
  - [fly_straight_line.py](/home/zn/IsaacLab/scripts/flapping_px4/fly_straight_line.py)
  - [fly_loiter.py](/home/zn/IsaacLab/scripts/flapping_px4/fly_loiter.py)

### IMU provider path

- Added repo-local IMU abstraction:
  - [imu_provider.py](/home/zn/IsaacLab/source/flapping_bot/flapping_bot/px4_like/imu_provider.py)
- Added Isaac Lab IMU adapter:
  - [isaacsim_imu_adapter.py](/home/zn/IsaacLab/source/flapping_bot/flapping_bot/px4_like/isaacsim_imu_adapter.py)
- Estimator now accepts external IMU measurements:
  - [state_estimation.py](/home/zn/IsaacLab/source/flapping_bot/flapping_bot/px4_like/state_estimation.py)

### Late-created sensor root cause and fix

The first live-runtime failure was not controller-side. It was Isaac Lab sensor lifecycle:

- `Imu` relies on a play-time lazy initialization callback in `SensorBase`.
- These controller scripts create the IMU after the environment has already started.
- That means a late-created IMU can miss the initialization callback and crash on `reset()` with missing internal buffers such as `_timestamp`.

Minimal fix:

- In both controller-only scripts, if a freshly created IMU sensor reports `is_initialized == False`, call `_initialize_impl()` once and set `_is_initialized = True` before `reset()/update()`.
- This matches existing Isaac Lab late-bound sensor usage patterns in task-space actions.

Relevant script sites:

- [fly_straight_line.py](/home/zn/IsaacLab/scripts/flapping_px4/fly_straight_line.py)
- [fly_loiter.py](/home/zn/IsaacLab/scripts/flapping_px4/fly_loiter.py)

## Verification

### Unit and contract tests

Command:

```bash
./isaaclab.sh -p -m pytest tests/test_fly_controller_state_source_contract.py tests/test_imu_provider.py tests/test_sensor_state_estimator.py tests/test_rl_teacher_guidance.py tests/test_path_tracking_env_contract.py -q
```

Observed result:

- `74 passed in 1.15s`

### Runtime smoke: straight

Command:

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_straight_line.py --state_source estimated --teacher_state_source estimated --imu_source isaacsim --steps 1200 --headless
```

Evidence:

- Truth baseline:
  - [summary.json](/home/zn/IsaacLab/logs/flapping_px4/straight_line/20260412_165617/summary.json)
- Estimated + synthetic:
  - [summary.json](/home/zn/IsaacLab/logs/flapping_px4/straight_line/20260412_165649/summary.json)
- Summary: [summary.json](/home/zn/IsaacLab/logs/flapping_px4/straight_line/20260412_165322/summary.json)
- Trajectory: [trajectory_env0.csv](/home/zn/IsaacLab/logs/flapping_px4/straight_line/20260412_165322/trajectory_env0.csv)
- `imu_measurement_mode == "isaacsim_live"`

Observed comparison:

- truth: mean abs track error `0.058 m`
- estimated + synthetic: mean abs track error `0.253 m`
- estimated + isaacsim: mean abs track error `3.372 m`

Interpretation:

- The live IMU path is frame-consistent enough to run without NaNs or sign flips.
- But on straight-flight it is currently materially worse than the synthetic estimated path.
- This is not a bring-up failure, but it is not yet ready to replace the synthetic estimated path as the default realism baseline everywhere.

### Runtime smoke: loiter

Command:

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_loiter.py --state_source estimated --teacher_state_source estimated --imu_source isaacsim --steps 1200 --min_loiter_turns 0.5 --metrics_warmup_s 2.0 --headless
```

Evidence:

- Truth baseline:
  - [summary.json](/home/zn/IsaacLab/logs/flapping_px4/loiter/20260412_165724/summary.json)
- Estimated + synthetic:
  - [summary.json](/home/zn/IsaacLab/logs/flapping_px4/loiter/20260412_165759/summary.json)
- Summary: [summary.json](/home/zn/IsaacLab/logs/flapping_px4/loiter/20260412_165420/summary.json)
- Trajectory: [trajectory_env0.csv](/home/zn/IsaacLab/logs/flapping_px4/loiter/20260412_165420/trajectory_env0.csv)
- `imu_measurement_mode == "isaacsim_live"`

Observed comparison:

- truth: mean abs track error `0.244 m`, mean abs height error `2.608 m`
- estimated + synthetic: mean abs track error `0.839 m`, mean abs height error `2.790 m`
- estimated + isaacsim: mean abs track error `0.618 m`, mean abs height error `1.491 m`

Interpretation:

- On loiter, the live IMU path is competitive with and in this smoke better than the synthetic estimated path.
- The straight/loiter split suggests the remaining issue is estimator tuning / phase behavior, not an immediate sign convention break.

## RL Ramp Decision

The RL ramp should not silently reuse the truth teacher path as its realism baseline.

Implemented policy:

- training/watcher defaults now inject:
  - `teacher_state_source=estimated`
  - `policy_state_source=estimated`
  - `imu_source=synthetic`
- path-tracking evaluation suites now separate:
  - truth-forensics: `path_tracking_truth_nowind_v1`, `path_tracking_truth_primitives_nowind_v1`
  - estimated-realism: `path_tracking_estimated_nowind_v1`, `path_tracking_estimated_primitives_nowind_v1`

This keeps the live Isaac IMU path available for controller bring-up while avoiding premature promotion of the current straight-flight-degraded live IMU path into the default RL realism route.

## Practical Meaning

After this bring-up:

- `teacher_state_source="estimated"` is explicit and separate from truth-controller debugging.
- `imu_source="isaacsim"` in the non-RL scripts now binds a live Isaac Lab IMU sensor.
- The summary files explicitly state whether the estimator used:
  - `isaacsim_live`
  - `synthetic_truth_derived`
  - `disabled`

That removes ambiguity when comparing truth-teacher vs estimated-teacher controller behavior.
