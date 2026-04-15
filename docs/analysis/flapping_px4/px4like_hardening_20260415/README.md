# PX4-Like Hardening 2026-04-15

This folder contains the current high-value evidence for the estimated-teacher controller after the PX4-inspired hardening pass.

## Scope

This pass focused on:

- estimated-teacher controller robustness
- shared controller logic with estimated-specific tuning surfaces
- live Isaac IMU transfer checks

This pass did not focus on:

- truth-teacher retuning as the main target
- RL reward, actor, observation, or student-teacher training changes
- DeLaurier backend replacement

## Code Context

Relevant code changes landed around:

- controller tuning profiles
- bank-aware TECS use of actual roll
- shared guidance minimum airspeed handling
- lateral-guidance uncertainty scaling
- runtime live Isaac IMU diagnostics and estimated-state plumbing
- complex-suite runner and associated tests

## Baseline Used For These Results

- Airframe backend: `DeLaurier`
- Mass baseline: `total_mass_kg_override=0.95`
- Main controller baseline:
  - `teacher_state_source=estimated`
  - `policy_state_source=estimated`
  - `imu_source=synthetic`

## Included Artifacts

- [px4like_height_error_curves.png](./px4like_height_error_curves.png)
  - No-wind estimated synthetic height-error comparison across key path-mission seeds.
- [px4like_trajectory_2d.png](./px4like_trajectory_2d.png)
  - No-wind estimated synthetic 2D trajectory comparison.
- [px4like_wind2_height_error_curves.png](./px4like_wind2_height_error_curves.png)
  - `2 m/s` steady-crosswind height-error comparison.
- [px4like_wind2_trajectory_2d.png](./px4like_wind2_trajectory_2d.png)
  - `2 m/s` steady-crosswind 2D trajectories.
- [px4like_wind2_summary.csv](./px4like_wind2_summary.csv)
  - No-wind versus wind summary for the synthetic path-mission runs.
- [live_vs_synth_wind2_path_summary.csv](./live_vs_synth_wind2_path_summary.csv)
  - `synthetic` versus `isaacsim` comparison under the same `2 m/s` crosswind battery.

## Main Findings

### 1. No-wind estimated synthetic is good enough to use as the tuning baseline

The no-wind path-mission runs showed that the PX4-inspired hardening materially improved lateral recapture on difficult seeds without introducing a new altitude-collapse mode.

### 2. `2 m/s` steady crosswind is survivable for the estimated baseline

The wind checks used:

- `wind_x_mps=0.0`
- `wind_y_mps=2.0`
- no OU gusts

This keeps total wind magnitude at exactly `2.0 m/s`.

Under this wind:

- altitude error stayed in roughly the same range as the no-wind baseline
- degradation was mainly lateral
- the remaining failures looked like difficult lateral recapture or long-loiter geometry, not gross estimator or TECS collapse

### 3. Live Isaac IMU is not showing a systematic wind-robustness penalty

The current `synthetic` vs `isaacsim` path battery under `2 m/s` crosswind does not show `isaacsim` being systematically worse.

From [live_vs_synth_wind2_path_summary.csv](./live_vs_synth_wind2_path_summary.csv):

- mean final progress ratio:
  - `isaacsim`: about `0.839`
  - `synthetic`: about `0.826`
- mean post-warmup lateral error:
  - `isaacsim`: about `1.154 m`
  - `synthetic`: about `1.486 m`
- mean post-warmup height error:
  - `isaacsim`: about `0.616 m`
  - `synthetic`: about `0.619 m`

Interpretation:

- `isaacsim` is not the obvious bottleneck under the current wind gate.
- The shared limitation is more like difficult lateral recapture in demanding path segments.

## Recommended Reproduction Commands

### No-wind baseline battery

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py \
  --mission_mode random --mission_seed 25 \
  --teacher_state_source estimated --policy_state_source estimated \
  --imu_source synthetic --headless --num_envs 1 --steps 3600 \
  --total_mass_kg_override 0.95
```

Repeat the same command for `--mission_seed 30` and `--mission_seed 33`.

### `2 m/s` steady-crosswind battery

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py \
  --mission_mode random --mission_seed 25 \
  --teacher_state_source estimated --policy_state_source estimated \
  --imu_source synthetic --headless --num_envs 1 --steps 3600 \
  --total_mass_kg_override 0.95 \
  --wind_x_mps 0.0 --wind_y_mps 2.0
```

Repeat for:

- `--mission_seed 30`
- `--mission_seed 33`
- `--imu_source isaacsim`

### Formal suite runner

```bash
./isaaclab.sh -p scripts/flapping_px4/run_estimated_path_tracking_complex_suite.py \
  --scenario live_imu_spotcheck \
  --teacher_state_source estimated \
  --policy_state_source estimated \
  --imu_source isaacsim \
  --total_mass_kg_override 0.95 \
  --random_seeds 25,30,33
```

## Decision

- Use `estimated + synthetic + 0.95kg` as the main controller baseline.
- Use `estimated + isaacsim` as the realism transfer gate.
- For the current `2 m/s` wind requirement, both paths are basically acceptable.
- The remaining work, if needed, should focus on lateral recapture and high-curvature path segments rather than reopening IMU plumbing first.
