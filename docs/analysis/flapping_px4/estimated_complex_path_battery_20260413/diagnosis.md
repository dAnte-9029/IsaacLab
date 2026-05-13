# Estimated Complex Path Battery Diagnosis

## Scope

- Runtime contract: `teacher_state_source=estimated`, `policy_state_source=estimated`, `imu_source=synthetic`
- Plant/runtime: `total_mass_kg_override=0.95`, `controller_tuning_profile=estimated_teacher`
- Baseline evidence root:
  - `logs/flapping_px4/estimated_path_tracking_complex_suite/baseline_nowind/20260413_204736`

## Baseline Ranking

- Worst mean lateral error: `case03_seed030` at `0.303 m`
- Worst p95 lateral error: `case04_seed033` at `0.717 m`
- Worst mean height error: `case03_seed030` at `0.614 m`
- Selected worst-two seeds for follow-up realism gates: `seed030` and `seed033`

## Shared Pattern

- Both worst cases finish successfully with `final_progress_ratio ~= 0.995`, so this is not a gross completion failure.
- Both cases accumulate their largest deviations in curved or curve-adjacent portions of the mission rather than in long steady straight segments.
- Both retain moderate end-state speed (`~9.36-10.34 m/s`), which points more toward path-following / phase behavior than toward a simple energy-collapse explanation.

## Case Notes

### `case03_seed030`

- Mission segments: `turn -> loiter -> straight(climb) -> loiter`
- Evidence:
  - `mean_abs_lateral_error_m = 0.303`
  - `mean_abs_height_error_m = 0.614`
  - peak lateral error `1.030 m` at `t=10.65 s`, `progress_ratio=0.081`
  - peak height error `2.881 m` at `t=31.39 s`, `progress_ratio=0.264`
  - final lateral error still elevated at `0.640 m`
- Interpretation:
  - This is the broadest combined degradation case.
  - Lateral error spikes early in the opening curved segment, then altitude peaks later during the loiter / climb portion.
  - The error is not just one late outlier; it is distributed enough to dominate both mean lateral and mean height metrics.

### `case04_seed033`

- Mission segments: `straight(climb) -> straight(climb) -> turn -> turn`
- Evidence:
  - `mean_abs_lateral_error_m = 0.183`
  - `p95_abs_lateral_error_m = 0.717`
  - `mean_abs_height_error_m = 0.532`
  - peak lateral error `0.845 m` at `t=33.02 s`, `progress_ratio=0.947`
  - peak height error `2.287 m` at `t=28.95 s`, `progress_ratio=0.826`
  - final lateral error `0.364 m`
- Interpretation:
  - This case is mainly a late-mission tail-risk case rather than a broad mean-error case.
  - Height error rises before the terminal turn sequence, then the lateral tail grows near the last turn.
  - That combination makes it the worst `p95` lateral case even though its mean lateral error is not the highest.

## Interim Decision

- The current estimated+synthetic baseline is viable enough to continue realism-gate testing.
- The next realism checks should focus on `seed030` and `seed033` for `isaacsim` live-IMU spot checks.
- Wind stress should still be run on the full 5-case formal battery to see whether the current controller degrades gracefully or turns these curve-heavy cases into outright failures.

## Realism Gates

### `steady_crosswind`

- Evidence root:
  - `logs/flapping_px4/estimated_path_tracking_complex_suite/steady_crosswind/20260413_212156`
- Outcome:
  - runner summary: `cases_requested=5`, `cases_attempted=1`, `cases=0`, `completion_rate=0.0`
  - `cases.csv` shows `multi_segment` returned `124` with `timed_out=True`
  - the Kit log for `portable/multi_segment` closes at `~900 s` wall-clock with no mission `summary.json` ever created
- Interpretation:
  - This is not graceful metric degradation.
  - Under constant `+2.0 m/s` lateral wind, the current estimated controller path battery does not finish the first formal case within the suite timeout budget.
  - Because no case-level mission summary exists, this gate currently supports a failure-mode conclusion, not a fine-grained tracking-error comparison.

### `ou_gust`

- Invalid first attempt:
  - root: `logs/flapping_px4/estimated_path_tracking_complex_suite/ou_gust/20260413_212156`
  - diagnosis: invalid
  - evidence:
    - logged `wind_x_mps` / `wind_y_mps` stayed identically zero
    - `trajectory_env0.csv` hash exactly matched the no-wind baseline for `multi_segment`
  - root cause:
    - the old runtime contract reused `wind_x_mps` / `wind_y_mps` as both mean wind and clip range
    - for zero-mean OU gusts with clipping enabled, that silently clipped the process to zero
- Corrected attempt:
  - root: `logs/flapping_px4/estimated_path_tracking_complex_suite/ou_gust/20260413_213511`
  - runtime fix:
    - explicit wind clip ranges were added to `fly_path_mission.py`
    - the suite runner now sets OU clip ranges to `±3 sigma`
  - validation:
    - corrected `multi_segment` logs `wind_x_range_mps=[-3.0, 3.0]`, `wind_y_range_mps=[-4.5, 4.5]`
    - corrected `multi_segment` wind stats are non-zero, with `mean_abs_x ~= 0.791 m/s`, `mean_abs_y ~= 1.082 m/s`
    - corrected trajectory hash differs from the no-wind baseline
  - aggregate:
    - `cases=5`, `completed_cases=5`, `completion_rate=1.0`
    - `mean_case_lateral_error_m = 0.26497` vs baseline `0.21687` (`+22.2%`)
    - `mean_case_height_error_m = 0.50781` vs baseline `0.57447` (`-11.6%`)
    - `mean_case_p95_lateral_error_m = 0.72907` vs baseline `0.63878` (`+14.1%`)
    - worst mean / p95 lateral and height case all move to `case04_seed033`
- Interpretation:
  - After the runtime fix, OU gusts produce a real but still graceful degradation.
  - The main penalty is lateral tracking, not gross loss of completion or altitude collapse.

### `live_imu_spotcheck`

- Evidence root:
  - `logs/flapping_px4/estimated_path_tracking_complex_suite/live_imu_spotcheck/20260413_213757`
- Outcome:
  - runner summary: `cases_requested=2`, `cases_attempted=1`, `cases=0`, `completion_rate=0.0`
  - `cases.csv` shows `case01_seed030` returned `124` with `timed_out=True`
  - the Kit log for `portable/case01_seed030` closes at `~900 s` wall-clock with no mission `summary.json` ever created
- Interpretation:
  - Live Isaac IMU does not preserve the current synthetic-estimated behavior on the worst seed.
  - The failure appears before we even reach the second diagnosed seed, so this is a stronger realism-gate failure than the corrected OU gust case.
  - At the current stage, the estimated controller is viable as a synthetic realism baseline, but not yet as a live-IMU path-mission baseline.

## Final Decision

- `estimated + synthetic + 0.95kg` remains the main baseline worth carrying forward for controller work and later RL realism baselines.
- Constant crosswind and live IMU both expose unresolved timeout-level failure modes, so they should not be treated as passed realism gates yet.
- Corrected OU gust evidence justifies keeping the current controller direction: it degrades in lateral tracking, but it still completes all 5 formal cases.
