# Straight Height Recovery Retune Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the DeLaurier non-RL controller-only `level_straight` rollout recover quickly back to `height_error ~= 0` after startup altitude loss, while keeping `max_flap_hz <= 5.0`.

**Architecture:** Keep the current self-consistent plant fixed (`base_body_com_override_x_m=-0.10`, tail effectiveness unchanged) and retune only PX4-like longitudinal controller behavior. Add temporary CLI/config pass-through for TECS height-capture parameters, run a compact straight-only search against fresh long-straight rollouts, then freeze the smallest winning controller default and verify canonical `straight/turn/loiter`.

**Tech Stack:** IsaacLab headless rollouts, `scripts/flapping_px4/fly_path_mission.py`, DeLaurier path-tracking task, Python CSV analysis, targeted `py_compile` and `pytest`.

---

## Baseline Context

Current saved baseline commit before this plan:

```bash
git rev-parse --short HEAD
# Expected: 85a10590
```

Current constraints:

- Do not raise `max_flap_hz` above `5.0`.
- Do not tune RL, reward, observation, actor policy, or teacher-student schedule.
- Do not change DeLaurier, COM, tail effectiveness, elevon limits, or wrench aggregation in this pass.
- Prior data showed the elevon is not saturated during long straight recovery; the first target is TECS outer-loop capture and pitch-side demand.

Primary success metric:

- After the startup dip, time and distance from minimum `height_error_m` to first sustained return to `abs(height_error_m) <= 0.05 m`.

Secondary metrics:

- `min_height_error_first3s`
- first crossing back to `height_error_m >= 0`
- `mean_abs_height_error` after recovery
- max positive overshoot after recovery
- `freq_hz` and `exec_freq_hz` saturation fraction near `5.0 Hz`
- `exec_action_elevon_pitch` saturation fraction near `abs(action) >= 0.98`
- min airspeed / speed during recovery
- `pitch_sp_deg` and `tecs_pitch_sp_deg` timing relative to height error

Acceptance target for straight:

- Recovery time/distance from the minimum error to the `0.05 m` band improves materially versus fresh baseline, target `>=30%`.
- Late steady height error stays near zero, target `mean_abs_height_error_after_recovery <= 0.08 m`.
- Overshoot is bounded, target max positive `height_error_m <= 0.20 m`.
- No obvious pitch oscillation, no early termination, no frequency limit violation.

### Task 1: Add straight recovery analysis utility

**Files:**

- Create: `scripts/flapping_px4/analyze_straight_height_recovery.py`
- Test manually with: existing `/tmp/.../trajectory_env0.csv` from long-straight runs

**Step 1: Implement CSV metric extraction**

Create a small script that reads `trajectory_env0.csv` and writes one JSON metrics file. It should accept:

```bash
python scripts/flapping_px4/analyze_straight_height_recovery.py \
  --traj_csv /tmp/run/level_straight/<timestamp>/trajectory_env0.csv \
  --summary_json /tmp/run/level_straight/<timestamp>/summary.json \
  --output_json /tmp/run/level_straight/<timestamp>/straight_height_recovery_metrics.json
```

Required metrics:

```python
{
    "min_height_error_first3s_m": ...,
    "min_height_error_m": ...,
    "time_at_min_height_error_s": ...,
    "distance_at_min_height_error_m": ...,
    "time_to_abs_0p05_after_min_s": ...,
    "distance_to_abs_0p05_after_min_m": ...,
    "time_to_cross_zero_after_min_s": ...,
    "distance_to_cross_zero_after_min_m": ...,
    "mean_abs_height_error_after_recovery_m": ...,
    "max_positive_height_error_after_recovery_m": ...,
    "freq_sat_fraction": ...,
    "elevon_pitch_sat_fraction": ...,
    "min_speed_mps": ...,
    "mean_tecs_pitch_sp_deg_first5s": ...,
    "mean_pitch_sp_deg_first5s": ...,
}
```

Use `abs(height_error_m) <= 0.05` as the band. To avoid one-sample noise, define "sustained" as staying inside `abs(height_error_m) <= 0.10` for at least `1.0 s` after first entering the `0.05 m` band.

**Step 2: Verify on an existing CSV**

Run:

```bash
python scripts/flapping_px4/analyze_straight_height_recovery.py \
  --traj_csv /tmp/final_long_straight_reset_speed8_20260410/level_straight/20260410_190026/trajectory_env0.csv \
  --summary_json /tmp/final_long_straight_reset_speed8_20260410/level_straight/20260410_190026/summary.json \
  --output_json /tmp/final_long_straight_reset_speed8_20260410/level_straight/20260410_190026/straight_height_recovery_metrics.json
```

Expected:

- Exit code `0`.
- JSON includes finite `min_height_error_m`, finite recovery time or `null` if the run never recovers.

### Task 2: Expose TECS height-capture parameters to the non-RL path mission CLI

**Files:**

- Modify: `scripts/flapping_px4/fly_path_mission.py`
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py`
- Test: `tests/test_fly_path_mission.py`

**Step 1: Add path env config fields**

Add pass-through fields with defaults matching `PX4LikeStraightLineControllerCfg`, not the proposed new tuned values yet:

```python
teacher_tecs_altitude_hold_error_band_m: float = 0.25
teacher_tecs_altitude_capture_error_m: float = 0.8
teacher_tecs_altitude_capture_time_const_s: float = 1.0
teacher_tecs_pitch_speed_weight: float = 0.8
teacher_tecs_pitch_speed_weight_capture: float = 0.35
teacher_tecs_capture_extra_climb_rate_mps: float = 0.7
teacher_tecs_capture_extra_sink_rate_mps: float = 0.2
teacher_tecs_pitch_damping_gain: float = 0.08
```

Pass those into `PX4LikeStraightLineControllerCfg(...)` where the teacher controller is constructed.

**Step 2: Add CLI args**

Add optional args with `default=None`:

```python
parser.add_argument("--teacher_tecs_altitude_hold_error_band_m", type=float, default=None)
parser.add_argument("--teacher_tecs_altitude_capture_error_m", type=float, default=None)
parser.add_argument("--teacher_tecs_altitude_capture_time_const_s", type=float, default=None)
parser.add_argument("--teacher_tecs_pitch_speed_weight", type=float, default=None)
parser.add_argument("--teacher_tecs_pitch_speed_weight_capture", type=float, default=None)
parser.add_argument("--teacher_tecs_capture_extra_climb_rate_mps", type=float, default=None)
parser.add_argument("--teacher_tecs_capture_extra_sink_rate_mps", type=float, default=None)
parser.add_argument("--teacher_tecs_pitch_damping_gain", type=float, default=None)
```

Apply each override in `_configure_env`.

**Step 3: Add parser/config tests**

Extend `test_path_mission_parser_accepts_teacher_tecs_overrides()` and `test_configure_env_applies_teacher_tecs_overrides()` with the new fields.

Example assertions:

```python
assert args.teacher_tecs_altitude_hold_error_band_m == pytest.approx(0.1)
assert configured_env_cfg.teacher_tecs_altitude_capture_time_const_s == pytest.approx(0.65)
assert configured_env_cfg.teacher_tecs_pitch_speed_weight == pytest.approx(0.55)
```

**Step 4: Verify**

Run:

```bash
python -m py_compile scripts/flapping_px4/fly_path_mission.py \
  source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py
python -m pytest tests/test_fly_path_mission.py tests/test_path_tracking_env_contract.py -q
```

Expected:

- `py_compile` exit code `0`.
- Pytest exit code `0`.

### Task 3: Run fresh straight baseline

**Files:**

- Reuse: `scripts/flapping_px4/fly_path_mission.py`
- Reuse: `scripts/flapping_px4/analyze_straight_height_recovery.py`
- Create output under: `/tmp/straight_height_recovery_baseline_20260411`

**Step 1: Run the controller-only DeLaurier long straight**

Run without path warmup so startup recovery is visible:

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py \
  --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 \
  --phase level_straight \
  --straight_length_m 180 \
  --steps 5000 \
  --no-path_warmup_enabled \
  --headless \
  --output_dir /tmp/straight_height_recovery_baseline_20260411
```

Expected:

- Rollout completes or reaches near-complete straight progress.
- `trajectory_env0.csv` exists.
- Summary shows controller-only teacher path, not RL actor path.

**Step 2: Analyze baseline**

Run the new analyzer on the generated `trajectory_env0.csv`.

Expected:

- Record baseline recovery time/distance and late height error.
- Save metrics JSON next to the rollout.

### Task 4: Run a compact TECS recovery candidate sweep

**Files:**

- Option A, preferred Create: `scripts/flapping_px4/run_straight_height_recovery_sweep.py`
- Option B, acceptable: run explicit shell commands and manually aggregate JSON outputs
- Read: `trajectory_env0.csv` and metrics JSON from each candidate

**Step 1: Use this small candidate set**

Keep plant and frequency limits fixed. Sweep only the TECS height-capture/pitch-demand parameters:

| Candidate | hold band | capture tc | extra climb | pitch speed weight | capture pitch speed weight | pitch damping |
|---|---:|---:|---:|---:|---:|---:|
| `baseline` | `0.25` | `1.00` | `0.70` | `0.80` | `0.35` | `0.08` |
| `mild_capture` | `0.15` | `0.80` | `1.00` | `0.65` | `0.35` | `0.08` |
| `balanced_fast` | `0.10` | `0.65` | `1.00` | `0.55` | `0.30` | `0.10` |
| `aggressive_capture` | `0.10` | `0.50` | `1.20` | `0.50` | `0.25` | `0.12` |
| `small_band` | `0.05` | `0.65` | `1.00` | `0.50` | `0.25` | `0.10` |
| `altitude_priority` | `0.10` | `0.65` | `1.20` | `0.45` | `0.20` | `0.12` |

Command template:

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py \
  --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 \
  --phase level_straight \
  --straight_length_m 180 \
  --steps 5000 \
  --no-path_warmup_enabled \
  --teacher_tecs_altitude_hold_error_band_m <band> \
  --teacher_tecs_altitude_capture_time_const_s <tc> \
  --teacher_tecs_capture_extra_climb_rate_mps <climb> \
  --teacher_tecs_pitch_speed_weight <psw> \
  --teacher_tecs_pitch_speed_weight_capture <psw_capture> \
  --teacher_tecs_pitch_damping_gain <damping> \
  --headless \
  --output_dir /tmp/straight_height_recovery_sweep_20260411/<candidate>
```

**Step 2: Rank candidates**

Rank by:

1. Valid rollout without early failure.
2. Shorter `distance_to_abs_0p05_after_min_m`.
3. Shorter `time_to_abs_0p05_after_min_s`.
4. Bounded overshoot and no sustained oscillation.
5. No large speed collapse and no excessive `freq_hz` saturation.

Reject candidates if:

- they never recover to the `0.05 m` band,
- they recover only by saturating frequency almost continuously,
- they cause pitch oscillation or late mean height error worse than baseline,
- they improve straight but create obvious controller noise in `pitch_sp_deg`.

### Task 5: Freeze the smallest winning controller default

**Files:**

- Modify: `source/flapping_bot/flapping_bot/px4_like/straight_line_controller.py`
- Possibly modify: `source/flapping_bot/flapping_bot/px4_like/tecs.py` only if diagnostics show a logic bug, not for tuning-only changes
- Test: `tests/test_path_tracking_env_contract.py` if defaults are asserted there

**Step 1: Apply only the minimal winning default**

Expected first candidate to try if the sweep is not yet conclusive:

```python
tecs_altitude_hold_error_band_m = 0.10
tecs_altitude_capture_time_const_s = 0.65
tecs_capture_extra_climb_rate_mps = 1.0
tecs_pitch_speed_weight = 0.55
tecs_pitch_speed_weight_capture = 0.30
tecs_pitch_damping_gain = 0.10
```

Do not change these in the same commit unless the sweep shows they are required:

```python
tecs_integrator_gain_pitch
tecs_pitch_sp_filter_tau_s
tecs_pitch_sp_filter_tau_capture_s
tecs_pitch_sp_rate_limit_deg_s
tecs_pitch_sp_rate_limit_capture_deg_s
load_factor_pitch_compensation_gain
min_flap_hz
max_flap_hz
tail_fixed_horizontal_effectiveness
tail_elevon_effectiveness
base_body_com_override_x_m
```

**Step 2: Keep the tuning explanation in code comments minimal**

Do not add long comments. If a comment is needed, state only that the capture defaults are tuned for fast straight-height recovery under the `5 Hz` flapping cap.

### Task 6: Verify straight recovery with fresh rollouts

**Files:**

- Reuse: `scripts/flapping_px4/fly_path_mission.py`
- Reuse: `scripts/flapping_px4/analyze_straight_height_recovery.py`
- Create output under: `/tmp/straight_height_recovery_final_20260411`

**Step 1: Run long straight without warmup**

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py \
  --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 \
  --phase level_straight \
  --straight_length_m 180 \
  --steps 5000 \
  --no-path_warmup_enabled \
  --headless \
  --output_dir /tmp/straight_height_recovery_final_20260411/no_warmup
```

Expected:

- `distance_to_abs_0p05_after_min_m` improves versus fresh baseline.
- Height error returns near zero and stays bounded.
- `freq_hz <= 5.0`.

**Step 2: Run scored canonical straight with default warmup**

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py \
  --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 \
  --phase level_straight \
  --steps 2200 \
  --headless \
  --output_dir /tmp/straight_height_recovery_final_20260411/canonical_straight
```

Expected:

- No obvious regression in scored `level_straight`.
- Mean/p95 height error should not worsen versus current baseline.

### Task 7: Smoke-check turn and loiter after straight passes

**Files:**

- Reuse: `scripts/flapping_px4/fly_path_mission.py`
- Create output under: `/tmp/straight_height_recovery_final_20260411`

**Step 1: Run canonical turn**

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py \
  --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 \
  --phase level_turn \
  --steps 2400 \
  --headless \
  --output_dir /tmp/straight_height_recovery_final_20260411/canonical_turn
```

**Step 2: Run canonical loiter**

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py \
  --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 \
  --phase level_loiter \
  --steps 2800 \
  --headless \
  --output_dir /tmp/straight_height_recovery_final_20260411/canonical_loiter
```

Expected:

- No obvious degradation in lateral tracking or attitude stability.
- Height response should not be worse than before this retune.
- If turn/loiter still drop height while straight is fixed, defer to a separate bank-specific pass on `load_factor_pitch_compensation_gain`.

### Task 8: Plot and document evidence

**Files:**

- Reuse: `scripts/flapping_px4/plot_path_mission.py`
- Create: `docs/analysis/flapping_px4/straight_height_recovery_20260411/`

**Step 1: Save comparison plots**

Generate plots for:

- baseline long straight
- winning candidate long straight
- final canonical straight/turn/loiter

At minimum, plots must show:

- `height_error_m`
- `pitch_sp_deg` and `tecs_pitch_sp_deg`
- `freq_hz` / `exec_freq_hz`
- `exec_action_elevon_pitch`
- `vertical_support_ratio` if present

**Step 2: Write results note**

Create:

```text
docs/analysis/flapping_px4/straight_height_recovery_20260411/notes.md
```

Include:

- fresh baseline metrics,
- winning parameter bundle,
- final straight metrics,
- canonical straight/turn/loiter summary,
- reason for not changing `max_flap_hz`, tail, COM, or RL.

### Task 9: Final verification and commit

**Files:**

- Modified code and tests from Tasks 1, 2, and 5
- Analysis note from Task 8

**Step 1: Run syntax checks**

```bash
python -m py_compile \
  scripts/flapping_px4/fly_path_mission.py \
  scripts/flapping_px4/analyze_straight_height_recovery.py \
  source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py \
  source/flapping_bot/flapping_bot/px4_like/straight_line_controller.py \
  source/flapping_bot/flapping_bot/px4_like/tecs.py
```

Expected:

- Exit code `0`.

**Step 2: Run targeted unit tests**

```bash
python -m pytest tests/test_fly_path_mission.py tests/test_path_tracking_env_contract.py -q
```

Expected:

- Exit code `0`.

**Step 3: Commit only relevant files**

Do not commit unrelated generated URDF hash/timestamp files or old polygon-tool artifacts.

```bash
git add \
  scripts/flapping_px4/fly_path_mission.py \
  scripts/flapping_px4/analyze_straight_height_recovery.py \
  source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py \
  source/flapping_bot/flapping_bot/px4_like/straight_line_controller.py \
  tests/test_fly_path_mission.py \
  tests/test_path_tracking_env_contract.py \
  docs/analysis/flapping_px4/straight_height_recovery_20260411/notes.md
git commit -m "tune: improve straight height recovery"
```

Expected:

- Commit succeeds.
- Final response reports exact commands run, metrics before/after, and any residual risk.
