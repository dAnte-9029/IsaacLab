# Straight Aggressive Height, Inner-Loop, And Lateral Retune Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the non-RL DeLaurier controller-only `level_straight` rollout recover altitude much faster, reduce flap-period control chatter without slowing altitude recovery, and remove the straight-line lateral steady error.

**Architecture:** Keep the plant fixed and controller-first: do not change DeLaurier, COM, tail geometry/effectiveness, RL reward/observation/policy, or `max_flap_hz > 5.0`. First freeze the single-teacher logging fix so every rollout has one stateful controller update per env step. Then tune three separated loops in order: TECS longitudinal recovery, inner-loop flap-period rejection, and lateral guidance/roll tracking.

**Tech Stack:** IsaacLab headless rollouts, `scripts/flapping_px4/fly_path_mission.py`, `scripts/flapping_px4/plot_path_mission.py`, CSV metrics scripts, Python `pytest`, `py_compile`, DeLaurier path-tracking task `Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0`.

---

## Current Baseline To Beat

Use the single-teacher logging run, not the older double-call retuned plot:

```text
/tmp/straight_height_recovery_single_teacher_20260411/no_warmup/level_straight/20260411_121651
```

Current metrics:

| Metric | Current |
|---|---:|
| min height error | `-0.4093 m` |
| time to sustained `abs(height_error)<=0.05 m` after min | `7.9667 s` |
| distance to sustained `abs(height_error)<=0.05 m` after min | `61.2131 m` |
| time to zero crossing after min | `8.4167 s` |
| distance to zero crossing after min | `64.7999 m` |
| `freq_sat_fraction` | `0.0000` |
| `elevon_pitch_sat_fraction` | `0.0000` |
| post-5s mean lateral error | about `0.89 m` |
| high-frequency control content | `action_elevon_pitch/rudder/roll` near `4.7 Hz` |

Significant improvement targets:

- Longitudinal: sustained `0.05 m` recovery after min in `<= 4.5 s` and `<= 35 m`, with zero crossing in `<= 5.5 s`.
- Second-dip suppression: cycle-averaged height error after `4.0 s` should not fall below `-0.15 m`.
- Saturation: `freq_sat_fraction <= 0.25`, `elevon_pitch_sat_fraction <= 0.05`, `max_flap_hz <= 5.0`.
- Control chatter: reduce 4-5 Hz elevon pitch action RMS or spectral magnitude by at least `30%` with no worse than `15%` degradation in longitudinal recovery time.
- Lateral: post-5s mean absolute lateral error `<= 0.25 m` and final absolute lateral error `<= 0.30 m` in no-wind long straight.

## Task 0: Freeze Single-Teacher Logging And Plot Fix

**Files:**

- Modify already in workspace: `scripts/flapping_px4/fly_path_mission.py`
- Modify already in workspace: `scripts/flapping_px4/plot_path_mission.py`
- Modify already in workspace: `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`
- Modify already in workspace: `tests/test_fly_path_mission.py`
- Modify already in workspace: `tests/test_path_tracking_env_contract.py`
- Create already in workspace: `tests/test_plot_path_mission.py`
- Create already in workspace: `docs/analysis/flapping_px4/single_teacher_logging_20260411/`

**Step 1: Run verification**

```bash
python -m py_compile scripts/flapping_px4/fly_path_mission.py scripts/flapping_px4/plot_path_mission.py source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py
python -m pytest tests/test_fly_path_mission.py tests/test_path_tracking_env_contract.py tests/test_plot_path_mission.py -q
git diff --check
```

Expected:

- `py_compile` exit code `0`.
- Pytest reports `70 passed`.
- `git diff --check` exit code `0`.

**Step 2: Stage only related files**

```bash
git add scripts/flapping_px4/fly_path_mission.py \
  scripts/flapping_px4/plot_path_mission.py \
  source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py \
  tests/test_fly_path_mission.py \
  tests/test_path_tracking_env_contract.py \
  tests/test_plot_path_mission.py \
  docs/analysis/flapping_px4/single_teacher_logging_20260411
```

Do not stage:

- `source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/.asset_hash`
- `source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/config.yaml`
- `.codex`
- old `docs/analysis/flapping_px4/*` untracked directories
- polygon tools/images unless separately requested

**Step 3: Commit**

```bash
git commit -m "fix: log path mission teacher once per step"
```

Expected:

- New focused commit containing only logging/plot/test artifacts.

## Task 1: Add Metrics For Height Second-Dip, Chatter, And Lateral Bias

**Files:**

- Modify: `scripts/flapping_px4/analyze_straight_height_recovery.py`
- Create: `tests/test_analyze_straight_height_recovery.py`

**Step 1: Write failing tests**

Add synthetic CSV rows that contain:

- One initial height drop and a second dip after `4.0 s`.
- A 4.8 Hz sinusoid in `action_elevon_pitch`.
- A constant post-5s lateral error.

Test expected fields:

```python
def test_straight_height_recovery_metrics_include_second_dip_chatter_and_lateral_bias():
    metrics = compute_metrics(...)
    assert metrics["min_cycle_mean_height_error_after_4s_m"] == pytest.approx(-0.22, abs=0.03)
    assert metrics["elevon_pitch_chatter_rms_4to8hz"] > 0.05
    assert metrics["post5_mean_lateral_error_m"] == pytest.approx(0.8, abs=0.05)
    assert metrics["post5_final_lateral_error_m"] == pytest.approx(0.8, abs=0.05)
```

**Step 2: Run tests to verify failure**

```bash
python -m pytest tests/test_analyze_straight_height_recovery.py -q
```

Expected:

- FAIL because the analyzer does not yet compute those fields.

**Step 3: Implement metrics**

In `compute_metrics(...)`, add:

- `cycle_mean_window_s`, default `0.25`.
- `min_cycle_mean_height_error_after_4s_m`.
- `time_at_min_cycle_mean_height_error_after_4s_s`.
- `post5_mean_lateral_error_m`.
- `post5_final_lateral_error_m`.
- `post5_mean_abs_lateral_error_m`.
- `elevon_pitch_chatter_rms_4to8hz`.
- `rudder_chatter_rms_4to8hz`.
- `elevon_roll_chatter_rms_4to8hz`.

Use simple deterministic analysis:

- Moving average over `round(cycle_mean_window_s / dt)` samples for cycle-mean height.
- FFT band RMS for 4-8 Hz after de-meaning the selected control signal.

**Step 4: Run tests**

```bash
python -m pytest tests/test_analyze_straight_height_recovery.py -q
```

Expected:

- PASS.

**Step 5: Run analyzer on current baseline**

```bash
python scripts/flapping_px4/analyze_straight_height_recovery.py \
  --run_dir /tmp/straight_height_recovery_single_teacher_20260411/no_warmup/level_straight/20260411_121651 \
  --output_json /tmp/straight_height_recovery_single_teacher_20260411/no_warmup/level_straight/20260411_121651/straight_height_recovery_metrics_v2.json
```

Expected:

- JSON includes the new fields and finite values.

**Step 6: Commit**

```bash
git add scripts/flapping_px4/analyze_straight_height_recovery.py tests/test_analyze_straight_height_recovery.py
git commit -m "test: add straight recovery diagnostic metrics"
```

## Task 2: Make TECS Height Recovery Aggressive Enough To Matter

**Files:**

- Modify: `source/flapping_bot/flapping_bot/px4_like/tecs.py`
- Modify: `source/flapping_bot/flapping_bot/px4_like/straight_line_controller.py`
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py`
- Modify: `scripts/flapping_px4/fly_path_mission.py`
- Modify: `scripts/flapping_px4/fly_straight_line.py`
- Modify: `scripts/flapping_px4/fly_loiter.py`
- Modify: `scripts/flapping_px4/run_straight_height_recovery_sweep.py`
- Test: `tests/test_fly_path_mission.py`
- Test: `tests/test_path_tracking_env_contract.py`

**Design:**

The current problem is that the capture blend exits early and the hold branch is too slow. Do not solve this by raising `max_flap_hz`. Add one TECS behavior: altitude capture hysteresis / persistence. Once the aircraft has dropped meaningfully, keep capture-mode pitch/speed weighting active until the cycle-mean height error is near zero, instead of fading out at `0.05-0.8 m` raw instantaneous error.

**Step 1: Write failing TECS unit test**

Create or extend a TECS test file, for example `tests/test_tecs.py`.

Test behavior:

```python
def test_tecs_capture_blend_persists_until_recovered():
    tecs = PX4LikeTECS(PX4LikeTECSCfg(
        altitude_hold_error_band_m=0.05,
        altitude_capture_error_m=0.8,
        altitude_capture_release_error_m=0.02,
        altitude_capture_release_time_s=0.5,
        altitude_capture_persistence_gain=1.0,
    ), device=torch.device("cpu"))
    # Feed a -0.4 m error, then reduce raw error to -0.08 m.
    # Expected: capture_blend remains materially active instead of dropping near zero immediately.
```

Expected failure:

- New config fields are missing or `capture_blend` does not persist.

**Step 2: Implement minimum TECS state**

Add config fields in `PX4LikeTECSCfg`:

```python
altitude_capture_release_error_m: float = 0.03
altitude_capture_release_time_s: float = 0.4
altitude_capture_persistence_gain: float = 1.0
```

Add state:

```python
self._capture_active: Tensor | None = None
self._capture_release_timer_s: Tensor | None = None
```

Update logic:

- Activate capture if `abs(height_err_raw) > altitude_hold_error_band_m`.
- Release only after `abs(height_err_raw) < altitude_capture_release_error_m` for `altitude_capture_release_time_s`.
- Use `capture_blend = torch.maximum(capture_blend, capture_active.float() * altitude_capture_persistence_gain)`.
- Clamp final blend to `[0, 1]`.

Keep this in [tecs.py](/home/zn/IsaacLab/source/flapping_bot/flapping_bot/px4_like/tecs.py) near the current `capture_blend` calculation.

**Step 3: Expose params through cfg/CLI**

Add pass-through fields:

- `tecs_altitude_capture_release_error_m`
- `tecs_altitude_capture_release_time_s`
- `tecs_altitude_capture_persistence_gain`

Apply to:

- `PX4LikeStraightLineControllerCfg`
- `FlappingBotPathTrackingEnvCfg` as `teacher_tecs_*`
- `fly_path_mission.py` parser and `_configure_env`
- standalone `fly_straight_line.py`
- standalone `fly_loiter.py`

**Step 4: Run unit/config tests**

```bash
python -m pytest tests/test_tecs.py tests/test_fly_path_mission.py tests/test_path_tracking_env_contract.py -q
```

Expected:

- PASS.

**Step 5: Run aggressive straight sweep**

Use single-teacher logging and no warmup.

Candidate set:

| Candidate | gain | capture tc | release err | release time | persistence | pitch speed weight | capture pitch speed weight | damping | extra climb |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `current` | 1.8 | 0.65 | none | none | none | 0.50 | 0.25 | 0.16 | 1.0 |
| `capture_hold_2p4` | 2.4 | 0.55 | 0.03 | 0.35 | 1.00 | 0.40 | 0.15 | 0.22 | 1.2 |
| `capture_hold_3p0` | 3.0 | 0.45 | 0.03 | 0.35 | 1.00 | 0.35 | 0.10 | 0.26 | 1.4 |
| `fast_pitch_2p6` | 2.6 | 0.45 | 0.02 | 0.50 | 1.00 | 0.30 | 0.05 | 0.24 | 1.4 |
| `very_aggressive_guarded` | 3.4 | 0.38 | 0.02 | 0.50 | 1.00 | 0.25 | 0.00 | 0.30 | 1.6 |

Command template:

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py \
  --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 \
  --phase level_straight \
  --straight_length_m 180 \
  --steps 5000 \
  --no-path_warmup_enabled \
  --teacher_tecs_altitude_error_gain <gain> \
  --teacher_tecs_altitude_capture_time_const_s <tc> \
  --teacher_tecs_altitude_capture_release_error_m <release_err> \
  --teacher_tecs_altitude_capture_release_time_s <release_time> \
  --teacher_tecs_altitude_capture_persistence_gain <persistence> \
  --teacher_tecs_pitch_speed_weight <psw> \
  --teacher_tecs_pitch_speed_weight_capture <psw_capture> \
  --teacher_tecs_pitch_damping_gain <damping> \
  --teacher_tecs_capture_extra_climb_rate_mps <extra_climb> \
  --headless \
  --out_dir /tmp/straight_height_aggressive_sweep_20260411/<candidate>
```

**Step 6: Choose only a significant winner**

Accept a candidate only if all are true:

- Recovery time after min `<= 4.5 s`.
- Recovery distance after min `<= 35 m`.
- Cycle-mean second dip after `4.0 s` is no worse than `-0.15 m`.
- `freq_sat_fraction <= 0.25`.
- `elevon_pitch_sat_fraction <= 0.05`.
- No obvious pitch divergence or early path termination.

If no candidate meets the threshold, do not freeze defaults. Report failure and keep the instrumentation/sweep results.

**Step 7: Freeze defaults if there is a winner**

Apply the winning values to:

- `PX4LikeStraightLineControllerCfg`
- `FlappingBotPathTrackingEnvCfg`
- `fly_straight_line.py`
- `fly_loiter.py`
- summary fallbacks in `fly_path_mission.py`

**Step 8: Verify canonical tasks**

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 --phase level_straight --steps 2200 --headless --out_dir /tmp/straight_height_aggressive_final_20260411/canonical_straight
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 --phase level_turn --steps 2400 --headless --out_dir /tmp/straight_height_aggressive_final_20260411/canonical_turn
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 --phase level_loiter --steps 2800 --headless --out_dir /tmp/straight_height_aggressive_final_20260411/canonical_loiter
```

Expected:

- Straight satisfies the significant longitudinal threshold.
- Turn/loiter do not regress catastrophically in pitch/roll/path completion.

**Step 9: Commit**

```bash
git add source/flapping_bot/flapping_bot/px4_like/tecs.py \
  source/flapping_bot/flapping_bot/px4_like/straight_line_controller.py \
  source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py \
  scripts/flapping_px4/fly_path_mission.py \
  scripts/flapping_px4/fly_straight_line.py \
  scripts/flapping_px4/fly_loiter.py \
  scripts/flapping_px4/run_straight_height_recovery_sweep.py \
  tests/test_tecs.py \
  tests/test_fly_path_mission.py \
  tests/test_path_tracking_env_contract.py
git commit -m "tune: make straight height recovery aggressive"
```

## Task 3: Reduce Flap-Period Inner-Loop Chatter Without Killing Recovery

**Files:**

- Modify: `source/flapping_bot/flapping_bot/px4_like/straight_line_controller.py`
- Modify: `scripts/flapping_px4/fly_path_mission.py` only if extra CLI pass-through is needed
- Test: `tests/test_straight_line_controller.py` or `tests/test_path_tracking_env_contract.py`

**Design:**

The high-frequency tail motion follows the flapping-period `pitch_deg`/`w_b_y`, while `pitch_sp_deg` is low-frequency. Do not add a broad action LPF first. Instead, add a targeted inner-loop measurement mode: use a cycle-mean pitch/pitch-rate estimate for attitude feedback while keeping TECS setpoint fast.

**Step 1: Add failing test for cycle-mean inner pitch feedback**

Create `tests/test_straight_line_controller.py` if missing.

Test behavior:

```python
def test_inner_pitch_cycle_mean_filter_rejects_flap_period_ripple():
    controller = PX4LikeStraightLineController(
        PX4LikeStraightLineControllerCfg(
            inner_pitch_cycle_mean_enabled=True,
            inner_pitch_cycle_mean_tau_s=0.22,
        ),
        device=torch.device("cpu"),
    )
    # Feed constant pitch_sp but pitch measurement with 5 Hz ripple.
    # Expected: filtered pitch used for control has much smaller 5 Hz component than raw pitch.
```

Expected:

- FAIL because the config and filter do not exist.

**Step 2: Implement minimal cycle-mean filter**

Add fields:

```python
inner_pitch_cycle_mean_enabled: bool = False
inner_pitch_cycle_mean_tau_s: float = 0.22
inner_pitch_rate_cycle_mean_tau_s: float = 0.18
```

Implementation:

- Reuse the existing `_pitch_meas_filt` and `_pitch_rate_filt`, but if cycle-mean is enabled, use the cycle-mean taus instead of the faster existing taus.
- Keep this as a config switch; default off until a sweep proves it helps.

**Step 3: Sweep inner-loop filtering after the longitudinal winner**

Candidates:

| Candidate | pitch tau | pitch-rate tau | pitch kd | pitch tc | pitch action rate limit |
|---|---:|---:|---:|---:|---:|
| `no_extra_filter` | current | current | current | current | current |
| `cycle_mean_soft` | 0.22 | 0.18 | current | current | current |
| `cycle_mean_medium` | 0.30 | 0.24 | current | current | current |
| `cycle_mean_with_kd_down` | 0.30 | 0.24 | 0.11 | 0.35 | current |
| `cycle_mean_slower_tail` | 0.30 | 0.24 | 0.11 | 0.40 | 1.5 |

Accept only if:

- 4-8 Hz `action_elevon_pitch` RMS drops by at least `30%`.
- Straight recovery time worsens by no more than `15%` versus the Task 2 winner.
- No increase in min height error beyond `0.05 m`.
- No increase in post-5s lateral error beyond `0.10 m`.

**Step 4: Freeze defaults only if it passes**

If no candidate passes, do not freeze filter defaults. Keep this as an optional flag and report that the tail chatter needs a model-side/actuator-side treatment later.

**Step 5: Commit**

```bash
git add source/flapping_bot/flapping_bot/px4_like/straight_line_controller.py tests/test_straight_line_controller.py
git commit -m "tune: add inner pitch cycle-mean filtering"
```

## Task 4: Remove Straight-Line Lateral Steady Error

**Files:**

- Modify: `source/flapping_bot/flapping_bot/px4_like/straight_line_controller.py`
- Modify: `scripts/flapping_px4/fly_path_mission.py` only if CLI pass-through is needed
- Test: `tests/test_straight_line_controller.py`
- Analysis output: `docs/analysis/flapping_px4/straight_lateral_retune_20260411/`

**Design:**

Do not tune lateral gains until sign is verified. Current data show `lateral_error > 0`, `heading_sp < 0`, but roll remains positive for much of the run. That can still be valid depending on this project’s roll sign convention, so the first step is an explicit sign audit with a controlled state.

**Step 1: Write failing or diagnostic sign test**

Add a deterministic test:

```python
def test_straight_guidance_commands_corrective_heading_for_positive_lateral_error():
    # Vehicle is north/positive-y of a +x line, moving forward, no wind.
    # Expected heading_sp/course_sp points toward negative y.
```

If the project sign convention makes roll sign non-obvious, assert the heading/course sign first, not the roll sign.

**Step 2: Run the test**

```bash
python -m pytest tests/test_straight_line_controller.py::test_straight_guidance_commands_corrective_heading_for_positive_lateral_error -q
```

Expected:

- If FAIL: fix sign bug before gain tuning.
- If PASS: proceed to gain tuning.

**Step 3: Add lateral sweep CLI or script**

Reuse `fly_path_mission.py` overrides if already exposed. If not exposed, add pass-throughs for:

- `teacher_guidance_period_s`
- `teacher_guidance_roll_time_const_s`
- `teacher_heading_p_gain`
- `teacher_roll_kp`
- `teacher_roll_kd`

Default values should initially match `PX4LikeStraightLineControllerCfg`.

**Step 4: Run lateral sweep after Tasks 2-3**

Candidate set:

| Candidate | guidance period | roll time const | heading p | roll kp | roll kd |
|---|---:|---:|---:|---:|---:|
| `current` | current | current | current | current | current |
| `harder_heading` | 4.0 | 0.25 | 1.20 | 2.5 | 0.35 |
| `harder_roll` | 4.0 | 0.20 | 1.20 | 3.2 | 0.45 |
| `aggressive_lateral` | 3.2 | 0.18 | 1.50 | 3.5 | 0.50 |
| `guarded_aggressive` | 3.2 | 0.20 | 1.35 | 3.2 | 0.50 |

Run:

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py \
  --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 \
  --phase level_straight \
  --straight_length_m 180 \
  --steps 5000 \
  --no-path_warmup_enabled \
  --headless \
  --out_dir /tmp/straight_lateral_sweep_20260411/<candidate>
```

**Step 5: Accept only a significant winner**

Accept if:

- post-5s mean absolute lateral error `<= 0.25 m`.
- final absolute lateral error `<= 0.30 m`.
- longitudinal recovery remains within `15%` of Task 2/3 chosen result.
- no roll oscillation or roll saturation.

If no candidate meets this threshold, do not freeze lateral defaults. Report whether the issue is sign/actuation/authority rather than gain.

**Step 6: Canonical regression**

Run:

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 --phase level_straight --steps 2200 --headless --out_dir /tmp/straight_lateral_final_20260411/canonical_straight
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 --phase level_turn --steps 2400 --headless --out_dir /tmp/straight_lateral_final_20260411/canonical_turn
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 --phase level_loiter --steps 2800 --headless --out_dir /tmp/straight_lateral_final_20260411/canonical_loiter
```

Expected:

- Straight lateral target achieved.
- Turn/loiter do not regress substantially.

**Step 7: Commit**

```bash
git add source/flapping_bot/flapping_bot/px4_like/straight_line_controller.py \
  scripts/flapping_px4/fly_path_mission.py \
  tests/test_straight_line_controller.py \
  docs/analysis/flapping_px4/straight_lateral_retune_20260411
git commit -m "tune: reduce straight lateral tracking bias"
```

## Final Verification

Run all targeted checks:

```bash
python -m py_compile \
  source/flapping_bot/flapping_bot/px4_like/tecs.py \
  source/flapping_bot/flapping_bot/px4_like/straight_line_controller.py \
  source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py \
  scripts/flapping_px4/fly_path_mission.py \
  scripts/flapping_px4/fly_straight_line.py \
  scripts/flapping_px4/fly_loiter.py \
  scripts/flapping_px4/analyze_straight_height_recovery.py \
  scripts/flapping_px4/run_straight_height_recovery_sweep.py \
  scripts/flapping_px4/plot_path_mission.py
python -m pytest \
  tests/test_tecs.py \
  tests/test_straight_line_controller.py \
  tests/test_analyze_straight_height_recovery.py \
  tests/test_fly_path_mission.py \
  tests/test_path_tracking_env_contract.py \
  tests/test_plot_path_mission.py \
  -q
git diff --check
```

Run final non-RL DeLaurier canonical rollouts:

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 --phase level_straight --steps 2200 --headless --out_dir /tmp/aggressive_controller_final_20260411/canonical_straight
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 --phase level_turn --steps 2400 --headless --out_dir /tmp/aggressive_controller_final_20260411/canonical_turn
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 --phase level_loiter --steps 2800 --headless --out_dir /tmp/aggressive_controller_final_20260411/canonical_loiter
```

Create final report:

- `docs/analysis/flapping_px4/aggressive_controller_final_20260411/metrics_summary.csv`
- `docs/analysis/flapping_px4/aggressive_controller_final_20260411/metrics_summary.json`
- `docs/analysis/flapping_px4/aggressive_controller_final_20260411/notes.md`
- Straight/turn/loiter plots.

Final acceptance:

- Straight height recovery is materially faster than the single-teacher baseline.
- Straight lateral steady error is materially smaller.
- Control chatter is reduced if, and only if, the filtered inner-loop candidate does not slow height recovery too much.
- No conclusion relies on RL, simple QSM, or old double-call logs.
