# Estimated Synthetic Controller Initial Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Apply a minimal, evidence-backed initial fix for `estimated + synthetic` controller errors without changing truth teacher, RL, IMU plumbing, or duplicating controller logic.

**Architecture:** Keep one controller implementation and extend the existing estimated-teacher tuning profile surface. Apply the loiter roll action-rate change only where evidence supports it, keep straight-line behavior unchanged for now, and gate TECS load-factor changes behind an explicit experiment before making it default.

**Tech Stack:** Python 3.11, IsaacLab wrapper `./isaaclab.sh`, pytest, existing `flapping_bot.px4_like` controller configs and scripts.

---

### Task 1: Add Loiter-Specific Estimated Profile Surface

**Files:**
- Modify: `source/flapping_bot/flapping_bot/px4_like/controller_tuning_profiles.py`
- Modify: `tests/test_controller_tuning_profiles.py`

**Step 1: Write the failing test**

Add a test that proves the estimated profile can apply a loiter-only override without changing straight-line kwargs.

```python
def test_estimated_profile_can_apply_loiter_only_roll_rate_override():
    straight_kwargs = {
        "max_roll_deg": 45.0,
        "roll_kd": 0.85,
        "inner_elevon_roll_rate_limit_per_s": 6.0,
    }
    loiter_kwargs = dict(straight_kwargs)

    straight_result = apply_controller_tuning_profile(
        straight_kwargs,
        controller_state_source="estimated",
        controller_kind="straight_line",
    )
    loiter_result = apply_controller_tuning_profile(
        loiter_kwargs,
        controller_state_source="estimated",
        controller_kind="loiter",
    )

    assert straight_result["inner_elevon_roll_rate_limit_per_s"] == 6.0
    assert loiter_result["inner_elevon_roll_rate_limit_per_s"] == 18.0
```

**Step 2: Run test to verify it fails**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_controller_tuning_profiles.py -q
```

Expected: FAIL because `apply_controller_tuning_profile()` does not yet accept `controller_kind`.

**Step 3: Write minimal implementation**

Update `apply_controller_tuning_profile()` to accept a keyword-only optional `controller_kind: str | None = None`.

Keep existing estimated overrides unchanged:

```python
ESTIMATED_CONTROLLER_OVERRIDES = {
    "max_roll_deg": 35.0,
    "roll_kd": 0.55,
}
```

Add a narrow loiter-only override map:

```python
ESTIMATED_CONTROLLER_KIND_OVERRIDES = {
    "loiter": {
        "inner_elevon_roll_rate_limit_per_s": 18.0,
    },
}
```

Implementation rule:

```python
if controller_state_source == "estimated":
    apply global estimated overrides
    if controller_kind is not None:
        apply ESTIMATED_CONTROLLER_KIND_OVERRIDES.get(controller_kind, {})
```

Do not add straight-line overrides.

**Step 4: Run test to verify it passes**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_controller_tuning_profiles.py -q
```

Expected: PASS.

---

### Task 2: Wire Controller Kind Into Non-RL Scripts

**Files:**
- Modify: `scripts/flapping_px4/fly_straight_line.py`
- Modify: `scripts/flapping_px4/fly_loiter.py`
- Modify: `tests/test_controller_tuning_profiles.py`

**Step 1: Write the failing test**

Extend the profile test to assert script-facing behavior through the helper rather than running Isaac Sim:

```python
def test_truth_profile_does_not_apply_loiter_roll_rate_override():
    kwargs = {
        "max_roll_deg": 45.0,
        "roll_kd": 0.85,
        "inner_elevon_roll_rate_limit_per_s": 6.0,
    }

    result = apply_controller_tuning_profile(
        kwargs,
        controller_state_source="truth",
        controller_kind="loiter",
    )

    assert result["max_roll_deg"] == 45.0
    assert result["roll_kd"] == 0.85
    assert result["inner_elevon_roll_rate_limit_per_s"] == 6.0
```

**Step 2: Run test to verify current helper behavior**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_controller_tuning_profiles.py -q
```

Expected: PASS after Task 1 implementation.

**Step 3: Wire script calls**

In `scripts/flapping_px4/fly_straight_line.py`, call:

```python
controller_kwargs = apply_controller_tuning_profile(
    controller_kwargs,
    controller_state_source=state_source_selection.controller_state_source,
    controller_kind="straight_line",
)
```

In `scripts/flapping_px4/fly_loiter.py`, call:

```python
controller_kwargs = apply_controller_tuning_profile(
    controller_kwargs,
    controller_state_source=state_source_selection.controller_state_source,
    controller_kind="loiter",
)
```

Do not change CLI defaults except for existing experiment flags.

**Step 4: Run compile checks**

Run:

```bash
python -m py_compile scripts/flapping_px4/fly_straight_line.py
python -m py_compile scripts/flapping_px4/fly_loiter.py
```

Expected: both commands exit 0.

---

### Task 3: Validate Estimated Synthetic Runtime Impact

**Files:**
- Read: `logs/flapping_px4/straight_line/20260412_191311/summary.json`
- Read: `logs/flapping_px4/loiter/20260412_191522/summary.json`
- New logs: `logs/flapping_px4/straight_line/`
- New logs: `logs/flapping_px4/loiter/`

**Step 1: Run straight estimated synthetic**

Run:

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_straight_line.py --state_source estimated --teacher_state_source estimated --imu_source synthetic --steps 1200 --headless
```

Expected: no regression target for initial acceptance. Because the roll-rate experiment worsened straight-line behavior, the target is to remain near baseline:

```text
mean_abs_track_error_m <= 0.25
final track error not materially worse than 0.773 m
```

If straight-line becomes worse, stop and remove any straight-line-specific application of the loiter override.

**Step 2: Run loiter estimated synthetic**

Run:

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_loiter.py --state_source estimated --teacher_state_source estimated --imu_source synthetic --steps 1200 --min_loiter_turns 0.5 --metrics_warmup_s 2.0 --headless
```

Expected acceptance targets based on experiment `logs/flapping_px4/loiter_rate_sweep/20260412_212852`:

```text
mean_abs_height_error_m <= 0.80
mean_abs_height_error_post_warmup_m <= 0.90
mean_abs_track_error_m <= 0.28
```

**Step 3: Verify truth is not affected by profile**

Run:

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_loiter.py --state_source truth --teacher_state_source truth --imu_source synthetic --steps 1200 --min_loiter_turns 0.5 --metrics_warmup_s 2.0 --headless
```

Expected: summary remains near prior forensic baseline:

```text
mean_abs_track_error_m ~= 0.244
mean_abs_height_error_m ~= 2.608
```

This is a guardrail only; do not tune truth in this task.

---

### Task 4: Decide Whether TECS Actual-Roll Load Factor Should Become Default

**Files:**
- Modify only if evidence supports it: `source/flapping_bot/flapping_bot/px4_like/controller_tuning_profiles.py`
- Read: `logs/flapping_px4/loiter_tecs_lf_sweep/20260412_213136/summary.json`
- Read: `logs/flapping_px4/loiter_tecs_lf_sweep/20260412_213243/summary.json`

**Step 1: Keep TECS actual-roll off by default initially**

Do not add this to the default estimated profile in the first patch:

```python
"tecs_load_factor_use_roll_sp": False
```

Rationale: actual-roll load factor helped loiter height error, but the combination with roll-rate-only was worse than roll-rate-only:

```text
roll-rate-only post-warmup height: 0.738 m
roll-rate + actual-roll post-warmup height: 1.677 m
```

**Step 2: Add a note to the final report**

Report it as a second-stage candidate, not part of the first fix.

Expected decision text:

```text
TECS actual-roll load factor is evidence-supported as a secondary effect, but it should not be merged into the first default estimated profile because it degraded the best observed loiter roll-rate-only result.
```

---

### Task 5: Run Required Verification

**Files:**
- Verify: `source/flapping_bot/flapping_bot/px4_like/controller_tuning_profiles.py`
- Verify: `scripts/flapping_px4/fly_straight_line.py`
- Verify: `scripts/flapping_px4/fly_loiter.py`
- Verify: `tests/test_controller_tuning_profiles.py`

**Step 1: Run compile checks**

Run:

```bash
python -m py_compile source/flapping_bot/flapping_bot/px4_like/controller_tuning_profiles.py
python -m py_compile scripts/flapping_px4/fly_straight_line.py
python -m py_compile scripts/flapping_px4/fly_loiter.py
```

Expected: all exit 0.

**Step 2: Run targeted unit tests**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_controller_tuning_profiles.py tests/test_train_and_watch.py tests/test_eval_suites.py tests/test_fly_controller_state_source_contract.py tests/test_imu_provider.py tests/test_sensor_state_estimator.py tests/test_rl_teacher_guidance.py tests/test_path_tracking_env_contract.py -q
```

Expected: all tests pass.

**Step 3: Run runtime evidence commands**

Run:

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_straight_line.py --state_source estimated --teacher_state_source estimated --imu_source synthetic --steps 1200 --headless
./isaaclab.sh -p scripts/flapping_px4/fly_loiter.py --state_source estimated --teacher_state_source estimated --imu_source synthetic --steps 1200 --min_loiter_turns 0.5 --metrics_warmup_s 2.0 --headless
```

Expected:

```text
straight: no material regression from 20260412_191311
loiter: close to 20260412_212852 roll-rate-only improvement
```

**Step 4: Summarize with evidence**

Final report must include:

```text
1. truth teacher was not tuned
2. estimated profile changed only by loiter-specific roll action-rate limit
3. straight did not inherit the loiter override
4. loiter estimated+synthetic improved or failed acceptance with exact metrics
5. TECS actual-roll load-factor remains a candidate, not default
```
