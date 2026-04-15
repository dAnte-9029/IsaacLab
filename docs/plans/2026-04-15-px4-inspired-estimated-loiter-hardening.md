# PX4-Inspired Estimated Loiter Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the estimated loiter/path-tracking teacher behave more like PX4 under long curved-flight conditions by tightening TECS load-factor inputs, guidance-driven airspeed constraints, and estimated-only loiter protection.

**Architecture:** Keep a single controller implementation and extend its existing parameter surface. Reuse the shared straight/loiter/path-tracking TECS plumbing instead of creating controller forks. Apply stronger behavior only through estimated-state tuning/profile gates so the truth baseline remains effectively unchanged.

**Tech Stack:** Python 3.11, PyTorch controller code, pytest, IsaacLab runtime scripts.

---

### Task 1: Lock Down TECS Load-Factor Semantics

**Files:**
- Modify: `tests/test_px4_loiter_controller.py`
- Modify: `tests/test_px4_path_tracking_controller.py`
- Modify: `source/flapping_bot/flapping_bot/px4_like/straight_line_controller.py`

**Step 1: Write the failing tests**

- Add a loiter test proving TECS load factor follows actual `roll` by default even when `roll_sp` would be larger.
- Add a path-tracking test proving `tecs_load_factor_use_roll_sp=False` keeps curved-path load factor near 1.0 if actual `roll` is near zero.

**Step 2: Run tests to verify they fail**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_px4_loiter_controller.py tests/test_px4_path_tracking_controller.py -q
```

**Step 3: Write minimal implementation**

- Change the default TECS load-factor source from `roll_sp` to actual `roll`.
- Preserve `tecs_load_factor_use_roll_sp=True` as an opt-in override.

**Step 4: Run tests to verify they pass**

Run the same pytest command.

### Task 2: Feed Guidance-Required Airspeed Into Curved-Path TECS

**Files:**
- Modify: `tests/test_px4_path_tracking_controller.py`
- Modify: `tests/test_px4_loiter_controller.py`
- Modify: `source/flapping_bot/flapping_bot/px4_like/straight_line_controller.py`
- Modify: `source/flapping_bot/flapping_bot/px4_like/loiter_controller.py`
- Modify: `source/flapping_bot/flapping_bot/px4_like/path_tracking_controller.py`

**Step 1: Write the failing tests**

- Add a loiter/path-tracking test showing the curved-path TECS target airspeed rises when guidance requires more minimum airspeed, even without relying purely on a fixed base speed setpoint.
- Assert diagnostics expose the guidance-required minimum airspeed contribution.

**Step 2: Run tests to verify they fail**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_px4_loiter_controller.py tests/test_px4_path_tracking_controller.py -q
```

**Step 3: Write minimal implementation**

- Add a shared helper in the controller base to compute a PX4-like guidance-required minimum airspeed from course/bearing, wind, and configured minimum ground speed.
- Merge that floor with the existing bank-aware speed/min-airspeed logic before calling TECS.
- Expose diagnostics so runtime evidence can show which term dominates.

**Step 4: Run tests to verify they pass**

Run the same pytest command.

### Task 3: Add Estimated-Only Loiter Protection Surface

**Files:**
- Modify: `tests/test_controller_tuning_profiles.py`
- Modify: `source/flapping_bot/flapping_bot/px4_like/controller_tuning_profiles.py`
- Modify: `source/flapping_bot/flapping_bot/px4_like/straight_line_controller.py`
- Modify: `source/flapping_bot/flapping_bot/px4_like/loiter_controller.py`
- Modify: `source/flapping_bot/flapping_bot/px4_like/path_tracking_controller.py`
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py`
- Modify: `tests/test_path_tracking_env_contract.py`

**Step 1: Write the failing tests**

- Add profile tests for estimated-only loiter/path-tracking overrides.
- Add env-contract tests ensuring any new controller knobs propagate through the runtime config surface.

**Step 2: Run tests to verify they fail**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_controller_tuning_profiles.py tests/test_path_tracking_env_contract.py -q
```

**Step 3: Write minimal implementation**

- Add estimated-only parameter knobs for loiter recapture moderation / heading-quality protection without forking controller logic.
- Keep truth profile unchanged.

**Step 4: Run tests to verify they pass**

Run the same pytest command.

### Task 4: Verify Compilation, Unit Tests, and Runtime Regression

**Files:**
- Modify only as needed from prior tasks.

**Step 1: Run compile checks**

```bash
python -m py_compile source/flapping_bot/flapping_bot/px4_like/straight_line_controller.py
python -m py_compile source/flapping_bot/flapping_bot/px4_like/loiter_controller.py
python -m py_compile source/flapping_bot/flapping_bot/px4_like/path_tracking_controller.py
python -m py_compile source/flapping_bot/flapping_bot/px4_like/controller_tuning_profiles.py
```

**Step 2: Run targeted pytest**

```bash
./isaaclab.sh -p -m pytest tests/test_controller_tuning_profiles.py tests/test_px4_loiter_controller.py tests/test_px4_path_tracking_controller.py tests/test_path_tracking_env_contract.py tests/test_fly_path_mission.py -q
```

**Step 3: Run runtime regression**

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py --teacher_state_source estimated --policy_state_source estimated --imu_source synthetic --headless --mission_seed 25 --num_envs 1 --max_steps 3600 --total_mass_kg_override 0.95
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py --teacher_state_source estimated --policy_state_source estimated --imu_source synthetic --headless --mission_seed 30 --num_envs 1 --max_steps 3600 --total_mass_kg_override 0.95
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py --teacher_state_source estimated --policy_state_source estimated --imu_source synthetic --headless --mission_seed 33 --num_envs 1 --max_steps 3600 --total_mass_kg_override 0.95
```

**Step 4: Summarize evidence**

- Compare loiter-phase lateral error growth, alignment error growth, TAS creep, and completion ratio against the current `baseline_nowind` suite.
- Explicitly call out whether truth baseline behavior changed.
