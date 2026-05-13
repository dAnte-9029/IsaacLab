# Estimated Path Tracking Lateral Robustness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce `estimated + synthetic` path-tracking lateral error by making the PX4-like lateral loop less sensitive to estimated velocity-heading bias and lag.

**Architecture:** Keep one shared controller implementation and add a small, opt-in lateral-heading fusion block in the controller base class. Use profile parameters so only the `estimated_teacher/path_tracking` tuning path enables the new behavior and adjusts heading-loop aggressiveness.

**Tech Stack:** Python 3.11, PyTorch controllers, pytest, IsaacLab rollout scripts.

---

### Task 1: Add failing tests for estimated lateral-heading robustness

**Files:**
- Modify: `tests/test_controller_tuning_profiles.py`
- Modify: `tests/test_px4_path_tracking_controller.py`

**Step 1: Write the failing tests**

- Add a profile test that expects `estimated_teacher/path_tracking` to override:
  - `heading_p_gain`
  - `lateral_heading_yaw_blend`
  - `lateral_heading_yaw_correction_limit_deg`
- Add a controller behavior test that creates a yaw / velocity-heading mismatch and expects the estimated-style config to command a smaller absolute `roll_sp` than the baseline config.

**Step 2: Run tests to verify they fail**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_controller_tuning_profiles.py tests/test_px4_path_tracking_controller.py -q
```

Expected: failures because the new config keys and behavior do not exist yet.

### Task 2: Implement shared controller heading-fusion support

**Files:**
- Modify: `source/flapping_bot/flapping_bot/px4_like/straight_line_controller.py`
- Modify: `source/flapping_bot/flapping_bot/px4_like/loiter_controller.py`
- Modify: `source/flapping_bot/flapping_bot/px4_like/path_tracking_controller.py`

**Step 1: Add config parameters**

- Add controller config fields:
  - `lateral_heading_yaw_blend`
  - `lateral_heading_yaw_correction_limit_deg`

Defaults should preserve current truth behavior.

**Step 2: Add a shared helper in the base controller**

- Resolve lateral heading from:
  - velocity-derived air-velocity heading
  - optional yaw correction blend
  - optional yaw-correction clamp
- Return diagnostics for:
  - velocity-heading
  - heading used by lateral control
  - applied yaw correction

**Step 3: Wire the helper into straight / loiter / path-tracking**

- Replace direct `heading = atan2(air_vel_xy...)` use in the heading controller path with the shared helper.
- Keep wind correction, guidance, and inner loop logic unchanged.

### Task 3: Enable the new behavior only for estimated path tracking

**Files:**
- Modify: `source/flapping_bot/flapping_bot/px4_like/controller_tuning_profiles.py`

**Step 1: Extend estimated path-tracking overrides**

- Add estimated path-tracking profile overrides for:
  - `heading_p_gain`
  - `lateral_heading_yaw_blend`
  - `lateral_heading_yaw_correction_limit_deg`

**Step 2: Keep truth and non-path defaults stable**

- Do not change truth defaults.
- Avoid creating a second controller implementation.

### Task 4: Verify unit tests and runtime behavior

**Files:**
- None

**Step 1: Re-run targeted tests**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_controller_tuning_profiles.py tests/test_px4_path_tracking_controller.py -q
```

Expected: pass.

**Step 2: Re-run path-tracking comparisons**

Run:

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py --mission_mode random --mission_seed 33 --seed 33 --steps 3600 --auto_extend_steps --nominal_speed_mps 7 --completion_margin_s 3 --teacher_state_source estimated --policy_state_source estimated --imu_source synthetic --total_mass_kg_override 0.95 --straight_length_m 80 --turn_radius_m 25 --loiter_radius_m 40 --loiter_turns 1.5 --headless
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py --mission_mode random --mission_seed 30 --seed 30 --steps 3600 --auto_extend_steps --nominal_speed_mps 7 --completion_margin_s 3 --teacher_state_source estimated --policy_state_source estimated --imu_source synthetic --total_mass_kg_override 0.95 --straight_length_m 80 --turn_radius_m 25 --loiter_radius_m 40 --loiter_turns 1.5 --headless
```

Collect:
- post-warmup lateral error
- post-warmup align error
- roll saturation fraction from trajectory logs

**Step 3: Compare against the current baseline**

- Compare to:
  - `logs/flapping_px4/estimated_path_tracking_complex_suite/baseline_nowind/20260414_214351/cases/random_seed_000033/20260414_214920`
  - `logs/flapping_px4/estimated_path_tracking_complex_suite/baseline_nowind/20260414_214351/cases/random_seed_000030/20260414_214644`

**Step 4: Sanity-check truth remains stable**

Run one truth spot check:

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py --mission_mode random --mission_seed 33 --seed 33 --steps 3600 --auto_extend_steps --nominal_speed_mps 7 --completion_margin_s 3 --teacher_state_source truth --policy_state_source truth --imu_source synthetic --total_mass_kg_override 0.95 --straight_length_m 80 --turn_radius_m 25 --loiter_radius_m 40 --loiter_turns 1.5 --headless
```

Expected: no material regression.
