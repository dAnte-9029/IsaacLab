# Reset Trim Toward Stable Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Retune reset defaults toward the observed long-straight stable state and verify that startup altitude loss is reduced.

**Architecture:** Keep behavior changes local to reset initialization defaults in the straight-flight and path-tracking env configs. Use long straight non-RL rollouts to choose values, then update contract tests and rerun the long-straight verification.

**Tech Stack:** Python 3.11, pytest, IsaacLab headless rollouts, CSV post-processing.

---

### Task 1: Pick reset candidates with rollout evidence

**Files:**
- Read: `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`
- Read: `scripts/flapping_px4/fly_path_mission.py`

**Step 1: Run a small long-straight reset sweep**

Run a narrow candidate set around the observed stable state and compare:
- first-3-second minimum `height_error_m`
- first-3-second `vertical_support_ratio`
- distance from `height_error_m <= -0.5` back to `>= 0`

**Step 2: Select one candidate**

Prefer the candidate that reduces startup dip and recovery distance without introducing early instability.

### Task 2: Update tests first

**Files:**
- Modify: `tests/test_path_tracking_env_contract.py`

**Step 1: Write the failing test**

Update the contract test for inherited reset defaults to the selected values.

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_path_tracking_env_contract.py -k retuned_reset_trim_defaults -q`

### Task 3: Implement reset default changes

**Files:**
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py`

**Step 1: Write minimal implementation**

Update only:
- `reset_pitch_deg`
- `reset_flap_hz`
- `reset_elevon_pitch_deg`

**Step 2: Run tests to verify they pass**

Run:
- `python -m pytest tests/test_path_tracking_env_contract.py -q`
- `python -m py_compile source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py`

### Task 4: Verify with a long straight rollout

**Files:**
- Read: `/tmp/.../trajectory_env0.csv`

**Step 1: Run long straight**

Run:
`./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 --phase level_straight --straight_length_m 180 --steps 5000 --headless --print_every 0`

**Step 2: Compare against current baseline**

Check:
- first 3 s minimum height error
- recovery distance from `-0.5 m` to `0`
- late-window mean height error

**Step 3: Commit**

Commit only if the startup dip is reduced and the long-straight rollout remains stable.
