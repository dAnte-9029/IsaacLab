# Straight Altitude Recovery Retune Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce long-straight altitude recovery distance under the hard `freq_hz <= 5.0` constraint, without re-expanding scope to turn/loiter.

**Architecture:** Keep the current self-consistent plant fixed, use `fly_path_mission.py` on `level_straight` with a longer straight segment, and optimize two layers in order: reset trim first, then aggressive longitudinal control. Success is not "lower final error" alone; it is a shorter distance to recover from `height_error = -0.5 m` back to `0 m`.

**Tech Stack:** IsaacLab headless rollouts, controller-only PX4-like teacher, Python CSV/JSON analysis, existing sweep entrypoints under `scripts/flapping_px4/`.

---

### Task 1: Lock the straight-only scoring target

**Files:**
- Modify: `docs/plans/2026-04-10-straight-altitude-recovery-retune.md`
- Read: `scripts/flapping_px4/fly_path_mission.py`
- Read: `/tmp/long_straight_baseline_20260410/level_straight/20260410_110756/summary.json`
- Read: `/tmp/long_straight_baseline_20260410/level_straight/20260410_110756/trajectory_env0.csv`

**Step 1: Record the baseline**

Baseline is the current default long straight:
- `phase=level_straight`
- `straight_length_m=180`
- `steps=5000`
- current default plant + controller

**Step 2: Use these metrics**

Primary metric:
- distance from first `height_error <= -0.5 m` to first `height_error >= 0.0 m`

Secondary metrics:
- minimum `height_error` in `0-3 s`
- distance from minimum `height_error` back to `0.0 m`
- `8 s-end` mean `height_error`
- `8 s-end` mean `exec_freq_hz`

Hard constraints:
- no early termination before `progress_ratio >= 0.97`
- `freq_hz` never exceeds `5.0`
- no obvious pitch instability

**Step 3: Commit nothing**

This task is analysis-only.

### Task 2: Sweep reset trim on long straight

**Files:**
- Reuse: `scripts/flapping_px4/run_reset_trim_sweep.py`
- Read: generated `summary.json` and `trajectory_env0.csv`

**Step 1: Run a focused trim sweep**

Use only long straight and keep controller defaults fixed.

Search dimensions:
- `reset_pitch_deg`
- `reset_flap_hz`
- `reset_elevon_pitch_deg`

Start with a compact grid around the current default, biased more aggressive than the current trim.

**Step 2: Rank by startup loss and recovery distance**

Prefer candidates that:
- reduce `0-3 s` minimum height loss
- reduce the `-0.5 m -> 0.0 m` recovery distance
- keep `8 s-end` mean height error near zero

**Step 3: Keep the top 2 candidates**

Do not edit defaults yet.

### Task 3: Sweep aggressive longitudinal control on the best trim candidate

**Files:**
- Read/modify only if needed:
  - `scripts/flapping_px4/fly_path_mission.py`
  - `source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py`
  - `source/flapping_bot/flapping_bot/px4_like/straight_line_controller.py`

**Step 1: Reuse existing CLI parameters first**

Sweep only a small set of longitudinal parameters:
- `teacher_pitch_kp`
- `teacher_inner_pitch_ki`
- optionally `teacher_tecs_load_factor_pitch_compensation_gain`

Do not touch roll/yaw gains.
Do not raise `freq` ceiling.

**Step 2: Bias toward faster recovery**

Accept more aggressive longitudinal behavior if it stays bounded.

**Step 3: Compare against trim-only winner**

Keep the smallest controller change that materially shortens recovery distance.

### Task 4: Freeze the best straight-only candidate and verify

**Files:**
- Modify only if a winner is clear:
  - `source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py`
  - `source/flapping_bot/flapping_bot/px4_like/straight_line_controller.py`
- Test:
  - targeted `py_compile`
  - relevant `pytest`

**Step 1: Apply the minimal winning default**

If the winner is trim-only, change only reset defaults.
If the winner needs controller help, change only the single winning controller parameter.

**Step 2: Verify on fresh long straight**

Run the same `180 m` straight again and confirm:
- less startup loss or faster recovery
- shorter `-0.5 m -> 0.0 m` distance
- no instability regression

**Step 3: Optional smoke**

Run one short canonical `turn` or `loiter` smoke only to ensure no obvious regression.

### Task 5: Document results and commit

**Files:**
- Create/update under: `docs/analysis/flapping_px4/`

**Step 1: Save comparison artifacts**

Keep:
- baseline long straight plot
- winning candidate long straight plot
- recovery distance table

**Step 2: Commit the minimal change**

Use a focused commit message describing straight recovery retune.
