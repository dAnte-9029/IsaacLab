# Rollout Horizon Unlock Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the non-RL rollout-side path horizon cap that truncates long injected missions before they can finish.

**Architecture:** Keep training-time env defaults unchanged and fix the evaluation script instead. Extend `fly_path_mission.py` so `ensure_episode_horizon()` raises both the config timeout and any path-specific `_path_episode_horizon_steps` buffer to the requested rollout length.

**Tech Stack:** Python 3.11, pytest, `scripts/flapping_px4/fly_path_mission.py`

---

### Task 1: Reproduce The Horizon Sync Bug In Tests

**Files:**
- Modify: `tests/test_fly_path_mission.py`

**Step 1: Write the failing test**

Add a unit test where:
- dummy env has `cfg.episode_length_s`
- dummy env also has `_path_episode_horizon_steps`
- `ensure_episode_horizon()` is called with a larger `effective_steps`
- assert both timeout surfaces are extended

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fly_path_mission.py -k path_specific_horizon -q`

Expected: FAIL because `_path_episode_horizon_steps` is left unchanged.

### Task 2: Apply Minimal Script-Level Fix

**Files:**
- Modify: `scripts/flapping_px4/fly_path_mission.py`
- Test: `tests/test_fly_path_mission.py`

**Step 1: Implement minimal fix**

Update `ensure_episode_horizon()` so it:
- computes `required_steps = effective_steps + 1`
- preserves the existing `cfg.episode_length_s` extension
- if `_path_episode_horizon_steps` exists, clamps it upward to `required_steps`

**Step 2: Run focused test**

Run: `python -m pytest tests/test_fly_path_mission.py -k path_specific_horizon -q`

Expected: PASS

**Step 3: Run broader script tests**

Run: `python -m pytest tests/test_fly_path_mission.py -q`

Expected: PASS

### Task 3: Verify With Previously Truncated Seeds

**Files:**
- No code changes required unless evidence says otherwise

**Step 1: Re-run representative truncated random seeds**

Re-run at least:
- one long loiter/straight seed that previously stopped at about `35.98 s`
- one shorter seed that previously stopped at about `30.47 s`

**Step 2: Confirm effect**

Verify:
- rollout duration exceeds the previous truncation point
- `failure_kind` is no longer `truncated` solely because of the capped path horizon
- progress ratio increases relative to the pre-fix truncated run

**Step 3: Commit**

```bash
git add scripts/flapping_px4/fly_path_mission.py tests/test_fly_path_mission.py docs/plans/2026-04-11-rollout-horizon-unlock.md
git commit -m "fix: extend rollout path horizons for long missions"
```
