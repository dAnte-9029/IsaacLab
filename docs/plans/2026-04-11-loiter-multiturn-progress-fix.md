# Loiter Multi-Turn Progress Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the path-manager bug that causes multi-turn loiter reference progress to stick at the first full-loop boundary instead of advancing into later laps.

**Architecture:** Keep the controller stack unchanged and fix the bug at the path-geometry layer. Add one regression test that reproduces the exact boundary-sticking failure, then minimally adjust the closed-loop arc progress selection so later-lap loiter progress can unwrap forward across `2π` boundaries without breaking the existing anti-teleport protections.

**Tech Stack:** Python 3.11, pytest, `flapping_bot.path_tracking.PathManager`

---

### Task 1: Reproduce The Boundary-Sticking Bug In Tests

**Files:**
- Modify: `tests/test_path_manager.py`

**Step 1: Write the failing test**

Add a regression test that:
- builds a single `loiter` mission with `loiter_turns=1.5`
- sets `_last_progress_s` to exactly one full-loop boundary
- queries a position slightly past the loiter start angle on the second lap
- asserts `progress_s` advances past the boundary instead of staying equal to it

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_path_manager.py -k second_lap -q`

Expected: FAIL because `progress_s` stays pinned at the one-loop boundary.

### Task 2: Apply Minimal Closed-Loop Progress Unwrap Fix

**Files:**
- Modify: `source/flapping_bot/flapping_bot/path_tracking/path_manager.py`
- Test: `tests/test_path_manager.py`

**Step 1: Write minimal implementation**

Adjust the closed-loop arc selection in `_query_segment()` so that when:
- the segment spans at least one full loop
- the prior segment-local progress is exactly at a loop boundary
- the current angular measurement is slightly ahead of that boundary

the chosen candidate advances to the forward unwrapped lap candidate instead of snapping back to the exact boundary point.

Keep the existing protections against:
- skipping ahead by a full loop
- jumping into later segments
- teleporting backward

**Step 2: Run the focused test**

Run: `python -m pytest tests/test_path_manager.py -k second_lap -q`

Expected: PASS

**Step 3: Run the full path-manager suite**

Run: `python -m pytest tests/test_path_manager.py -q`

Expected: PASS with no regressions in the existing closed-loop overlap cases.

### Task 3: Verify With The Real Failing Seeds

**Files:**
- No code changes required unless regression evidence says otherwise

**Step 1: Re-run representative loiter-heavy seeds**

Run the seeded non-RL random mission rollouts for representative failing loiter-heavy cases such as:
- `seed=1`
- `seed=3`
- `seed=5`

Use the same `run_random_path_tracking_teacher_suite` mission geometry or direct `fly_path_mission.py --mission_mode random` reproduction.

**Step 2: Check the post-fix symptom**

Confirm that after the first full loop:
- `progress_s` no longer stays fixed at the one-loop boundary
- `reference_x/reference_y` continue moving around the loiter path
- the rollout no longer loses the target immediately because of a frozen reference point

**Step 3: Commit**

```bash
git add tests/test_path_manager.py source/flapping_bot/flapping_bot/path_tracking/path_manager.py docs/plans/2026-04-11-loiter-multiturn-progress-fix.md
git commit -m "fix: advance multi-turn loiter progress past loop boundary"
```
