# Random Teacher Progress Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the random teacher benchmark report completion and progress correctly for loiter-rich missions, then re-run the 50-episode benchmark.

**Architecture:** Add regression tests first for the two observed failures: path-manager segment snapping at overlapping junctions and rollout summaries dropping the completion-triggering step. Then implement the smallest changes in the path query / mission summary logic so the teacher benchmark reflects actual mission completion instead of a bookkeeping artifact.

**Tech Stack:** Python 3.11, pytest, IsaacLab flapping-bot path-tracking scripts.

---

### Task 1: Add regression tests

**Files:**
- Modify: `tests/test_path_manager.py`
- Modify: `tests/test_fly_path_mission.py`

**Step 1: Write the failing test**

- Add a path-manager test proving that a mission starting with `loiter` does not jump directly onto the following straight segment when the vehicle is at the shared junction point.
- Add a rollout-summary test proving that completion uses the final observed progress from the completion-triggering step rather than the previous logged step.

**Step 2: Run test to verify it fails**

Run: `./isaaclab.sh -p -m pytest tests/test_path_manager.py tests/test_fly_path_mission.py -q`

Expected: the new regression tests fail for the current implementation.

### Task 2: Implement minimal fix

**Files:**
- Modify: `source/flapping_bot/flapping_bot/path_tracking/path_manager.py`
- Modify: `scripts/flapping_px4/fly_path_mission.py`

**Step 1: Write minimal implementation**

- Tighten `PathManager.query()` tie-breaking so the manager does not skip an in-progress closed/open segment at a shared endpoint merely because another segment has zero Euclidean distance.
- Extract summary completion bookkeeping in `fly_path_mission.py` so the final progress/completion state is computed from the last evaluated step, including the step that triggered success.

**Step 2: Run tests to verify they pass**

Run: `./isaaclab.sh -p -m pytest tests/test_path_manager.py tests/test_fly_path_mission.py tests/test_random_path_tracking_teacher_suite.py -q`

Expected: all targeted tests pass.

### Task 3: Re-run benchmark verification

**Files:**
- Reuse: `scripts/flapping_px4/run_random_path_tracking_teacher_suite.py`

**Step 1: Run benchmark**

Run: `./isaaclab.sh -p scripts/flapping_px4/run_random_path_tracking_teacher_suite.py --episodes 50 --headless --print_every 0 --episode_timeout_s 600`

**Step 2: Inspect outputs**

- Check `logs/flapping_px4/random_path_tracking_teacher_suite/<timestamp>/summary.json`
- Check `logs/flapping_px4/random_path_tracking_teacher_suite/<timestamp>/manifest.json`

**Step 3: Confirm no low-progress false successes remain**

- Verify low `final_progress_ratio` cases no longer coexist with `completed_path=true` due to bookkeeping artifacts.
