# Truth Teacher Random Benchmark Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a repeatable truth-teacher random-mission benchmark for the path-tracking controller.

**Architecture:** Extend the existing `fly_path_mission.py` rollout path so it can inject either fixed canonical missions or sampled random missions, then add a batch runner that executes multiple seeded episodes and aggregates per-episode summaries. Reuse the existing teacher rollout, completion logic, and plotting format instead of building a second evaluation stack.

**Tech Stack:** Python 3.11, IsaacLab direct RL env, existing flapping PX4 teacher scripts, pytest.

---

### Task 1: Add benchmark helper tests

**Files:**
- Modify: `tests/test_fly_path_mission.py`
- Create: `tests/test_random_path_tracking_teacher_suite.py`

**Step 1: Write the failing tests**

- Add tests for deterministic random-mission construction from a seed.
- Add tests for aggregate benchmark summary helpers.
- Add tests that the random suite builds the expected seed list and output layout metadata.

**Step 2: Run test to verify it fails**

Run: `TERM=xterm ./isaaclab.sh -p -m pytest tests/test_fly_path_mission.py tests/test_random_path_tracking_teacher_suite.py -q`

Expected: FAIL because the random benchmark helpers do not exist yet.

**Step 3: Write minimal implementation**

- Add only the helpers needed by the tests.

**Step 4: Run test to verify it passes**

Run: `TERM=xterm ./isaaclab.sh -p -m pytest tests/test_fly_path_mission.py tests/test_random_path_tracking_teacher_suite.py -q`

Expected: PASS.

### Task 2: Extend single-rollout script for random missions

**Files:**
- Modify: `scripts/flapping_px4/fly_path_mission.py`

**Step 1: Write the failing test**

- Add a test that `fly_path_mission.py` can build a random mission from a seed and expose mission metadata.

**Step 2: Run test to verify it fails**

Run: `TERM=xterm ./isaaclab.sh -p -m pytest tests/test_fly_path_mission.py -q`

Expected: FAIL because random mission mode is missing.

**Step 3: Write minimal implementation**

- Add random mission sampling/injection helpers.
- Keep truth teacher path unchanged.
- Reuse existing rollout summary format.

**Step 4: Run tests**

Run: `TERM=xterm ./isaaclab.sh -p -m pytest tests/test_fly_path_mission.py tests/test_path_manager.py tests/test_path_tracking_baseline_suite.py -q`

Expected: PASS.

### Task 3: Add batch random benchmark runner

**Files:**
- Create: `scripts/flapping_px4/run_random_path_tracking_teacher_suite.py`
- Test: `tests/test_random_path_tracking_teacher_suite.py`

**Step 1: Write the failing test**

- Add tests for case generation, aggregate metrics, and summary writing.

**Step 2: Run test to verify it fails**

Run: `TERM=xterm ./isaaclab.sh -p -m pytest tests/test_random_path_tracking_teacher_suite.py -q`

Expected: FAIL because the runner does not exist.

**Step 3: Write minimal implementation**

- Iterate over seeds.
- Launch random teacher rollouts.
- Collect per-episode summaries into an aggregate JSON/CSV report.

**Step 4: Run tests**

Run: `TERM=xterm ./isaaclab.sh -p -m pytest tests/test_random_path_tracking_teacher_suite.py -q`

Expected: PASS.

### Task 4: Validate the full benchmark path

**Files:**
- Modify: `scripts/flapping_px4/run_random_path_tracking_teacher_suite.py`
- Modify: `scripts/flapping_px4/fly_path_mission.py`

**Step 1: Run a small smoke benchmark**

Run: `TERM=xterm ./isaaclab.sh -p scripts/flapping_px4/run_random_path_tracking_teacher_suite.py --episodes 3 --headless`

Expected: completion with aggregate `summary.json`.

**Step 2: Fix any rollout or aggregation bug**

- Address only the specific root cause revealed by smoke results.

**Step 3: Re-run smoke benchmark**

Run the same command again.

Expected: PASS with valid artifacts.

### Task 5: Run the first real teacher benchmark

**Files:**
- No required code changes unless benchmark reveals a concrete bug

**Step 1: Run a first benchmark pass**

Run: `TERM=xterm ./isaaclab.sh -p scripts/flapping_px4/run_random_path_tracking_teacher_suite.py --episodes 10 --headless`

Expected: aggregate metrics and per-episode logs.

**Step 2: Summarize results**

- Identify completion rate.
- Identify weakest mission patterns.
- Decide whether the next task is teacher tuning or RL progression.
