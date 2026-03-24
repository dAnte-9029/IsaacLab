# Loiter Teacher Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the loiter-related path-progress bug so the PX4-like teacher truly flies loiter segments instead of stalling at the entry point or teleporting to later segments.

**Architecture:** The fix stays local to the primitive path manager. First, lock down the bug with deterministic unit tests that reproduce the three failure modes: loiter entry stall, loiter exit teleport, and repeated-loiter ambiguity. Then update `PathManager.query()` and the arc candidate logic so overlapping segment endpoints cannot cause large forward progress jumps, while valid small forward motion on the active loiter remains admissible. Finally, rerun targeted teacher cases and the 50-case benchmark to separate geometry bugs from any remaining controller-tracking limits.

**Tech Stack:** Python 3.11, pytest, IsaacLab wrapper scripts, custom flapping-bot path manager and teacher rollout scripts.

---

### Task 1: Freeze the exact loiter regressions in unit tests

**Files:**
- Modify: `tests/test_path_manager.py`
- Test: `tests/test_path_manager.py`

**Step 1: Write the failing tests**

Add three deterministic tests:

```python
def test_straight_to_loiter_advances_into_arc_instead_of_sticking_to_entry():
    manager = PathManager(...)
    manager._last_progress_s = 60.0
    query = manager.query(position_xy=(61.0, 0.0), altitude_m=10.0, speed_mps=7.0)
    assert query.progress_s > 60.0


def test_loiter_to_straight_does_not_teleport_to_following_segment():
    manager = PathManager(...)
    query = manager.query(position_xy=(1.0, 0.0), altitude_m=10.0, speed_mps=7.0)
    assert query.progress_s < 5.0


def test_loiter_progress_remains_continuous_across_overlapping_endpoints():
    manager = PathManager(...)
    manager._last_progress_s = 80.0
    query = manager.query(position_xy=(59.86, 0.00014), altitude_m=10.0, speed_mps=7.0)
    assert 79.0 <= query.progress_s <= 81.0
```

Also add one “no teleport” regression over a short synthetic rollout:

```python
def test_loiter_missions_do_not_make_large_progress_jumps():
    progress = [manager.query(...).progress_s for ... in samples]
    deltas = [b - a for a, b in zip(progress, progress[1:])]
    assert max(deltas) < 5.0
```

**Step 2: Run tests to verify they fail**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_path_manager.py -q
```

Expected: at least the new loiter tests fail against the current implementation.

**Step 3: Commit the failing-test checkpoint**

```bash
git add tests/test_path_manager.py
git commit -m "test: capture loiter path-manager regressions"
```

---

### Task 2: Fix local loiter progress selection inside one arc

**Files:**
- Modify: `source/flapping_bot/flapping_bot/path_tracking/path_manager.py`
- Test: `tests/test_path_manager.py`

**Step 1: Change the closed-loop arc candidate policy**

Refactor the arc query path so closed-loop loiter segments do **not** prefer exact boundary candidates (`0`, `2π`, `4π`, ...) when the raw geometry indicates the aircraft has already moved slightly forward into the loop.

Concretely:

- keep `_arc_progress_candidates(...)`, but also track which candidates are:
  - geometry-derived continuous candidates
  - artificial boundary candidates added for wrap handling
- in `_query_segment(...)`, prefer:
  1. smallest progress gap to the current segment-relative progress
  2. non-boundary continuous candidate over an injected boundary candidate when both are nearly tied
  3. forward local motion over “snap back to segment start” at loiter entry

The core behavior after this step:

```python
target_progress_rad = ...
delta_candidates = ...
delta_rad = min(
    delta_candidates,
    key=lambda c: (
        abs(c.value - target_progress_rad),
        c.is_boundary,
        abs(c.value - c.raw_delta_hint),
    ),
)
```

Use the repo’s existing style; the exact shape can differ, but the semantics must match.

**Step 2: Run the focused unit tests**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_path_manager.py -q
```

Expected: the straight→loiter entry test passes and no existing path-manager tests regress.

**Step 3: Commit**

```bash
git add source/flapping_bot/flapping_bot/path_tracking/path_manager.py tests/test_path_manager.py
git commit -m "fix: preserve continuous progress within loiter arcs"
```

---

### Task 3: Add a forward-jump guard across overlapping segments

**Files:**
- Modify: `source/flapping_bot/flapping_bot/path_tracking/path_manager.py`
- Test: `tests/test_path_manager.py`

**Step 1: Tighten global segment selection in `query()`**

Today `query()` penalizes only backward jumps. Add a second guard against implausibly large **forward** jumps when multiple segments share the same point and tangent.

Implementation requirements:

- keep monotonic progress
- allow small forward motion inside the active segment
- forbid selecting a later overlapping segment when it would jump far beyond the current progress without spatial evidence

One acceptable pattern:

```python
forward_m = max(0.0, candidate.progress_s - self._last_progress_s - self._FORWARD_PROGRESS_TOL_M)
score = (
    candidate.distance_sq_xy
    + self._BACKWARD_PROGRESS_PENALTY * backward_m * backward_m
    + self._FORWARD_PROGRESS_PENALTY * forward_m * forward_m
)
```

Add a small symmetric forward tolerance (for example a few meters, clearly larger than one control step but far smaller than a whole loiter circumference).

**Step 2: Add or update tests for teleport prevention**

Strengthen the new tests so these cases hold:

- `loiter -> straight` near the common start/end point stays on loiter until the loop is actually finished
- `straight -> loiter` does not stay pinned to the straight endpoint
- `loiter -> loiter` does not jump an entire loop ahead

**Step 3: Run tests**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_path_manager.py tests/test_fly_path_mission.py -q
```

Expected: all path-manager and rollout-summary tests pass.

**Step 4: Commit**

```bash
git add source/flapping_bot/flapping_bot/path_tracking/path_manager.py tests/test_path_manager.py tests/test_fly_path_mission.py
git commit -m "fix: block loiter progress teleports across overlapping segments"
```

---

### Task 4: Verify with targeted failing seeds before full benchmark

**Files:**
- Modify: none
- Use: `scripts/flapping_px4/fly_path_mission.py`

**Step 1: Re-run the representative bad seeds**

Run these one by one:

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py --mission_mode random --mission_seed 3 --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 --num_envs 1 --steps 3400 --auto_extend_steps --nominal_speed_mps 7.0 --completion_margin_s 2.0 --headless
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py --mission_mode random --mission_seed 15 --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 --num_envs 1 --steps 3400 --auto_extend_steps --nominal_speed_mps 7.0 --completion_margin_s 2.0 --headless
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py --mission_mode random --mission_seed 30 --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 --num_envs 1 --steps 3400 --auto_extend_steps --nominal_speed_mps 7.0 --completion_margin_s 2.0 --headless
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py --mission_mode random --mission_seed 44 --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 --num_envs 1 --steps 3400 --auto_extend_steps --nominal_speed_mps 7.0 --completion_margin_s 2.0 --headless
```

**Step 2: Inspect acceptance criteria**

For each generated `summary.json` / `trajectory_env0.csv`:

- `final_progress_ratio` is no longer stuck at `0.0`, `0.323165`, `0.600000`, or `0.633359` just because a segment boundary was hit
- progress increases continuously through the loiter
- no single-step `progress_s` jump is on the order of `125 m` or `251 m`

**Step 3: If any seed still fails, stop and classify**

Only two valid remaining classes after the geometry fix:

- **path-manager bug remains**: progress still stalls or teleports at a boundary
- **controller-tracking limit**: progress is continuous, but the airframe physically departs from the path

Do not retune the controller until the first class is eliminated.

---

### Task 5: Run the clean 50-case teacher benchmark

**Files:**
- Modify: `scripts/flapping_px4/run_random_path_tracking_teacher_suite.py` (only if needed)
- Use: `scripts/flapping_px4/run_random_path_tracking_teacher_suite.py`
- Test: `tests/test_random_path_tracking_teacher_suite.py`

**Step 1: Remove the runner early-exit behavior if it is still present**

The benchmark runner currently stops at the first nonzero subprocess return. Change that behavior to record the failure and continue so one timeout does not invalidate the sweep.

**Step 2: Add/adjust a regression test for “continue on failure”**

If the runner code changes, add a small unit test in:

```python
tests/test_random_path_tracking_teacher_suite.py
```

to assert that one failed episode does not prevent later episodes from being launched.

**Step 3: Run the 50-case suite**

Run:

```bash
./isaaclab.sh -p scripts/flapping_px4/run_random_path_tracking_teacher_suite.py --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 --episodes 50 --seed_start 0 --steps 3400 --auto_extend_steps --nominal_speed_mps 7.0 --completion_margin_s 2.0 --headless
```

**Step 4: Compare against the current baseline**

Current baseline to beat:

- overall completion over 50 seeds: about `0.66`
- loiter-containing missions: effectively unreliable

Acceptance target for this fix phase:

- no fake success caused by progress teleport
- the previously pathological loiter seeds no longer fail due to geometry bookkeeping
- if overall completion is still limited, the remaining misses are genuine controller issues rather than path-manager aliasing

**Step 5: Commit**

```bash
git add scripts/flapping_px4/run_random_path_tracking_teacher_suite.py tests/test_random_path_tracking_teacher_suite.py
git commit -m "chore: make teacher benchmark robust to individual episode failures"
```

---

### Task 6: Document post-fix findings and next branch point

**Files:**
- Modify: `docs/plans/2026-03-20-loiter-teacher-fix.md`
- Modify: benchmark output summaries under `logs/flapping_px4/...` (generated, not committed unless explicitly desired)

**Step 1: Record the outcome**

Append a short results section to this plan document with:

- fixed root cause
- targeted seeds before/after
- 50-case completion rate before/after
- whether any remaining failures are now controller-only

**Step 2: Choose the next branch**

If loiter geometry is fixed and completion is still not high enough, the next work item is **teacher control tuning**, not more path-manager surgery.

If loiter geometry is not fixed, do not start RL teacher-guided training yet.

**Step 3: Final commit**

```bash
git add docs/plans/2026-03-20-loiter-teacher-fix.md
git commit -m "docs: record loiter teacher fix plan and results"
```
