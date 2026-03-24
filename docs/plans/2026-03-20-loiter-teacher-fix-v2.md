# Loiter Teacher Fix V2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate fake teacher success by making path progression segment-aware, so loiter-rich missions cannot teleport across overlapping segment boundaries.

**Architecture:** The current bug is not primarily a controller problem. It is a path-manager state problem: `PathManager.query()` scores all segments globally using only distance plus a scalar progress hysteresis, while many missions contain exact geometric overlaps (`loiter -> straight`, `loiter -> loiter`, `turn -> loiter -> straight`). Once the vehicle moves about 2 m past a shared boundary, the current forward-jump gate disables the penalty and a later segment can look slightly closer than the active loiter, producing a non-physical progress jump of about one loop (`~125.7 m`) or two loops (`~251.3 m`). The fix is to stop relying on a brittle distance-only global tie-break and instead track active-segment continuity explicitly.

**Tech Stack:** Python 3.11, pytest, IsaacLab wrapper scripts, flapping-bot path manager, teacher rollout and benchmark scripts.

---

### Task 1: Freeze the real remaining failures with deterministic regressions

**Files:**
- Modify: `tests/test_path_manager.py`
- Test: `tests/test_path_manager.py`

**Step 1: Add exact regression cases matching the surviving bad patterns**

Add deterministic tests for these mission topologies:

- `loiter -> straight`
- `loiter -> loiter -> straight`
- `turn -> loiter -> straight`
- `loiter -> loiter -> loiter`

Each test should place the aircraft near a shared boundary at the same geometry seen in the logs and assert:

- no jump larger than `5 m` in one query step
- no transition to a segment more than one step ahead
- current segment progress stays continuous when the aircraft is only `~2 m` past the overlap point

Use explicit positions already observed in the logs, for example:

```python
position_xy = (2.0234274864196777, 0.028812089934945107)  # seed 1 / 40
position_xy = (19.60279083251953, 22.022756576538086)     # seed 30
```

**Step 2: Add a score-diagnostics helper test**

Add one unit test that inspects raw candidates and proves the current failure mode:

- active loiter candidate is physically correct
- later straight/loiter candidate is only slightly closer in XY
- forward-jump penalty is incorrectly disabled because `boundary_distance_m` is just over `2.0`

This locks down the real root cause and prevents another “looks fixed but still teleports” iteration.

**Step 3: Run the tests**

```bash
export TERM=xterm
./isaaclab.sh -p -m pytest tests/test_path_manager.py -q
```

Expected: the new regressions fail against the current implementation.

---

### Task 2: Replace scalar-only hysteresis with segment-aware continuity state

**Files:**
- Modify: `source/flapping_bot/flapping_bot/path_tracking/path_manager.py`
- Test: `tests/test_path_manager.py`

**Step 1: Extend internal state**

Track at least:

- `self._last_segment_idx`
- `self._last_segment_progress_s`

Keep `self._last_progress_s` for public monotonic progress, but do not use it as the only state for segment selection.

**Step 2: Restrict query candidates to a local transition neighborhood**

Instead of letting any segment win globally, evaluate:

- current active segment
- immediate next segment
- optionally immediate previous segment only for numerical robustness

Do **not** allow jumping over one or more whole segments in a single query.

For any candidate with `segment_idx > self._last_segment_idx + 1`, either:

- reject it directly, or
- assign an effectively infinite penalty

This removes the entire class of `0 -> 2`, `1 -> 3`, or `0 -> 2 -> straight` teleports.

**Step 3: Add an explicit transition rule**

Only allow switching from segment `i` to segment `i+1` when both are true:

- current-segment local progress is near its end
- next-segment local progress is near its start and geometrically plausible

One acceptable rule:

```python
can_advance = (
    current.remaining_length_m <= transition_window_m
    and next_candidate.progress_local_m <= transition_window_m
)
```

The window should be based on physical step size, not an arbitrary fixed `2 m` tied to one exact rollout.

---

### Task 3: Preserve closed-loop continuity inside loiter segments

**Files:**
- Modify: `source/flapping_bot/flapping_bot/path_tracking/path_manager.py`
- Test: `tests/test_path_manager.py`

**Step 1: Keep the existing good arc-entry fix**

Retain the earlier improvement that prevents a loiter from snapping back to `0 rad` at entry when the raw geometry already indicates small forward motion.

**Step 2: Make local arc selection relative to the active segment**

For a closed loop, choose the candidate angle that is continuous with the active segment’s local progress, not just the one that is closest to global scalar progress after all segments are mixed together.

This means:

- local loiter continuity is decided inside the segment
- segment transitions are decided outside the segment by the segment-aware state machine

That separation is the key design correction.

**Step 3: Remove or demote the brittle forward-boundary gate**

`_FORWARD_PROGRESS_BOUNDARY_GATE_M = 2.0` is the mechanism that is currently failing. Do not rely on it as the primary protection.

After segment-aware continuity exists, either:

- remove the forward-boundary gate entirely, or
- keep it only as a secondary heuristic with no authority to override segment continuity

---

### Task 4: Make benchmark success criteria reject fake completion

**Files:**
- Modify: `scripts/flapping_px4/run_random_path_tracking_teacher_suite.py`
- Modify: `scripts/flapping_px4/fly_path_mission.py` if needed for exported metadata
- Test: `tests/test_fly_path_mission.py` or a new lightweight regression script

**Step 1: Promote post-check logic into normal validation**

Add automated failure criteria such as:

- `max_single_step_progress_jump_m <= 5.0`
- no jump larger than `transition_window_m + tolerance`
- no impossible early jump before the path length could physically be traversed

**Step 2: Ensure summary metrics surface geometry failures**

`completed_path = true` must no longer be treated as sufficient evidence.

At suite level, report:

- `jump_free_completion_rate`
- seeds with suspicious progress discontinuities
- worst offending segment topology

This prevents benchmark summaries from hiding path-manager bugs.

---

### Task 5: Validate in stages before any controller retuning

**Files:**
- Modify: none
- Use: `scripts/flapping_px4/fly_path_mission.py`
- Use: `scripts/flapping_px4/run_random_path_tracking_teacher_suite.py`

**Step 1: Re-run targeted seeds first**

Focus on the exact failing seeds:

- `1`
- `5`
- `20`
- `30`
- `40`
- `44`

Acceptance:

- no single-step progress jump `> 5 m`
- no segment skipping
- progress stays continuous through overlapping boundaries

**Step 2: Re-run the 50-case suite**

Only after the targeted seeds are clean:

```bash
export TERM=xterm
./isaaclab.sh -p scripts/flapping_px4/run_random_path_tracking_teacher_suite.py --episodes 50 ...
```

Acceptance:

- `completion_rate` remains high
- `jump_free_completion_rate == completion_rate`
- `postcheck.json` shows `suspicious_jump_cases_gt_5m = []`

**Step 3: Only then evaluate controller quality**

If geometry is clean and some missions still fail, then the remaining issue is controller-tracking quality rather than fake path completion. Only at that point should TECS / lateral tuning be revisited.

---

### Task 6: Commit in small checkpoints

**Files:**
- Commit after each completed task

Suggested commits:

```bash
git commit -m "test: capture remaining loiter teleport regressions"
git commit -m "fix: make path manager segment-aware across overlaps"
git commit -m "test: fail teacher benchmark on progress teleports"
```

---

## Why this plan, not another threshold tweak

Raising `_FORWARD_PROGRESS_BOUNDARY_GATE_M` from `2.0` to `3.0` might hide the currently observed jumps, but it would still be a fragile speed- and timestep-dependent heuristic. The real ambiguity is topological: multiple future segments share the same point and nearly the same tangent, so a distance-only global search is underconstrained. The correct fix is to encode path continuity explicitly.
