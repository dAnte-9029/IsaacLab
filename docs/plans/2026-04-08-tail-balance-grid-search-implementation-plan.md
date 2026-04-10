# Tail Balance Grid Search Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a reproducible grid-search runner that tunes `tail_fixed_horizontal_effectiveness` and `tail_elevon_effectiveness` on the non-RL DeLaurier path-tracking stack, enforces strict `level_straight` regression guards, and reports the best candidate plus the full response surface.

**Architecture:** Add a new orchestration script that launches `fly_path_mission.py` for canonical and random missions, parses each rollout's `summary.json` and `trajectory_env0.csv`, aggregates candidate-level metrics, applies hard constraints, scores feasible candidates, and writes machine-readable plus human-readable artifacts. Reuse existing rollout paths and existing trajectory fields instead of creating a second simulation stack.

**Tech Stack:** Python 3.11, argparse, subprocess, csv/json/pathlib, existing IsaacLab `./isaaclab.sh -p` rollout wrappers, pytest

---

### Task 1: Lock the search contract with unit tests

**Files:**
- Create: `tests/test_tail_balance_grid_search.py`
- Check: `scripts/flapping_px4/run_random_path_tracking_teacher_suite.py`
- Check: `scripts/flapping_px4/run_path_tracking_baseline_suite.py`

**Step 1: Write the failing tests**

Add unit tests that define the expected pure helper behavior for the new search runner:

```python
def test_build_stage1_grid_has_25_pairs() -> None:
    pairs = build_stage1_grid()
    assert len(pairs) == 25
    assert pairs[0] == (0.60, 1.00)
    assert pairs[-1] == (1.00, 1.80)


def test_candidate_is_infeasible_when_straight_height_guard_breaks() -> None:
    baseline = {"straight_height_error": 1.0, "random_completion_rate": 1.0, "straight_completed": True}
    candidate = {"straight_height_error": 1.06, "random_completion_rate": 1.0, "straight_completed": True}
    result = evaluate_hard_constraints(candidate, baseline)
    assert result.feasible is False
    assert "straight_height_guard" in result.failed_constraints


def test_score_candidate_weights_turn_loiter_and_random_metrics() -> None:
    score = score_candidate(...)
    assert score == pytest.approx(expected_score)
```

Also add a test that trajectory-derived saturation metrics are computed from CSV rows using:
- `freq_hz >= 4.9`
- `action_elevon_pitch <= -0.98`

**Step 2: Run the new tests and confirm they fail**

Run:

```bash
python -m pytest tests/test_tail_balance_grid_search.py -q
```

Expected:
- import failure because the search runner does not exist yet

**Step 3: Commit**

```bash
git add tests/test_tail_balance_grid_search.py
git commit -m "test: lock tail balance search contract"
```

### Task 2: Add the search runner skeleton and pure helpers

**Files:**
- Create: `scripts/flapping_px4/run_tail_balance_grid_search.py`
- Test: `tests/test_tail_balance_grid_search.py`

**Step 1: Write the minimal runner structure**

Add:
- parser builder
- stage-1 grid builder
- local-refinement grid builder
- hard-constraint evaluator
- score function
- small data containers for rollout and candidate metrics

Keep these helpers pure so they can be tested without IsaacSim.

**Step 2: Run the unit tests**

Run:

```bash
python -m pytest tests/test_tail_balance_grid_search.py -q
```

Expected:
- helper tests pass
- integration-style tests for subprocess/output handling still fail if not implemented yet

**Step 3: Commit**

```bash
git add scripts/flapping_px4/run_tail_balance_grid_search.py tests/test_tail_balance_grid_search.py
git commit -m "feat: add tail balance search helpers"
```

### Task 3: Implement rollout command construction and per-run parsing

**Files:**
- Modify: `scripts/flapping_px4/run_tail_balance_grid_search.py`
- Test: `tests/test_tail_balance_grid_search.py`
- Check: `scripts/flapping_px4/fly_path_mission.py`

**Step 1: Build rollout commands**

Implement helpers that create canonical and random rollout commands using:
- `./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py`
- controller-only DeLaurier task
- tail parameter overrides
- `--out_dir`
- `--headless`

Canonical cases:
- `level_straight`, `2200`
- `level_turn`, `2400`
- `level_loiter`, `2800`

Random seeds:
- `11, 23, 37, 53, 71, 89`
- `2600` steps
- `mission_mode=random`

**Step 2: Parse rollout outputs**

Implement helpers that load:
- `summary.json`
- `trajectory_env0.csv`

From those files compute:
- post-warmup height metrics
- completion/progress metrics
- `freq_sat_frac`
- `pitch_sat_frac`

**Step 3: Add tests for command shape and saturation parsing**

Example:

```python
def test_build_random_command_includes_tail_overrides() -> None:
    cmd = build_random_rollout_command(...)
    assert "--tail_fixed_horizontal_effectiveness" in cmd
    assert "--tail_elevon_effectiveness" in cmd
    assert "--mission_seed" in cmd
```

**Step 4: Run tests**

Run:

```bash
python -m pytest tests/test_tail_balance_grid_search.py -q
```

Expected:
- all pure/parser tests pass

**Step 5: Commit**

```bash
git add scripts/flapping_px4/run_tail_balance_grid_search.py tests/test_tail_balance_grid_search.py
git commit -m "feat: parse rollout metrics for tail balance search"
```

### Task 4: Implement candidate aggregation, hard constraints, and artifacts

**Files:**
- Modify: `scripts/flapping_px4/run_tail_balance_grid_search.py`
- Test: `tests/test_tail_balance_grid_search.py`

**Step 1: Aggregate rollout metrics by candidate**

For each parameter pair:
- run canonical rollouts
- run seeded random rollouts
- compute candidate-level metrics
- compare against baseline
- mark feasible/infeasible
- compute score for feasible candidates

**Step 2: Write search artifacts**

Write:
- `manifest.json`
- `per_run_metrics.csv`
- `leaderboard.csv`
- `response_surface.csv`
- `summary.json`
- `summary.md`

`summary.md` should include:
- baseline parameter row
- best feasible candidate
- next two candidates
- constraint failures by count

**Step 3: Add tests for aggregation and ranking**

Add a test like:

```python
def test_rank_candidates_sorts_feasible_before_infeasible() -> None:
    ranked = rank_candidates([...])
    assert ranked[0]["feasible"] is True
    assert ranked[-1]["feasible"] is False
```

**Step 4: Run tests**

Run:

```bash
python -m pytest tests/test_tail_balance_grid_search.py -q
```

Expected:
- green

**Step 5: Commit**

```bash
git add scripts/flapping_px4/run_tail_balance_grid_search.py tests/test_tail_balance_grid_search.py
git commit -m "feat: add tail balance search scoring and reports"
```

### Task 5: Add lightweight CLI coverage and repo-level compile checks

**Files:**
- Modify: `tests/test_tail_balance_grid_search.py`
- Check: `tests/test_fly_path_mission.py`

**Step 1: Add parser/default tests**

Cover:
- default seeds
- default coarse grid values
- default strict straight guard
- default output filenames

**Step 2: Run Python compile checks**

Run:

```bash
python -m py_compile scripts/flapping_px4/run_tail_balance_grid_search.py
python -m py_compile scripts/flapping_px4/fly_path_mission.py
```

Expected:
- both succeed

**Step 3: Run unit tests**

Run:

```bash
python -m pytest tests/test_tail_balance_grid_search.py tests/test_fly_path_mission.py -q
```

Expected:
- green

**Step 4: Commit**

```bash
git add tests/test_tail_balance_grid_search.py
git commit -m "test: cover tail balance search cli defaults"
```

### Task 6: Run the baseline search dry-run and one real baseline candidate

**Files:**
- Run only: `scripts/flapping_px4/run_tail_balance_grid_search.py`
- Inspect: `logs/flapping_px4/tail_balance_grid_search/...`

**Step 1: Dry-run the command graph**

Run:

```bash
./isaaclab.sh -p scripts/flapping_px4/run_tail_balance_grid_search.py --dry_run --headless
```

Expected:
- prints canonical and random rollout commands
- writes no invalid manifest structure

**Step 2: Run a reduced real check**

Run:

```bash
./isaaclab.sh -p scripts/flapping_px4/run_tail_balance_grid_search.py \
  --stage1_fixed_values 1.0 \
  --stage1_elevon_values 1.0 \
  --top_k_refine 0 \
  --random_seeds 11 23 \
  --out_root logs/flapping_px4/tail_balance_grid_search_smoke \
  --headless
```

Expected:
- baseline candidate completes
- `leaderboard.csv` and `summary.json` exist
- candidate row records feasibility and score

**Step 3: Commit**

```bash
git add scripts/flapping_px4/run_tail_balance_grid_search.py tests/test_tail_balance_grid_search.py
git commit -m "feat: add executable tail balance search runner"
```

### Task 7: Run the full target verification

**Files:**
- Run only: `scripts/flapping_px4/run_tail_balance_grid_search.py`
- Inspect: `logs/flapping_px4/tail_balance_grid_search/...`

**Step 1: Compile**

Run:

```bash
python -m py_compile scripts/flapping_px4/run_tail_balance_grid_search.py
python -m py_compile scripts/flapping_px4/fly_path_mission.py
```

**Step 2: Unit tests**

Run:

```bash
python -m pytest tests/test_tail_balance_grid_search.py tests/test_fly_path_mission.py -q
```

**Step 3: Full search**

Run:

```bash
./isaaclab.sh -p scripts/flapping_px4/run_tail_balance_grid_search.py \
  --out_root logs/flapping_px4/tail_balance_grid_search \
  --headless
```

Expected:
- baseline is recorded
- stage 1 coarse grid completes
- stage 2 refinement completes
- `leaderboard.csv`, `per_run_metrics.csv`, `response_surface.csv`, `summary.json`, and `summary.md` are written

**Step 4: Manual acceptance**

Confirm:
- `level_straight` hard guard is enforced
- feasible best candidate improves `turn/loiter`
- random seeded mission completion is not materially degraded

**Step 5: Commit**

```bash
git add docs/plans/2026-04-08-tail-balance-grid-search-design.md \
        docs/plans/2026-04-08-tail-balance-grid-search-implementation-plan.md \
        scripts/flapping_px4/run_tail_balance_grid_search.py \
        tests/test_tail_balance_grid_search.py
git commit -m "feat: add tail balance grid search workflow"
```
