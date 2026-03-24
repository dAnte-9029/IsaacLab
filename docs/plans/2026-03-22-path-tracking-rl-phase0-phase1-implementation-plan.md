# Path-Tracking RL Phase 0/1 Stabilization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Turn the current teacher-guided generic path-tracking PPO into a trustworthy Phase 0/1 pipeline by fixing teacher-off evaluation, penalizing no-progress policies, and adding a reproducible no-wind primitive curriculum before returning to harder mixed missions.

**Architecture:** Keep `FlappingBotPathTrackingEnv` as the single environment family. Tighten the teacher-off evaluation contract in `scripts/flapping_rl`, add no-progress detection and metrics inside `path_tracking_env.py`, and introduce explicit primitive-stage task configs that reuse the same environment with a simpler mission distribution. Use the existing `train.py`, `train_and_watch.py`, and `watch_and_eval.py` stack; do not add a second RL pipeline.

**Tech Stack:** IsaacLab direct RL envs, Isaac Sim, RSL-RL PPO, current PX4-like teacher envelope, `train_and_watch.py`, `watch_and_eval.py`, pytest.

---

## Scope

This plan only covers the next RL step:

- teacher-off evaluation that does not over-credit low-error / zero-progress policies
- no-progress / stall detection in the path-tracking RL env
- no-wind primitive curriculum for single-segment `straight` / `turn` / `loiter`
- weak-teacher PPO followed by pure-RL continuation on that primitive curriculum

This plan explicitly does **not** yet cover:

- multi-segment mixed-mission curriculum
- wind curriculum during RL training
- estimated-state / noisy-observation RL
- behavior cloning or offline distillation

---

### Task 1: Lock the teacher-off evaluation contract

**Files:**
- Modify: `scripts/flapping_rl/path_tracking_eval_common.py`
- Modify: `scripts/flapping_rl/checkpoint_selection.py`
- Modify: `scripts/flapping_rl/eval_path_tracking_checkpoint.py`
- Modify: `scripts/flapping_rl/watch_and_eval.py`
- Test: `tests/test_checkpoint_selection.py`
- Test: `tests/test_path_tracking_eval_common.py`
- Test: `tests/test_watch_and_eval.py`

**Step 1: Write the failing tests**

Extend the existing path-tracking eval tests so they require:

- a path-tracking row with tiny tracking error but near-zero progress to be marked as a failed success gate
- checkpoint selection to preserve “best available” ordering while also surfacing whether the selected checkpoint actually passed the success gate
- watcher summary rows to persist the new gate field(s)

Example tests:

```python
def test_path_tracking_success_gate_rejects_near_zero_progress_timeout() -> None:
    row = {
        "completion_rate": 0.0,
        "mean_final_progress_ratio": 0.004,
        "termination_rate": 0.0,
        "timeout_rate": 1.0,
    }
    assert not row_meets_path_tracking_success_gate(row)


def test_select_best_checkpoint_row_keeps_best_available_but_marks_gate_failure(tmp_path: Path) -> None:
    ...
    best = select_best_checkpoint_row(summary_csv)
    assert best["checkpoint"].endswith("model_139.pt")
    assert best["success_gate_passed"] == "0"
```

**Step 2: Run tests to verify they fail**

Run:

```bash
./isaaclab.sh -p -m pytest \
  tests/test_checkpoint_selection.py \
  tests/test_path_tracking_eval_common.py \
  tests/test_watch_and_eval.py -q
```

Expected: FAIL because the current eval path does not expose an explicit success gate for path-tracking checkpoints.

**Step 3: Implement the minimal evaluation changes**

In `path_tracking_eval_common.py`:

- add a small helper such as `row_meets_path_tracking_success_gate(...)`
- define the first gate around:
  - `completion_rate`
  - `mean_final_progress_ratio`
  - `termination_rate`
- keep the current scalar score, but separate “ranking” from “success”

In `checkpoint_selection.py` and `watch_and_eval.py`:

- propagate `success_gate_passed`
- keep selecting the best available checkpoint even when none pass
- avoid writing misleading “best” artifacts that look like a successful policy when the gate is not met

In `eval_path_tracking_checkpoint.py`:

- include the new gate field in the aggregated JSON / CSV rows

**Step 4: Run tests to verify they pass**

Run:

```bash
./isaaclab.sh -p -m pytest \
  tests/test_checkpoint_selection.py \
  tests/test_path_tracking_eval_common.py \
  tests/test_watch_and_eval.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add \
  scripts/flapping_rl/path_tracking_eval_common.py \
  scripts/flapping_rl/checkpoint_selection.py \
  scripts/flapping_rl/eval_path_tracking_checkpoint.py \
  scripts/flapping_rl/watch_and_eval.py \
  tests/test_checkpoint_selection.py \
  tests/test_path_tracking_eval_common.py \
  tests/test_watch_and_eval.py
git commit -m "fix: gate path-tracking rl checkpoint success"
```

---

### Task 2: Add no-progress termination and metrics

**Files:**
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py`
- Modify: `scripts/flapping_rl/path_tracking_eval_common.py`
- Test: `tests/test_path_tracking_rewards.py`
- Test: `tests/test_path_tracking_env_contract.py`
- Test: `tests/test_path_tracking_eval_common.py`

**Step 1: Write the failing tests**

Add tests that require:

- a policy that makes no path progress after warmup to receive a worse reward than one that progresses
- the env config to expose explicit no-progress settings
- aggregated eval rows to expose `stall_rate` / `no_progress_rate`

Example tests:

```python
def test_tracking_reward_penalizes_no_progress() -> None:
    stalled = _compute_tracking_reward(..., delta_s=torch.tensor([0.0]), stalled=torch.tensor([True]))
    moving = _compute_tracking_reward(..., delta_s=torch.tensor([0.2]), stalled=torch.tensor([False]))
    assert moving.item() > stalled.item()


def test_aggregate_case_row_reports_stall_rate() -> None:
    ...
    assert math.isclose(row["stall_rate"], 0.5)
```

**Step 2: Run tests to verify they fail**

Run:

```bash
./isaaclab.sh -p -m pytest \
  tests/test_path_tracking_rewards.py \
  tests/test_path_tracking_env_contract.py \
  tests/test_path_tracking_eval_common.py -q
```

Expected: FAIL because the current path-tracking env does not expose explicit no-progress termination or stall metrics.

**Step 3: Implement the minimal no-progress logic**

In `path_tracking_env.py`:

- add config fields such as:
  - `no_progress_warmup_s`
  - `no_progress_window_s`
  - `no_progress_min_delta_s_m`
  - `no_progress_penalty`
- keep a per-env progress reference after warmup
- mark an env as stalled when progress over the window stays below the threshold
- terminate stalled envs
- store a per-step / per-episode stall flag for evaluation

In `_compute_tracking_reward(...)`:

- add a small explicit stalled penalty
- keep the current structure centered on `delta_s` plus tracking penalties

In `path_tracking_eval_common.py`:

- aggregate `stall_rate` beside `completion_rate`, `termination_rate`, and `timeout_rate`

**Step 4: Run tests to verify they pass**

Run:

```bash
./isaaclab.sh -p -m pytest \
  tests/test_path_tracking_rewards.py \
  tests/test_path_tracking_env_contract.py \
  tests/test_path_tracking_eval_common.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add \
  source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py \
  scripts/flapping_rl/path_tracking_eval_common.py \
  tests/test_path_tracking_rewards.py \
  tests/test_path_tracking_env_contract.py \
  tests/test_path_tracking_eval_common.py
git commit -m "fix: terminate stalled path-tracking rl episodes"
```

---

### Task 3: Add a primitive-stage no-wind curriculum and task IDs

**Files:**
- Modify: `scripts/flapping_rl/eval_suites.py`
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py`
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/__init__.py`
- Modify: `source/flapping_bot/flapping_bot/__init__.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/direct/flapping_bot/__init__.py`
- Test: `tests/test_flapping_task_registration.py`
- Test: `tests/test_path_tracking_env_contract.py`
- Test: `tests/test_watch_and_eval.py`

**Step 1: Write the failing tests**

Add tests that require:

- a dedicated primitive eval suite such as `path_tracking_truth_primitives_nowind_v1`
- new task registrations for:
  - `Isaac-FlappingBot-PathTracking-DeLaurier-PrimitiveWeakTeacherRL-Direct-v0`
  - `Isaac-FlappingBot-PathTracking-DeLaurier-PrimitivePureRL-Direct-v0`
- primitive tasks to use:
  - `mission_num_segments_min = mission_num_segments_max = 1`
  - `straight` / `turn` / `loiter` enabled
  - climb on straight disabled for the first curriculum stage

Example tests:

```python
def test_primitive_path_tracking_suite_exists() -> None:
    assert "path_tracking_truth_primitives_nowind_v1" in get_eval_suite_choices()


def test_primitive_path_tracking_task_ids_registered() -> None:
    assert "Isaac-FlappingBot-PathTracking-DeLaurier-PrimitiveWeakTeacherRL-Direct-v0" in registered_ids
    assert "Isaac-FlappingBot-PathTracking-DeLaurier-PrimitivePureRL-Direct-v0" in registered_ids
```

**Step 2: Run tests to verify they fail**

Run:

```bash
./isaaclab.sh -p -m pytest \
  tests/test_flapping_task_registration.py \
  tests/test_path_tracking_env_contract.py \
  tests/test_watch_and_eval.py -q
```

Expected: FAIL because the primitive curriculum tasks and suite do not exist yet.

**Step 3: Implement the minimal curriculum additions**

In `path_tracking_env.py`:

- add `FlappingBotPathTrackingPrimitiveWeakTeacherRLEnvCfg`
- add `FlappingBotPathTrackingPrimitivePureRLEnvCfg`
- keep observation / action space unchanged
- keep wind disabled for this stage

In `eval_suites.py`:

- add `path_tracking_truth_primitives_nowind_v1`
- include one fixed case each for:
  - `straight_primitive_nowind`
  - `turn_primitive_nowind`
  - `loiter_primitive_nowind`

In the task registration files:

- export the new config classes
- register the two new task ids

**Step 4: Run tests to verify they pass**

Run:

```bash
./isaaclab.sh -p -m pytest \
  tests/test_flapping_task_registration.py \
  tests/test_path_tracking_env_contract.py \
  tests/test_watch_and_eval.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add \
  scripts/flapping_rl/eval_suites.py \
  source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py \
  source/flapping_bot/flapping_bot/direct/flapping_bot/__init__.py \
  source/flapping_bot/flapping_bot/__init__.py \
  source/isaaclab_tasks/isaaclab_tasks/direct/flapping_bot/__init__.py \
  tests/test_flapping_task_registration.py \
  tests/test_path_tracking_env_contract.py \
  tests/test_watch_and_eval.py
git commit -m "feat: add primitive path-tracking rl curriculum tasks"
```

---

### Task 4: Smoke-verify the primitive weak-teacher stage

**Files:**
- No code changes expected
- Output: `logs/rsl_rl/flapping_bot_path_tracking/...`

**Step 1: Run targeted unit tests first**

Run:

```bash
./isaaclab.sh -p -m pytest \
  tests/test_checkpoint_selection.py \
  tests/test_path_tracking_eval_common.py \
  tests/test_path_tracking_rewards.py \
  tests/test_path_tracking_env_contract.py \
  tests/test_flapping_task_registration.py \
  tests/test_watch_and_eval.py -q
```

Expected: PASS

**Step 2: Run a 1-iteration training smoke**

Run:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-FlappingBot-PathTracking-DeLaurier-PrimitiveWeakTeacherRL-Direct-v0 \
  --device cuda:0 \
  --num_envs 64 \
  --max_iterations 1 \
  --headless \
  --run_name pt_primitive_weakteacher_smoke \
  agent.save_interval=1
```

Expected:

- Isaac starts cleanly
- one PPO iteration completes
- a `model_0.pt` checkpoint is written

**Step 3: Run a watcher smoke on the primitive suite**

Run:

```bash
./isaaclab.sh -p scripts/flapping_rl/train_and_watch.py \
  --task Isaac-FlappingBot-PathTracking-DeLaurier-PrimitiveWeakTeacherRL-Direct-v0 \
  --run-name pt_primitive_weakteacher_watch_smoke \
  --train-device cuda:0 \
  --eval-device cuda:1 \
  --num-envs 64 \
  --max-iterations 20 \
  --save-interval 20 \
  --episodes 2 \
  --poll-s 30 \
  --eval-suite path_tracking_truth_primitives_nowind_v1 \
  --headless
```

Expected:

- eval rows appear for the primitive suite
- `best_checkpoint.json` is written
- `success_gate_passed` is visible in the watcher artifacts

**Step 4: Inspect the primitive weak-teacher exit criteria**

Require all of the following before moving to pure RL:

- primitive suite `completion_rate >= 0.70`
- primitive suite `mean_final_progress_ratio >= 0.85`
- primitive suite `termination_rate <= 0.20`
- primitive suite `success_gate_passed == 1`

If these fail, stop and tune reward / no-progress thresholds before starting pure-RL continuation.

**Step 5: Commit**

No commit if only logs changed.

---

### Task 5: Run pure-RL continuation from the primitive weak-teacher checkpoint

**Files:**
- No code changes expected
- Output: `logs/rsl_rl/flapping_bot_path_tracking/...`

**Step 1: Resume into the primitive pure-RL task**

Run:

```bash
./isaaclab.sh -p scripts/flapping_rl/train_and_watch.py \
  --task Isaac-FlappingBot-PathTracking-DeLaurier-PrimitivePureRL-Direct-v0 \
  --run-name pt_primitive_purerl_resume \
  --train-device cuda:0 \
  --eval-device cuda:1 \
  --num-envs 64 \
  --max-iterations 40 \
  --save-interval 10 \
  --episodes 2 \
  --poll-s 30 \
  --eval-suite path_tracking_truth_primitives_nowind_v1 \
  --resume \
  --load_run <best weak-teacher run> \
  --checkpoint <best weak-teacher checkpoint> \
  --headless
```

Expected: training runs with teacher guidance disabled from step 0.

**Step 2: Verify the runtime metrics actually show teacher-off**

Check the training log for:

- `Teacher/enabled: 0.0000` or missing teacher activity metrics
- no eval-time teacher leakage

**Step 3: Evaluate the pure-RL primitive checkpoint**

Require all of the following:

- primitive suite `completion_rate >= 0.60`
- primitive suite `mean_final_progress_ratio >= 0.80`
- primitive suite `termination_rate <= 0.25`
- primitive suite `success_gate_passed == 1`

**Step 4: Record the handoff result**

Write the winning checkpoint path and primitive-suite summary into the run notes or final implementation summary. This checkpoint becomes the starting point for the next plan: mixed multi-segment no-wind missions.

**Step 5: Commit**

No commit if only logs changed.
