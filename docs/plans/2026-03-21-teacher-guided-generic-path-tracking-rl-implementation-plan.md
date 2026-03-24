# Teacher-Guided Generic Path-Tracking RL Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the first end-to-end training pipeline for teacher-guided generic path-tracking RL, then continue to a teacher-off pure-RL checkpoint on no-wind truth-state missions.

**Architecture:** Reuse the existing `FlappingBotPathTrackingEnv` as the single environment family. Add RL-specific observation, reward, and teacher-schedule controls there instead of creating a separate environment. Keep the PX4-like path-tracking teacher as an optional training prior, not as the runtime controller. Reuse the existing watcher/evaluation pipeline so every checkpoint is scored on a standard path-tracking suite while training runs.

**Tech Stack:** IsaacLab direct RL envs, Isaac Sim, RSL-RL PPO, current PX4-like path-tracking teacher, path manager mission generator, pytest, `train_and_watch.py`, `watch_and_eval.py`.

---

## Scope

This plan covers the first practical RL milestone only:

- truth-state observations
- no-wind missions
- teacher-guided PPO
- teacher-off continuation
- mixed path missions through one policy

This plan does **not** yet include:

- estimated-state / sensor-noise training
- wind curriculum in the training distribution
- BC warm start unless PPO startup is clearly too slow

---

### Task 1: Freeze the first RL target and evaluation contract

**Files:**
- Modify: `scripts/flapping_rl/eval_suites.py`
- Modify: `scripts/flapping_rl/watch_and_eval.py`
- Test: `tests/test_eval_suites.py`
- Test: `tests/test_watch_and_eval.py`

**Step 1: Write the failing tests**

Add tests that require a new no-wind generic path-tracking suite, for example:

```python
def test_path_tracking_truth_suite_exists():
    assert "path_tracking_truth_nowind_v1" in get_eval_suite_choices()


def test_path_tracking_truth_suite_contains_mixed_missions():
    cases = build_eval_cases("path_tracking_truth_nowind_v1")
    case_names = {case["case"] for case in cases}
    assert "straight_nowind" in case_names
    assert "loiter_nowind" in case_names
    assert "random_mission_nowind" in case_names
```

**Step 2: Run tests to verify they fail**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_eval_suites.py tests/test_watch_and_eval.py -q
```

Expected: FAIL because the new suite is not implemented yet.

**Step 3: Implement the minimal suite changes**

Add:

- one new eval-suite name for no-wind truth-state path tracking
- a small fixed set of straight / loiter / random-mission cases
- watcher support so this suite can be selected from `train_and_watch.py`

**Step 4: Run tests to verify they pass**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_eval_suites.py tests/test_watch_and_eval.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add scripts/flapping_rl/eval_suites.py scripts/flapping_rl/watch_and_eval.py tests/test_eval_suites.py tests/test_watch_and_eval.py
git commit -m "feat: add no-wind path-tracking eval suite"
```

---

### Task 2: Add RL observation support for generic path tracking

**Files:**
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py`
- Test: `tests/test_path_tracking_observations.py`
- Test: `tests/test_path_tracking_env_contract.py`

**Step 1: Write the failing tests**

Add tests that require the RL observation to include:

- current path geometry
- 5 preview points in body frame
- previous action

Example:

```python
def test_path_tracking_observation_contains_five_preview_points():
    obs = _build_preview_observation(preview_points_body_xyz=torch.zeros(2, 5, 3))
    assert obs.shape[-1] == 15
```

Add a contract test asserting preview length stays fixed at 5 in the first version.

**Step 2: Run tests to verify they fail**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_path_tracking_observations.py tests/test_path_tracking_env_contract.py -q
```

Expected: FAIL because the exact RL observation contract is not yet implemented.

**Step 3: Implement the minimal observation changes**

In `path_tracking_env.py`:

- ensure current local path geometry is always available
- ensure preview points are included in the policy observation
- ensure previous action is included or stacked consistently
- keep observation shape deterministic

Do not change action space.

**Step 4: Run tests to verify they pass**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_path_tracking_observations.py tests/test_path_tracking_env_contract.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py tests/test_path_tracking_observations.py tests/test_path_tracking_env_contract.py
git commit -m "feat: expose generic path-tracking observations for rl"
```

---

### Task 3: Add the first RL reward for generic path tracking

**Files:**
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py`
- Test: `tests/test_path_tracking_rewards.py`

**Step 1: Write the failing tests**

Add tests that require:

- better progress gives higher reward
- larger lateral error lowers reward
- larger height error lowers reward
- large action change lowers reward

Example:

```python
def test_tracking_reward_prefers_progress_with_small_errors():
    better = _compute_tracking_reward(...)
    worse = _compute_tracking_reward(...)
    assert better > worse
```

**Step 2: Run tests to verify they fail**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_path_tracking_rewards.py -q
```

Expected: FAIL because the first RL reward shape is incomplete or inconsistent.

**Step 3: Implement the minimal reward**

Add reward terms for:

- `delta_s`
- lateral error
- height error
- alignment error
- action rate / action magnitude
- termination penalty

Do **not** add hard cruise-speed tracking as a primary objective.

**Step 4: Run tests to verify they pass**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_path_tracking_rewards.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py tests/test_path_tracking_rewards.py
git commit -m "feat: add generic path-tracking rl reward"
```

---

### Task 4: Add explicit teacher schedule stages for path-tracking RL

**Files:**
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py`
- Modify: `source/flapping_bot/flapping_bot/px4_like/rl_training_utils.py`
- Test: `tests/test_teacher_schedule.py`
- Test: `tests/test_rl_teacher_guidance.py`

**Step 1: Write the failing tests**

Add tests for a staged schedule such as:

- strong teacher at step 0
- weaker teacher at configured milestone
- teacher fully disabled at configured step

Example:

```python
def test_path_tracking_teacher_schedule_reaches_zero_guidance():
    delta = piecewise_linear_anneal(...)
    assert delta == 2.0
```

**Step 2: Run tests to verify they fail**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_teacher_schedule.py tests/test_rl_teacher_guidance.py -q
```

Expected: FAIL because the new staged schedule is not wired for the generic path-tracking task.

**Step 3: Implement the minimal schedule**

Add named or documented stages:

- strong teacher
- weak teacher
- teacher off

Expose them through env config so training commands can switch behavior without editing code.

**Step 4: Run tests to verify they pass**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_teacher_schedule.py tests/test_rl_teacher_guidance.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py source/flapping_bot/flapping_bot/px4_like/rl_training_utils.py tests/test_teacher_schedule.py tests/test_rl_teacher_guidance.py
git commit -m "feat: add staged teacher schedule for path-tracking rl"
```

---

### Task 5: Register the first trainable path-tracking RL task config

**Files:**
- Modify: `source/isaaclab_tasks/isaaclab_tasks/direct/flapping_bot/__init__.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/direct/flapping_bot/agents/` (existing PPO cfg or new cfg file)
- Test: `tests/test_flapping_task_registration.py`
- Test: `tests/test_path_tracking_env_contract.py`

**Step 1: Write the failing tests**

Add tests asserting a new task ID exists, for example:

```python
def test_path_tracking_teacher_rl_task_is_registered():
    assert "Isaac-FlappingBot-PathTracking-DeLaurier-TeacherRL-Direct-v0" in registered
```

**Step 2: Run tests to verify they fail**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_flapping_task_registration.py tests/test_path_tracking_env_contract.py -q
```

Expected: FAIL because the new RL task or agent cfg is not registered yet.

**Step 3: Implement the minimal task registration**

Add:

- one teacher-guided generic path-tracking RL task
- one weak-teacher or teacher-off continuation task if needed
- PPO defaults consistent with no-wind truth-state training

**Step 4: Run tests to verify they pass**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_flapping_task_registration.py tests/test_path_tracking_env_contract.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add source/isaaclab_tasks/isaaclab_tasks/direct/flapping_bot/__init__.py source/isaaclab_tasks/isaaclab_tasks/direct/flapping_bot/agents tests/test_flapping_task_registration.py tests/test_path_tracking_env_contract.py
git commit -m "feat: register generic path-tracking rl task"
```

---

### Task 6: Wire the train-and-watch loop for the new RL task

**Files:**
- Modify: `scripts/flapping_rl/train_and_watch.py`
- Modify: `scripts/flapping_rl/watch_and_eval.py`
- Create or modify: `tests/test_train_and_watch.py`
- Create or modify: `tests/test_watch_and_eval.py`

**Step 1: Write the failing tests**

Require:

- selecting the new eval suite from `train_and_watch.py`
- writing `suite` rows for the new path-tracking RL task

**Step 2: Run tests to verify they fail**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_train_and_watch.py tests/test_watch_and_eval.py -q
```

Expected: FAIL until the new task/eval suite flow is wired.

**Step 3: Implement the minimal watcher support**

Make sure:

- the new task is accepted cleanly by `train_and_watch.py`
- watcher evaluation picks the correct path-tracking eval suite
- best-checkpoint artifacts are still refreshed

**Step 4: Run tests to verify they pass**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_train_and_watch.py tests/test_watch_and_eval.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add scripts/flapping_rl/train_and_watch.py scripts/flapping_rl/watch_and_eval.py tests/test_train_and_watch.py tests/test_watch_and_eval.py
git commit -m "feat: support generic path-tracking rl train-and-watch"
```

---

### Task 7: Run the first teacher-guided PPO training job

**Files:**
- Validate outputs under: `logs/rsl_rl/...`
- Validate summaries under: `<run_dir>/eval/summary.csv`
- Optional note: `docs/flapping_rl/`

**Step 1: Launch the first teacher-guided run**

Run something like:

```bash
./isaaclab.sh -p scripts/flapping_rl/train_and_watch.py \
  --task Isaac-FlappingBot-PathTracking-DeLaurier-TeacherRL-Direct-v0 \
  --run-name path_tracking_truth_v1 \
  --train-device cuda:0 \
  --eval-device cuda:1 \
  --num-envs 512 \
  --max-iterations 2000 \
  --save-interval 100 \
  --episodes 5 \
  --poll-s 120 \
  --eval-suite path_tracking_truth_nowind_v1 \
  --headless
```

**Step 2: Verify watcher output is being populated**

Check:

```bash
tail -n 20 <run_dir>/eval/summary.csv
```

Expected: rows appear for each new checkpoint and suite summary.

**Step 3: Pick the first usable checkpoint**

Use:

```bash
./isaaclab.sh -p scripts/flapping_rl/select_best_checkpoint.py --log_dir <run_dir>
```

**Step 4: Run a one-shot visual or headless evaluation**

Run:

```bash
./isaaclab.sh -p scripts/flapping_rl/eval_path_tracking_checkpoint.py \
  --task Isaac-FlappingBot-PathTracking-DeLaurier-TeacherRL-Direct-v0 \
  --checkpoint <best_checkpoint> \
  --eval-suite path_tracking_truth_nowind_v1 \
  --headless
```

**Step 5: Record the baseline metrics**

Save:

- best checkpoint path
- completion rate
- lateral / height / alignment errors
- whether the policy is still clearly teacher-dependent

**Step 6: Commit**

Commit only scripts/config/docs changed during this task, not generated logs.

---

### Task 8: Continue from best checkpoint with weaker teacher, then teacher off

**Files:**
- Validate task cfgs
- Validate training logs
- Optional doc update under `docs/flapping_rl/`

**Step 1: Launch weak-teacher continuation**

Resume from the best checkpoint with the weaker teacher task/config.

**Step 2: Compare metrics against the strong-teacher run**

Require that:

- completion remains high
- path errors do not collapse
- action deviation from teacher grows

**Step 3: Launch teacher-off continuation**

Resume from the best weak-teacher checkpoint using a pure-RL continuation config.

**Step 4: Evaluate teacher-off checkpoints with the same suite**

Use the exact same `path_tracking_truth_nowind_v1` suite.

**Step 5: Decide pass/fail**

Pass if:

- teacher-off policy remains stable
- completion remains competitive
- errors are near teacher-guided levels

**Step 6: Commit**

Commit only config/script changes, not logs.

---

### Task 9: Add optional BC warm start only if PPO startup is too slow

**Files:**
- Modify: `scripts/flapping_rl/pretrain_path_tracking_bc.py`
- Add tests as needed near existing BC tests

**Step 1: Do nothing unless needed**

Only execute this task if strong-teacher PPO still wastes substantial compute before becoming flyable.

**Step 2: If needed, write failing tests**

Require:

- teacher rollout dataset generation for path-tracking missions
- BC pretrain compatible with the path-tracking observation/action format

**Step 3: Implement the smallest BC warm start**

Do not overbuild dataset infrastructure. Use direct teacher rollouts from the existing env.

**Step 4: Re-run a shortened training comparison**

Compare:

- PPO from scratch
- PPO from BC warm start

**Step 5: Keep BC only if it materially helps**

If it does not improve time-to-first-good-checkpoint, remove or defer it.

---

## Exit Criteria for This Plan

This implementation plan is complete when all of the following are true:

1. a new teacher-guided generic path-tracking RL task is registered
2. train-and-watch works on that task
3. the policy learns usable no-wind mixed-mission path tracking
4. a teacher-off continuation run remains stable
5. best-checkpoint selection and evaluation are reproducible

At that point, the next plan should cover:

- wind curriculum in training
- stronger path distributions
- estimated-state / noisy-observation robustness
