# Loiter-First Primitive RL Repair Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Turn the current primitive path-tracking PPO from “turn improved but loiter still fails” into a stable no-wind primitive policy by adding loiter-specific diagnostics, staged loiter curriculum, orbit-aware reward shaping, loiter-only teacher tightening, and straight rehearsal.

**Architecture:** Keep `FlappingBotPathTrackingEnv` as the single training environment and keep the current RSL-RL PPO + watcher pipeline. Add a thin layer of loiter-specific state, reward, and curriculum logic inside the existing env/eval stack rather than introducing a second loiter environment or a second controller. Validate only on the primitive no-wind suite until `loiter` is repaired.

**Tech Stack:** IsaacLab direct RL envs, Isaac Sim, RSL-RL PPO, existing PX4-like teacher envelope, `train_and_watch.py`, `watch_and_eval.py`, pytest.

---

## Scope

This plan covers only the next repair stage:

- isolate why `loiter` fails while `turn` passes
- make primitive curriculum explicitly loiter-progressive
- add orbit-aware reward and loiter milestones
- keep `straight` alive via rehearsal
- validate with primitive no-wind watcher runs

This plan does **not** cover:

- mixed multi-segment mission RL
- wind curriculum
- estimated-state RL
- BC redesign

---

### Task 1: Add loiter-specific diagnostics to env and eval

**Files:**
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py`
- Modify: `scripts/flapping_rl/path_tracking_eval_common.py`
- Test: `tests/test_path_tracking_rewards.py`
- Test: `tests/test_path_tracking_eval_common.py`

**Step 1: Write the failing tests**

Add tests that require:

- a helper to compute loiter radial error from `position_xy` and circle center / radius
- a helper to compute normalized loiter angular progress
- aggregated eval rows to persist loiter progress diagnostics when a case is `loiter_primitive_nowind`

Example tests:

```python
def test_compute_loiter_radial_error_is_zero_on_target_circle() -> None:
    error = _compute_loiter_radial_error(
        position_xy=torch.tensor([[20.0, 0.0]]),
        center_xy=torch.tensor([[0.0, 0.0]]),
        radius_m=torch.tensor([20.0]),
    )
    assert torch.allclose(error, torch.tensor([0.0]))


def test_compute_loiter_progress_reaches_quarter_turn() -> None:
    progress = _compute_loiter_angular_progress(
        start_angle_rad=torch.tensor([0.0]),
        current_angle_rad=torch.tensor([0.5 * math.pi]),
        turn_direction=torch.tensor([1]),
        total_turns=torch.tensor([1.0]),
    )
    assert torch.allclose(progress, torch.tensor([0.25]), atol=1.0e-4)
```

**Step 2: Run tests to verify they fail**

Run:

```bash
./isaaclab.sh -p -m pytest \
  tests/test_path_tracking_rewards.py \
  tests/test_path_tracking_eval_common.py -q
```

Expected: FAIL because the current env does not expose any loiter-only geometric diagnostics.

**Step 3: Implement the minimal diagnostics**

In `path_tracking_env.py`:

- add helper functions for:
  - loiter radial error
  - loiter angular progress ratio
  - loiter milestone mask (`0.25`, `0.50`, `0.75`, `1.00`)
- store per-env loiter diagnostic buffers only when the active mission segment is `loiter`

In `path_tracking_eval_common.py`:

- add optional columns for loiter cases such as:
  - `mean_loiter_radial_error_m`
  - `mean_loiter_progress_ratio`
  - `quarter_turn_rate`

Do not change the suite schema for non-loiter rows.

**Step 4: Run tests to verify they pass**

Run:

```bash
./isaaclab.sh -p -m pytest \
  tests/test_path_tracking_rewards.py \
  tests/test_path_tracking_eval_common.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add \
  source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py \
  scripts/flapping_rl/path_tracking_eval_common.py \
  tests/test_path_tracking_rewards.py \
  tests/test_path_tracking_eval_common.py
git commit -m "feat: add loiter-specific rl diagnostics"
```

---

### Task 2: Add staged loiter curriculum with explicit rehearsal

**Files:**
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py`
- Modify: `scripts/flapping_rl/eval_suites.py`
- Test: `tests/test_path_tracking_env_contract.py`
- Test: `tests/test_eval_suites.py`

**Step 1: Write the failing tests**

Add tests that require:

- curriculum stages to support:
  - `turn_only`
  - `loiter_quarter`
  - `loiter_half`
  - `loiter_full_with_straight_rehearsal`
- primitive eval suite to remain unchanged while training curriculum changes
- a small `straight` rehearsal probability / enablement during loiter-focused stages

Example tests:

```python
def test_loiter_curriculum_stage_maps_to_quarter_turn() -> None:
    cfg = FlappingBotPathTrackingPrimitiveWeakTeacherRLEnvCfg()
    stage = _resolve_loiter_curriculum_stage(step=20_000, cfg=cfg)
    assert stage.loiter_turns == 0.25


def test_loiter_curriculum_keeps_straight_rehearsal_enabled() -> None:
    flags = _resolve_path_tracking_curriculum(...)
    assert flags.allow_straight is True
    assert flags.allow_loiter is True
```

**Step 2: Run tests to verify they fail**

Run:

```bash
./isaaclab.sh -p -m pytest \
  tests/test_path_tracking_env_contract.py \
  tests/test_eval_suites.py -q
```

Expected: FAIL because the current primitive curriculum only switches primitive kinds, not loiter difficulty.

**Step 3: Implement the staged loiter curriculum**

In `path_tracking_env.py`:

- add loiter-stage config such as:
  - `loiter_curriculum_enabled`
  - `loiter_curriculum_stage_steps`
  - `loiter_curriculum_stage_turns`
  - `loiter_curriculum_straight_rehearsal_prob`
- make mission sampling respect both primitive-kind stage and current loiter-turn fraction

In `eval_suites.py`:

- keep primitive evaluation fixed at full-turn loiter
- do **not** leak training-stage loiter fraction into the evaluation suite

**Step 4: Run tests to verify they pass**

Run:

```bash
./isaaclab.sh -p -m pytest \
  tests/test_path_tracking_env_contract.py \
  tests/test_eval_suites.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add \
  source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py \
  scripts/flapping_rl/eval_suites.py \
  tests/test_path_tracking_env_contract.py \
  tests/test_eval_suites.py
git commit -m "feat: add staged loiter curriculum for primitive rl"
```

---

### Task 3: Add loiter-only orbit reward and milestone bonuses

**Files:**
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py`
- Test: `tests/test_path_tracking_rewards.py`

**Step 1: Write the failing tests**

Add tests that require:

- lower radial error on a loiter segment to produce higher reward
- better tangential alignment on a loiter segment to produce higher reward
- quarter-turn milestone completion to increase reward
- non-loiter segments to remain governed mainly by the existing generic reward

Example tests:

```python
def test_loiter_reward_prefers_small_radial_error() -> None:
    near = _compute_tracking_reward(..., segment_kind="loiter", loiter_radial_error=torch.tensor([0.2]))
    far = _compute_tracking_reward(..., segment_kind="loiter", loiter_radial_error=torch.tensor([3.0]))
    assert near.item() > far.item()


def test_loiter_reward_applies_quarter_turn_bonus_once() -> None:
    reward = _compute_tracking_reward(
        ...,
        segment_kind="loiter",
        loiter_milestone_bonus=torch.tensor([1.0]),
    )
    assert reward.item() > 0.0
```

**Step 2: Run tests to verify they fail**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_path_tracking_rewards.py -q
```

Expected: FAIL because the current reward only has generic curve shaping.

**Step 3: Implement the minimal loiter reward changes**

In `_compute_tracking_reward(...)`:

- add optional arguments:
  - `segment_kind`
  - `loiter_radial_error`
  - `loiter_tangential_error`
  - `loiter_progress_delta`
  - `loiter_milestone_bonus`
- activate additional reward terms only when `segment_kind == "loiter"`
- keep the existing generic path reward untouched for `straight` and `turn`

The intended structure is:

```python
if segment_kind == "loiter":
    reward += radial_track_bonus
    reward += tangential_track_bonus
    reward += loiter_progress_bonus
    reward += loiter_milestone_bonus
```

**Step 4: Run tests to verify they pass**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_path_tracking_rewards.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add \
  source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py \
  tests/test_path_tracking_rewards.py
git commit -m "feat: add orbit-aware loiter reward shaping"
```

---

### Task 4: Tighten teacher only on loiter and protect straight

**Files:**
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/direct/flapping_bot/agents/rsl_rl_ppo_straightflight_cfg.py`
- Test: `tests/test_teacher_schedule.py`
- Test: `tests/test_path_tracking_ppo_cfg.py`
- Test: `tests/test_path_tracking_rewards.py`

**Step 1: Write the failing tests**

Add tests that require:

- loiter segments to receive a stricter teacher envelope than turn segments
- straight rehearsal sampling to remain enabled while loiter-specific tightening is active
- PPO rollout / discount settings to remain long-horizon after the new loiter changes

Example tests:

```python
def test_loiter_teacher_delta_is_tighter_than_turn() -> None:
    delta = _compute_segment_aware_teacher_delta(
        base_delta=1.0,
        segment_kind=("turn", "loiter"),
    )
    assert delta[1] < delta[0]
```

**Step 2: Run tests to verify they fail**

Run:

```bash
./isaaclab.sh -p -m pytest \
  tests/test_teacher_schedule.py \
  tests/test_path_tracking_ppo_cfg.py \
  tests/test_path_tracking_rewards.py -q
```

Expected: FAIL because teacher tightening is only curve-aware, not loiter-aware.

**Step 3: Implement the minimal teacher and rehearsal changes**

In `path_tracking_env.py`:

- split the current curve-aware teacher delta into:
  - base curve-aware tightening
  - extra `loiter` tightening
- make sure explicit evaluation cases still disable curriculum and teacher guidance

In the PPO config:

- keep `num_steps_per_env = 192`
- keep long-horizon `gamma` / `lam`
- only adjust if the new loiter milestones prove too sparse

**Step 4: Run tests to verify they pass**

Run:

```bash
./isaaclab.sh -p -m pytest \
  tests/test_teacher_schedule.py \
  tests/test_path_tracking_ppo_cfg.py \
  tests/test_path_tracking_rewards.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add \
  source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py \
  source/isaaclab_tasks/isaaclab_tasks/direct/flapping_bot/agents/rsl_rl_ppo_straightflight_cfg.py \
  tests/test_teacher_schedule.py \
  tests/test_path_tracking_ppo_cfg.py \
  tests/test_path_tracking_rewards.py
git commit -m "feat: make primitive teacher guidance loiter-aware"
```

---

### Task 5: Run targeted training/eval and decide whether loiter is fixed

**Files:**
- No code changes required
- Output: `logs/rsl_rl/flapping_bot_path_tracking/...`

**Step 1: Run a watcher smoke from the current best primitive checkpoint**

Run:

```bash
./isaaclab.sh -p scripts/flapping_rl/train_and_watch.py \
  --task Isaac-FlappingBot-PathTracking-DeLaurier-PrimitiveWeakTeacherRL-Direct-v0 \
  --run-name pt_loiter_fix_resume \
  --train-device cuda:0 \
  --eval-device cuda:1 \
  --num-envs 64 \
  --eval-num-envs 16 \
  --max-iterations 10 \
  --save-interval 5 \
  --episodes 2 \
  --poll-s 10 \
  --eval-suite path_tracking_truth_primitives_nowind_v1 \
  --resume \
  --load_run 2026-03-22_13-51-36_pt_primitive_bcwarm_v1 \
  --checkpoint model_40.pt \
  --headless
```

Expected:

- training completes
- watcher writes `eval/summary.csv`
- no curriculum/eval crash

**Step 2: Inspect primitive-case metrics**

Check:

- `straight_primitive_nowind`
- `turn_primitive_nowind`
- `loiter_primitive_nowind`
- `suite`

The key question is not the best scalar score. It is whether:

- `turn` stays passed
- `loiter` materially improves
- `straight` remains acceptable

**Step 3: Escalate to a longer run only if the short run is directionally correct**

Run:

```bash
./isaaclab.sh -p scripts/flapping_rl/train_and_watch.py \
  --task Isaac-FlappingBot-PathTracking-DeLaurier-PrimitiveWeakTeacherRL-Direct-v0 \
  --run-name pt_loiter_fix_long \
  --train-device cuda:0 \
  --eval-device cuda:1 \
  --num-envs 128 \
  --eval-num-envs 16 \
  --max-iterations 40 \
  --save-interval 10 \
  --episodes 6 \
  --poll-s 30 \
  --eval-suite path_tracking_truth_primitives_nowind_v1 \
  --resume \
  --load_run <best short loiter-fix run> \
  --checkpoint <best short loiter-fix checkpoint> \
  --headless
```

Expected: loiter metrics continue improving without turn collapse.

**Step 4: Apply explicit go / no-go gates**

Require all of the following before returning to mixed missions:

- `turn_primitive_nowind`: `success_gate_passed == 1`
- `loiter_primitive_nowind`: `completion_rate >= 0.70`
- `loiter_primitive_nowind`: `termination_rate <= 0.30`
- `straight_primitive_nowind`: no catastrophic drop relative to the pre-fix primitive best
- `suite`: no longer dominated by a single passing primitive

If these fail, stop and inspect the new loiter diagnostics before making further reward changes.

**Step 5: Commit**

No commit if only logs changed.
