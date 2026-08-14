# PureRL Curriculum 2 Longitudinal Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add reproducible no-target-speed C2a/C2b/C2c climb/descent training and evaluation on the measured CPU-native PureRL plant while preserving exact C1 reward and observation behavior.

**Architecture:** Add a pure-Torch longitudinal path sampler/query module beside the existing PureRL observation and reward helpers. The measured straight-flight environment remains the runtime owner and selects the new path logic only in explicit C2 config subclasses; the C1 task keeps its current branch. Evaluation, promotion, and retention stay outside the environment and never mutate a running task distribution.

**Tech Stack:** Python 3.11, PyTorch tensors, Isaac Lab DirectRLEnv/configclass, Gymnasium task registration, RSL-RL launcher helpers, pytest, JSON/CSV evaluation artifacts.

---

### Task 1: Add the pure longitudinal stage and sampler contract

**Files:**
- Create: `source/flapping_bot/flapping_bot/direct/flapping_bot/pure_rl_longitudinal_path.py`
- Create: `tests/test_pure_rl_longitudinal_path.py`

**Step 1: Write failing stage-contract tests**

Cover the exact approved values:

```python
def test_stage_configs_freeze_geometry_and_task_weights():
    assert LONGITUDINAL_STAGE_CONFIGS["c2a"].absolute_slope_deg_range == (1.5, 4.0)
    assert LONGITUDINAL_STAGE_CONFIGS["c2a"].task_probabilities == (0.50, 0.25, 0.25)
    assert LONGITUDINAL_STAGE_CONFIGS["c2b"].absolute_slope_deg_range == (2.0, 6.0)
    assert LONGITUDINAL_STAGE_CONFIGS["c2b"].task_probabilities == (0.30, 0.35, 0.35)
    assert LONGITUDINAL_STAGE_CONFIGS["c2c"].absolute_slope_deg_range == (4.0, 12.0)
    assert LONGITUDINAL_STAGE_CONFIGS["c2c"].task_probabilities == (0.25, 0.375, 0.375)
```

Also require 15--20 m entry, 20--30 m slope, at least 15 m recovery, equal climb/descent signs, probabilities summing to one, finite ordered bounds, and explicit rejection of unknown stages.

**Step 2: Run RED**

```bash
source /home/zn/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
TERM=xterm ./isaaclab.sh -p -m pytest -q tests/test_pure_rl_longitudinal_path.py
```

Expected: import failure because `pure_rl_longitudinal_path` does not exist.

**Step 3: Implement immutable stage types and vectorized sampling**

Implement:

```python
LEVEL_TASK_ID = 0
CLIMB_TASK_ID = 1
DESCENT_TASK_ID = 2

@dataclass(frozen=True)
class PureRLLongitudinalStageConfig:
    stage_id: str
    absolute_slope_deg_range: tuple[float, float]
    task_probabilities: tuple[float, float, float]
    entry_length_m_range: tuple[float, float] = (15.0, 20.0)
    slope_length_m_range: tuple[float, float] = (20.0, 30.0)
    minimum_recovery_length_m: float = 15.0

@dataclass(frozen=True)
class PureRLLongitudinalPathBatch:
    task_id: torch.Tensor
    heading_rad: torch.Tensor
    signed_slope_rad: torch.Tensor
    entry_length_m: torch.Tensor
    slope_length_m: torch.Tensor
    initial_altitude_m: torch.Tensor
```

`sample_longitudinal_path_batch()` must accept `num_paths`, `stage`, `device`, `dtype`, and an optional device-compatible `torch.Generator`. It must use batched Torch operations, preserve device/dtype, set level slopes exactly to zero, and never loop over environments.

**Step 4: Run GREEN and the C1 contract test**

```bash
TERM=xterm ./isaaclab.sh -p -m pytest -q \
  tests/test_pure_rl_longitudinal_path.py \
  tests/test_pure_rl_curriculum_contract.py
```

Expected: all pass.

**Step 5: Commit**

```bash
git add source/flapping_bot/flapping_bot/direct/flapping_bot/pure_rl_longitudinal_path.py tests/test_pure_rl_longitudinal_path.py
git commit -m "feat(rl): add longitudinal curriculum sampler"
```

### Task 2: Add vectorized path query, preview, and orthonormal basis

**Files:**
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/pure_rl_longitudinal_path.py`
- Modify: `tests/test_pure_rl_longitudinal_path.py`

**Step 1: Write failing geometry tests**

Test level, entry, slope, recovery, climb, descent, and batch-one cases. For a 20 m entry, 25 m slope, and 4-degree climb, require:

```python
assert query_at_10.reference_altitude_m == pytest.approx(10.0)
assert query_at_30.reference_altitude_m == pytest.approx(10.0 + 10.0 * math.tan(math.radians(4.0)))
assert query_at_60.reference_altitude_m == pytest.approx(10.0 + 25.0 * math.tan(math.radians(4.0)))
assert query_at_30.active_slope_rad == pytest.approx(math.radians(4.0))
assert query_at_60.active_slope_rad == 0.0
```

Require five preview points to cross entry/slope/recovery boundaries correctly. Require `t_hat`, `n_h`, and `n_v` to have unit norm, pairwise zero dot products, and right-handed consistent signs in z-up coordinates.

**Step 2: Run RED**

Run the focused test and require missing query functions.

**Step 3: Implement the query contract**

Add:

```python
@dataclass(frozen=True)
class PureRLLongitudinalPathQuery:
    horizontal_progress_m: torch.Tensor
    reference_altitude_m: torch.Tensor
    height_error_m: torch.Tensor
    cross_track_error_m: torch.Tensor
    active_slope_rad: torch.Tensor
    tangent_world: torch.Tensor
    lateral_normal_world: torch.Tensor
    vertical_normal_world: torch.Tensor
    preview_points_world_m: torch.Tensor
    reached_recovery: torch.Tensor
```

Use horizontal progress along the sampled heading. Evaluate the piecewise-linear altitude with `torch.where`; after the slope, extend the final level segment without clamping progress. Compute time-based preview horizontal increments from current horizontal along-heading speed clamped by the existing 1--12 m/s preview bounds. Do not add Bézier smoothing or a new observation channel.

**Step 4: Verify C1 geometric equivalence**

For zero slope and arbitrary randomized heading, compare the new query against the current straight-line equations using `torch.testing.assert_close(..., rtol=0, atol=0)` where operations are identical and a documented machine-epsilon tolerance otherwise.

**Step 5: Run GREEN and commit**

```bash
git add source/flapping_bot/flapping_bot/direct/flapping_bot/pure_rl_longitudinal_path.py tests/test_pure_rl_longitudinal_path.py
git commit -m "feat(rl): query longitudinal path geometry"
```

### Task 3: Generalize the reward without breaking the C1 API

**Files:**
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/pure_rl_reward.py`
- Modify: `tests/test_pure_rl_reward_contract.py`

**Step 1: Write failing tangent/normal reward tests**

Add a new API:

```python
compute_pure_rl_path_reward_terms(
    tangent_velocity_mps=...,
    lateral_normal_velocity_mps=...,
    vertical_normal_velocity_mps=...,
    ...,
)
```

Test a velocity exactly aligned with a 6-degree climb: tangent reward must be positive and both normal rewards must equal one. Test the same world velocity against a level path: vertical-normal reward must be below one.

**Step 2: Write an exact C1 compatibility test**

Keep `compute_pure_rl_reward_terms()` as a compatibility wrapper. Map its existing `along_track_velocity_mps`, `cross_track_velocity_mps`, and `vertical_velocity_mps` arguments into the new API. For fixed tensors, require every old and new `PureRLRewardTerms` field to be equal.

**Step 3: Run RED**

Expected: missing `compute_pure_rl_path_reward_terms`.

**Step 4: Implement the minimal reward change**

Use:

```python
progress_reward = torch.tanh(tangent_velocity_mps / config.progress_speed_scale_mps)
lateral_reward = torch.exp(-torch.square(lateral_normal_velocity_mps / config.cross_track_speed_scale_mps))
vertical_reward = torch.exp(-torch.square(vertical_normal_velocity_mps / config.vertical_speed_scale_mps))
velocity_reward = 0.5 * (lateral_reward + vertical_reward)
```

Do not change any scale, weight, penalty, termination threshold, or existing public wrapper.

**Step 5: Run GREEN and commit**

```bash
TERM=xterm ./isaaclab.sh -p -m pytest -q tests/test_pure_rl_reward_contract.py
git add source/flapping_bot/flapping_bot/direct/flapping_bot/pure_rl_reward.py tests/test_pure_rl_reward_contract.py
git commit -m "feat(rl): reward three-dimensional path motion"
```

### Task 4: Add explicit C2 environment configs and integrate the path state

**Files:**
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/__init__.py`
- Modify: `source/flapping_bot/flapping_bot/__init__.py`
- Create: `tests/test_pure_rl_longitudinal_env_contract.py`
- Modify: `tests/test_straight_flight_env_reset_contract.py`

**Step 1: Write failing AST/config tests**

Require three explicit subclasses:

```python
FlappingBotStraightFlightDeLaurierMeasuredPureRLC2aEnvCfg
FlappingBotStraightFlightDeLaurierMeasuredPureRLC2bEnvCfg
FlappingBotStraightFlightDeLaurierMeasuredPureRLC2cEnvCfg
```

Each must inherit the measured C1 config, select one immutable longitudinal stage, keep wind disabled, preserve observation dimension 555, and keep the shared action/backend contract. The C1 class must still default to no longitudinal sampler.

**Step 2: Run RED**

Expected: missing C2 config classes.

**Step 3: Add environment buffers and reset sampling**

Add a base config field such as:

```python
pure_rl_longitudinal_stage_id: str | None = None
```

Allocate longitudinal task, angle, length, basis, reference-height, recovery, and telemetry tensors only when the field is not `None`. In `_reset_idx`, sample only `env_ids` and never overwrite other environments. Evaluation schedules, when configured, replace random task/angle/heading/phase sampling and must have aligned lengths.

**Step 4: Route observation preview, reward, and termination**

- C1 continues through the current straight-line code path.
- C2 observation preview comes from the new query and still enters the unchanged normalized 555-value builder.
- C2 reward projects world velocity on the query basis and calls `compute_pure_rl_path_reward_terms`.
- C2 termination uses current horizontal cross-track and query-relative height errors with the existing 3 m thresholds.
- Telemetry exposes sampled task/stage/slope, active slope, tangent velocity, both normal velocities, and recovery reached.

Do not import or instantiate the legacy `path_tracking_env.py` task stack.

**Step 5: Run pure integration tests**

Use a small fake environment state or extracted pure adapter to verify partial resets, preview boundary crossing, C1 disabled equivalence, and no change in observation shape.

**Step 6: Run the existing focused suite and commit**

```bash
TERM=xterm ./isaaclab.sh -p -m pytest -q \
  tests/test_pure_rl_longitudinal_env_contract.py \
  tests/test_straight_flight_env_reset_contract.py \
  tests/test_pure_rl_observation_contract.py \
  tests/test_pure_rl_reward_contract.py
git add source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py \
  source/flapping_bot/flapping_bot/direct/flapping_bot/__init__.py \
  source/flapping_bot/flapping_bot/__init__.py \
  tests/test_pure_rl_longitudinal_env_contract.py \
  tests/test_straight_flight_env_reset_contract.py
git commit -m "feat(env): integrate longitudinal PureRL curriculum"
```

### Task 5: Build the deterministic C2 evaluation and promotion contracts

**Files:**
- Create: `scripts/flapping_rl/pure_rl_longitudinal_eval.py`
- Create: `scripts/flapping_rl/pure_rl_longitudinal_promotion.py`
- Create: `tests/test_pure_rl_longitudinal_eval.py`
- Create: `tests/test_pure_rl_longitudinal_promotion.py`
- Modify: `scripts/flapping_rl/pure_rl_retention.py`
- Modify: `tests/test_pure_rl_retention.py`

**Step 1: Write failing grid tests**

Require exact case counts: C2a 80, C2b 112, C2c 112, with the Cartesian product of approved slopes, four headings, and four phases. C2a/C2b retain signed 10-degree diagnostics; require signed 15-degree C2c cases in a separate diagnostic grid.

**Step 2: Implement pure grid generation**

Return immutable case records with `stage_id`, `task`, signed slope, heading, flap phase, 17.5 m entry, 25 m slope, 12 s duration, and `promotion_eligible`.

**Step 3: Write failing summary/gate tests**

Require direction-specific aggregation and the approved hard gates:

```python
overall_survival_rate >= 0.95
climb_success_rate >= 0.90
descent_success_rate >= 0.90
recovery_reached_rate >= 0.95
mean_abs_cross_track_error_m <= 0.50
mean_abs_height_error_m <= 0.50
p95_abs_height_error_m <= 1.50
reverse_motion_fraction <= 0.01
finite_metrics is True
```

Saturation remains telemetry-only.

**Step 4: Implement consecutive-checkpoint promotion**

The promotion helper reads ordered evaluation rows and requires two consecutive passing checkpoints after at least 200 PPO iterations. It also requires a matching C1 retention row with success at least 95%, termination at most 5%, score drop at most five, and mean errors within `max(2 * baseline, 0.25 m)`.

Fail closed on duplicate checkpoints, missing directions, incomplete grids, non-finite values, or absent source-C1 baseline.

**Step 5: Extend retention cells without changing old callers**

Add optional detailed metrics to retention records. Existing minimal C1/trajectory/wind records must continue to pass unchanged. New longitudinal promotion code performs the stricter C1 metric checks.

**Step 6: Run tests and commit**

```bash
TERM=xterm ./isaaclab.sh -p -m pytest -q \
  tests/test_pure_rl_longitudinal_eval.py \
  tests/test_pure_rl_longitudinal_promotion.py \
  tests/test_pure_rl_retention.py
git add scripts/flapping_rl/pure_rl_longitudinal_eval.py \
  scripts/flapping_rl/pure_rl_longitudinal_promotion.py \
  scripts/flapping_rl/pure_rl_retention.py \
  tests/test_pure_rl_longitudinal_eval.py \
  tests/test_pure_rl_longitudinal_promotion.py \
  tests/test_pure_rl_retention.py
git commit -m "feat(rl): gate longitudinal curriculum promotion"
```

### Task 6: Register C2 tasks and connect launcher/evaluation routing

**Files:**
- Modify with explicit user approval: `source/isaaclab_tasks/isaaclab_tasks/direct/flapping_bot/__init__.py`
- Modify: `scripts/flapping_rl/pure_rl_eval_common.py`
- Modify: `scripts/flapping_rl/eval_suites.py`
- Modify: `scripts/flapping_rl/watch_and_eval.py`
- Modify: `scripts/flapping_rl/train_and_watch.py`
- Modify: `scripts/flapping_rl/checkpoint_selection.py`
- Create: `tests/test_pure_rl_longitudinal_task_registration.py`
- Modify: `tests/test_pure_rl_eval_common.py`
- Modify: `tests/test_eval_suites.py`
- Modify: `tests/test_watch_and_eval.py`
- Modify: `tests/test_train_and_watch.py`
- Modify: `tests/test_checkpoint_selection.py`

**Approval gate:** Repository policy forbids modifying `source/isaaclab_tasks/` without approval of the exact file and reason. Before this task, obtain approval to add only three project task registrations to `source/isaaclab_tasks/isaaclab_tasks/direct/flapping_bot/__init__.py`. Do not modify upstream environment or framework APIs.

**Step 1: Write failing registration/routing tests**

Require task IDs:

```text
Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C2a-Direct-v0
Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C2b-Direct-v0
Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C2c-Direct-v0
```

Each must select the corresponding config and reuse `FlappingBotStraightFlightPPORunnerCfg`. C1 task behavior and ID remain unchanged.

**Step 2: Register the tasks**

Add only imports for the three config classes and three `gym.register` blocks in the approved registration file.

**Step 3: Route evaluation contracts**

Keep `pure_rl_curriculum1_v2` unchanged. Add versioned C2 contracts and have the watcher build the fixed longitudinal grid, collect per-step C2 telemetry, aggregate direction-specific metrics, and write promotion plus diagnostic rows. Checkpoint selection must reject a C2 checkpoint that lacks its matching contract or retention pass.

**Step 4: Keep stage changes external**

Do not mutate C2a into C2b inside a running process. A promoted checkpoint starts a new run under the next task ID using the existing resume/load-checkpoint interface. Record source checkpoint path/hash and source stage in run metadata.

**Step 5: Run routing tests and commit**

```bash
TERM=xterm ./isaaclab.sh -p -m pytest -q \
  tests/test_pure_rl_longitudinal_task_registration.py \
  tests/test_pure_rl_eval_common.py \
  tests/test_eval_suites.py \
  tests/test_watch_and_eval.py \
  tests/test_train_and_watch.py \
  tests/test_checkpoint_selection.py
git add source/isaaclab_tasks/isaaclab_tasks/direct/flapping_bot/__init__.py \
  scripts/flapping_rl/pure_rl_eval_common.py \
  scripts/flapping_rl/eval_suites.py \
  scripts/flapping_rl/watch_and_eval.py \
  scripts/flapping_rl/train_and_watch.py \
  scripts/flapping_rl/checkpoint_selection.py \
  tests/test_pure_rl_longitudinal_task_registration.py \
  tests/test_pure_rl_eval_common.py tests/test_eval_suites.py tests/test_watch_and_eval.py \
  tests/test_train_and_watch.py tests/test_checkpoint_selection.py
git commit -m "feat(rl): register longitudinal curriculum tasks"
```

### Task 7: Add fresh-process CPU runtime gates

**Files:**
- Create: `tests/test_native_cpu_pure_rl_longitudinal_runtime_isaac.py`
- Modify: `docs/PROJECT_STATE.md`

**Step 1: Write the runtime test**

Start C2a in a fresh CPU process with a small environment count. Force level, climb, and descent schedules and verify:

- native holonomic backend is active and GPU remains disabled;
- observation shape stays `(N, 555)`;
- sampled schedule is reproduced after reset;
- preview points show positive/negative z changes before slope entry;
- tangent/normal basis and reward/termination telemetry are finite;
- level cases reproduce the C1 reward decomposition within the approved tolerance;
- partial reset changes only selected environments;
- repeated resets do not hang.

Do not train PPO in this gate.

**Step 2: Run fresh-process runtime validation**

```bash
source /home/zn/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
TERM=xterm ./isaaclab.sh -p -m pytest -q tests/test_native_cpu_pure_rl_longitudinal_runtime_isaac.py
```

Expected: one passing runtime test; headless GLFW warnings are acceptable, NaN/Inf, hangs, or missing native constraints are not.

**Step 3: Run the full relevant regression suite**

Run all PureRL action, observation, reward, reset, evaluation, launcher, retention, and both CPU-native runtime tests. Run `py_compile` for new modules and `git diff --check`.

**Step 4: Update project state**

Record C2 implementation status and exact validation evidence. State explicitly that no PPO, wind, C3, GPU qualification, or reward-weight tuning was performed.

**Step 5: Commit**

```bash
git add tests/test_native_cpu_pure_rl_longitudinal_runtime_isaac.py docs/PROJECT_STATE.md
git commit -m "test(rl): gate longitudinal curriculum runtime"
```

### Task 8: Produce the C2 training handoff without starting training

**Files:**
- Create: `docs/handoffs/2026-08-10-pure-rl-curriculum2.md`

**Step 1: Record exact launch order**

Document:

1. start C2a from a selected C1 checkpoint;
2. evaluate every 100 iterations after iteration 200;
3. require two consecutive C2a plus C1-retention passes;
4. start a new C2b run from the promoted C2a checkpoint;
5. repeat for C2c;
6. keep stage-specific diagnostics out of selection.

Include exact `train_and_watch.py` commands only after task registration and launcher tests establish the supported CLI.

**Step 2: Final verification**

Review each commit, confirm only approved project files plus the explicitly approved registration file changed, confirm playback/GIF files remain untracked, and report tests not run.

**Step 3: Commit**

```bash
git add docs/handoffs/2026-08-10-pure-rl-curriculum2.md
git commit -m "docs(rl): hand off longitudinal curriculum training"
```
