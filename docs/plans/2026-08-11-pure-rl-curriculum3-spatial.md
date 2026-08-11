# PureRL Curriculum 3 Spatial Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement CPU-native PureRL C3a, C3b, and C3c randomized spatial-path curricula with fixed evaluation grids, safety gates, and C1/C2/C3 retention evidence.

**Architecture:** Add one batched Torch module that owns C3 stage sampling, dense arc-length centerlines, local projection, and preview queries. Integrate it as an explicit alternative to the existing C1/C2 route path inside `FlappingBotStraightFlightEnv`, then add project-local task registration and script-side evaluation/promotion helpers. Preserve the four-action, 555-observation, 60 Hz policy, 480 Hz physics, no-wind, and CPU-native contracts.

**Tech Stack:** Python 3.11, PyTorch tensors, Isaac Lab direct environments, Gymnasium registration, RSL-RL PPO, pytest.

---

The approved design is [2026-08-11-pure-rl-curriculum3-spatial-design.md](2026-08-11-pure-rl-curriculum3-spatial-design.md). Do not change its geometry, reward, evaluation, or retention thresholds while implementing this plan.

Run every Python, test, or simulation command after:

```bash
cd /home/zn/IsaacLab/.worktrees/native-multibody-rl
source /home/zn/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
```

Do not start PPO training as part of this plan. Use the CPU-native runtime only for the final small smoke.

### Task 1: Add C3 stage contracts and batched centerline generation

**Files:**

- Create: `source/flapping_bot/flapping_bot/direct/flapping_bot/pure_rl_spatial_path.py`
- Create: `tests/test_pure_rl_spatial_path.py`

**Step 1: Write failing configuration and sampling tests**

Cover the frozen stage values and batch contract:

```python
def test_spatial_stage_configs_match_approved_ranges_and_probabilities():
    assert SPATIAL_STAGE_CONFIGS["c3a"].geometry_roll_deg_range == (8.0, 14.0)
    assert SPATIAL_STAGE_CONFIGS["c3b"].geometry_roll_deg_range == (10.0, 17.0)
    assert SPATIAL_STAGE_CONFIGS["c3c"].coupled_slope_deg_range == (1.5, 6.0)
    assert SPATIAL_STAGE_CONFIGS["c3a"].task_probabilities == (0.15, 0.25, 0.60)
    assert SPATIAL_STAGE_CONFIGS["c3b"].task_probabilities == (0.15, 0.20, 0.15, 0.50)
    assert SPATIAL_STAGE_CONFIGS["c3c"].task_probabilities == (0.15, 0.20, 0.15, 0.50)


def test_sampled_c3c_paths_respect_coupled_bound_and_metadata():
    batch = sample_spatial_path_batch(
        num_paths=128,
        stage="c3c",
        device="cpu",
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(7),
    )
    demand = torch.square(batch.peak_geometry_roll_rad / math.radians(20.0))
    demand += torch.square(torch.abs(batch.peak_slope_rad) / math.radians(6.0))
    assert bool(torch.all(demand <= 1.0 + 1.0e-12))
    assert batch.points_world_m.shape == (128, 1201, 3)
    assert batch.points_world_m.dtype == torch.float64
```

Also test deterministic repeatability from equal generators, different samples from different seeds, equal left/right and climb/descent support, 2--4 C3c events, 0.25 m spacing, approximately 300 m total length, initial zero curvature/slope, and finite tensors.

**Step 2: Run the tests and verify the missing-module failure**

```bash
./isaaclab.sh -p -m pytest tests/test_pure_rl_spatial_path.py -q
```

Expected: collection fails because `pure_rl_spatial_path` does not exist.

**Step 3: Implement the minimal spatial-path data model**

Define typed immutable configuration plus batched outputs:

```python
@dataclass(frozen=True)
class PureRLSpatialStageConfig:
    stage_id: str
    task_probabilities: tuple[float, ...]
    geometry_roll_deg_range: tuple[float, float]
    finite_turn_deg_range: tuple[float, float]
    coupled_slope_deg_range: tuple[float, float] = (0.0, 0.0)
    event_count_range: tuple[int, int] = (1, 1)
    transition_length_m_range: tuple[float, float] = (8.0, 12.0)
    guard_speed_mps: float = 12.0
    sample_spacing_m: float = 0.25
    path_length_m: float = 300.0


@dataclass(frozen=True)
class PureRLSpatialPathBatch:
    points_world_m: Tensor
    tangent_world: Tensor
    lateral_normal_world: Tensor
    vertical_normal_world: Tensor
    curvature_rad_per_m: Tensor
    slope_rad: Tensor
    turn_activity: Tensor
    final_event_progress_m: Tensor
    task_family_id: Tensor
    turn_sign: Tensor
    vertical_sign: Tensor
    peak_geometry_roll_rad: Tensor
    peak_slope_rad: Tensor
```

Use a fixed maximum of four events so sampling and profile construction remain batched. Convert geometry roll to curvature with

```python
curvature = 9.81 * torch.tan(geometry_roll_rad) / guard_speed_mps**2
```

Build entry, quintic transition, plateau, and exit masks over the fixed arc-length grid. Generate `kappa(s)` and `gamma(s)` with Torch broadcasting, integrate heading and position with cumulative sums, and derive the approved `t_hat`, `n_h`, and `n_v` frame. Dedicated loiter templates may intentionally revisit their circle; other templates must fail validation on a nonadjacent self-intersection.

Use one bounded resampling pass for invalid rows and raise `RuntimeError` if any remain invalid. Do not add adaptive ranges or silent clipping.

**Step 4: Run focused tests**

```bash
./isaaclab.sh -p -m pytest tests/test_pure_rl_spatial_path.py -q
```

Expected: all stage, sampling, bound, shape, dtype, and determinism tests pass.

**Step 5: Commit**

```bash
git add source/flapping_bot/flapping_bot/direct/flapping_bot/pure_rl_spatial_path.py tests/test_pure_rl_spatial_path.py
git commit -m "feat(rl): add spatial curriculum path generator"
```

### Task 2: Add local projection, preview queries, and partial-reset writes

**Files:**

- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/pure_rl_spatial_path.py`
- Modify: `tests/test_pure_rl_spatial_path.py`

**Step 1: Write failing geometry-query tests**

Add tests for:

- exact interpolation on a straight table;
- signed horizontal and vertical normal errors;
- orthonormal path frames for level, slope, turn, and coupled cases;
- a search window that permits 2 m backward progress but does not jump to a nonadjacent branch;
- preview times `(0.12, 0.24, 0.36, 0.48, 0.60)` and speed clamp `[1, 12]` m/s;
- preview output shape `(N, 5, 3)`;
- zero-curvature/constant-slope points and frames matching `query_longitudinal_path`;
- copying only selected rows during a partial reset.

Use an explicit result contract:

```python
@dataclass(frozen=True)
class PureRLSpatialPathQuery:
    progress_m: Tensor
    reference_position_world_m: Tensor
    horizontal_normal_error_m: Tensor
    vertical_normal_error_m: Tensor
    tangent_world: Tensor
    lateral_normal_world: Tensor
    vertical_normal_world: Tensor
    active_curvature_rad_per_m: Tensor
    active_slope_rad: Tensor
    turn_activity: Tensor
    preview_points_world_m: Tensor
    reached_all_events: Tensor
```

**Step 2: Verify failure**

```bash
./isaaclab.sh -p -m pytest tests/test_pure_rl_spatial_path.py -k "query or preview or partial" -q
```

Expected: failures identify the missing query and row-write functions.

**Step 3: Implement local segment projection and interpolation**

Implement:

```python
def query_spatial_path(
    *,
    path: PureRLSpatialPathBatch,
    previous_progress_m: Tensor,
    position_world_m: Tensor,
    ground_velocity_world_mps: Tensor,
    minimum_preview_speed_mps: float = 1.0,
    maximum_preview_speed_mps: float = 12.0,
    preview_times_s: tuple[float, ...] = PURE_RL_PREVIEW_TIMES_S,
) -> PureRLSpatialPathQuery: ...


def write_spatial_path_batch_rows_(
    *, destination: PureRLSpatialPathBatch, env_ids: Tensor, source: PureRLSpatialPathBatch
) -> None: ...
```

Resolve the previous table index from `previous_progress_m`, gather a fixed local segment window containing at least 2 m behind and 4 m ahead, project onto each segment with a guarded denominator, and select the minimum-distance projection. Linearly interpolate the path table for the closest point and all preview queries. Do not search the full 1,201-point table every policy step.

**Step 4: Run all pure geometry tests**

```bash
./isaaclab.sh -p -m pytest tests/test_pure_rl_spatial_path.py tests/test_pure_rl_longitudinal_path.py tests/test_pure_rl_observation_contract.py -q
```

Expected: all tests pass, including C1/C2 preview-contract regressions.

**Step 5: Commit**

```bash
git add source/flapping_bot/flapping_bot/direct/flapping_bot/pure_rl_spatial_path.py tests/test_pure_rl_spatial_path.py
git commit -m "feat(rl): query spatial curriculum paths"
```

### Task 3: Add curvature-conditioned roll reward and roll-limit termination

**Files:**

- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/pure_rl_reward.py`
- Modify: `tests/test_pure_rl_reward_contract.py`

**Step 1: Write failing reward and termination tests**

Preserve exact C1 behavior at `turn_activity=0`, then test active turns:

```python
def test_spatial_roll_term_preserves_c1_and_relaxes_valid_turn_bank():
    straight = compute_pure_rl_spatial_path_reward_terms(..., turn_activity=torch.zeros(4))
    baseline = compute_pure_rl_path_reward_terms(...)
    torch.testing.assert_close(straight.total_reward, baseline.total_reward)

    turning = compute_pure_rl_spatial_path_reward_terms(
        ...,
        roll_rad=torch.deg2rad(torch.tensor([0.0, 25.0, 30.0, 35.0])),
        turn_activity=torch.ones(4),
    )
    assert turning.roll_reward[0] == turning.roll_reward[1] == 1.0
    assert 0.0 < turning.roll_reward[2] < 1.0
    assert turning.roll_reward[3] == 0.0
```

Test smooth blending at turn activity 0, 0.5, and 1, plus a distinct roll-limit termination at exactly 35 degrees. Confirm existing C1/C2 reward and termination tests remain unchanged.

**Step 2: Verify failure**

```bash
./isaaclab.sh -p -m pytest tests/test_pure_rl_reward_contract.py -k spatial -q
```

Expected: missing spatial reward/termination APIs.

**Step 3: Implement wrappers without changing C1/C2 defaults**

Add a `PureRLSpatialTerminationTerms` result and two functions:

```python
def compute_pure_rl_spatial_path_reward_terms(*, turn_activity: Tensor, **path_inputs) -> PureRLRewardTerms:
    base = compute_pure_rl_path_reward_terms(**path_inputs)
    roll = path_inputs["roll_rad"]
    normalized_excess = torch.clamp(
        (torch.abs(roll) - math.radians(25.0)) / math.radians(10.0), min=0.0, max=1.0
    )
    turn_roll_reward = 1.0 - torch.square(normalized_excess)
    spatial_roll_reward = torch.lerp(base.roll_reward, turn_roll_reward, turn_activity)
    spatial_total = base.total_reward + path_inputs["config"].roll_reward_weight * (
        spatial_roll_reward - base.roll_reward
    )
    return replace(base, roll_reward=spatial_roll_reward, total_reward=spatial_total)


def compute_pure_rl_spatial_termination_terms(*, roll_rad: Tensor, **base_inputs) -> PureRLSpatialTerminationTerms:
    base = compute_pure_rl_termination_terms(**base_inputs)
    roll_limit = torch.abs(roll_rad) >= math.radians(35.0)
    return PureRLSpatialTerminationTerms.from_base(base, roll_limit=roll_limit)
```

Validate shape, finite values, and `turn_activity` in `[0, 1]`. Keep all existing public functions and defaults intact.

**Step 4: Run reward regressions**

```bash
./isaaclab.sh -p -m pytest tests/test_pure_rl_reward_contract.py tests/test_pure_rl_fixed_action_reward_validation.py tests/test_pure_rl_random_action_reward_validation.py -q
```

Expected: all tests pass and existing C1/C2 numerical contracts remain unchanged.

**Step 5: Commit**

```bash
git add source/flapping_bot/flapping_bot/direct/flapping_bot/pure_rl_reward.py tests/test_pure_rl_reward_contract.py
git commit -m "feat(rl): gate roll reward for spatial turns"
```

### Task 4: Integrate C3 state, reset, observations, rewards, and terminations

**Files:**

- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/__init__.py`
- Modify: `source/flapping_bot/flapping_bot/__init__.py`
- Create: `tests/test_pure_rl_spatial_env_contract.py`

**Step 1: Write failing environment-contract tests**

Use the existing AST-style C2 contract test as the pattern. Assert:

- base and C1/C2 configs default `pure_rl_spatial_stage_id` to `None`;
- C3a/C3b/C3c config classes inherit directly from the measured C1 config, select only their spatial stage, and set `episode_length_s=20.0`;
- spatial and longitudinal stages cannot both be active;
- reset calls the spatial sampler and partial-row writer;
- actor observations use spatial preview but remain 555 values;
- rewards use `compute_pure_rl_spatial_path_reward_terms`;
- dones use `compute_pure_rl_spatial_termination_terms`;
- both package surfaces export all three C3 config classes.

Add a pure helper test that a deterministic evaluation batch can replace only selected environment rows.

**Step 2: Verify failure**

```bash
./isaaclab.sh -p -m pytest tests/test_pure_rl_spatial_env_contract.py -q
```

Expected: C3 config classes and environment routing are absent.

**Step 3: Add explicit C3 configuration fields and classes**

Add only the fields needed by current callers:

```python
pure_rl_spatial_stage_id: str | None = None
pure_rl_eval_spatial_template_schedule: tuple[int, ...] | None = None
pure_rl_eval_spatial_geometry_roll_deg_schedule: tuple[float, ...] | None = None
pure_rl_eval_spatial_slope_deg_schedule: tuple[float, ...] | None = None
pure_rl_eval_spatial_turn_sign_schedule: tuple[int, ...] | None = None
```

Define `FlappingBotStraightFlightDeLaurierMeasuredPureRLC3aEnvCfg`, `C3bEnvCfg`, and `C3cEnvCfg`. Do not change C1/C2 config values.

**Step 4: Integrate one mutually exclusive route path**

At construction, resolve either a longitudinal stage or a spatial stage and raise `ValueError` if both are configured. Allocate the dense spatial path batch, progress buffer, query telemetry, and roll-limit termination buffer only for C3.

On reset:

- sample independent rows for the reset `env_ids`;
- apply fixed evaluation schedules when present;
- write only reset rows;
- reset path progress to zero;
- preserve randomized heading and flap phase behavior.

Use one `_query_pure_rl_spatial_path()` helper for preview, reward, and termination. Update progress from its local projection and expose:

- horizontal/vertical normal errors;
- tangent/lateral-normal/vertical-normal velocity;
- active curvature and slope;
- turn activity;
- all-event completion;
- sampled family and signs;
- absolute roll and roll-limit termination.

Do not add any actor-observation fields.

**Step 5: Run contract and regression tests**

```bash
./isaaclab.sh -p -m pytest \
  tests/test_pure_rl_spatial_env_contract.py \
  tests/test_pure_rl_longitudinal_env_contract.py \
  tests/test_pure_rl_observation_contract.py \
  tests/test_pure_rl_reward_contract.py -q
```

Expected: all tests pass, including unchanged C1/C2 contracts.

**Step 6: Commit**

```bash
git add \
  source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py \
  source/flapping_bot/flapping_bot/direct/flapping_bot/__init__.py \
  source/flapping_bot/flapping_bot/__init__.py \
  tests/test_pure_rl_spatial_env_contract.py
git commit -m "feat(rl): integrate spatial curriculum environments"
```

### Task 5: Register C3 tasks locally and extend curriculum provenance

**Files:**

- Modify: `source/flapping_bot/flapping_bot/direct/__init__.py`
- Modify: `scripts/flapping_rl/pure_rl_eval_common.py`
- Modify: `scripts/flapping_rl/train_and_watch.py`
- Create: `tests/test_pure_rl_spatial_task_registration.py`
- Modify: `tests/test_pure_rl_eval_common.py`
- Modify: `tests/test_train_and_watch.py`

**Step 1: Write failing task and source-chain tests**

Require these exact task IDs:

```text
Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C3a-Direct-v0
Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C3b-Direct-v0
Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C3c-Direct-v0
```

Test the chain `c3a <- c2c`, `c3b <- c3a`, and `c3c <- c3b`, rejection of missing/wrong source stages, same-stage resume behavior, CPU-native backend ownership, and default 256/16/500/25 launcher settings.

**Step 2: Verify failure**

```bash
./isaaclab.sh -p -m pytest \
  tests/test_pure_rl_spatial_task_registration.py \
  tests/test_pure_rl_eval_common.py \
  tests/test_train_and_watch.py -q
```

Expected: C3 mappings and registrations are missing.

**Step 3: Add project-local Gym registration**

Register only the three C3 IDs in `source/flapping_bot/flapping_bot/direct/__init__.py`. Use the three C3 environment config classes and the existing runner config as a string entry point:

```python
_RSL_RL_CFG = (
    "isaaclab_tasks.direct.flapping_bot.agents."
    "rsl_rl_ppo_straightflight_cfg:FlappingBotStraightFlightPPORunnerCfg"
)
```

This package is imported while the existing flapping task module resolves `flapping_bot` classes, so the C3 IDs become available before Hydra calls `gym.spec`. Do not modify `source/isaaclab_tasks/isaaclab_tasks/direct/flapping_bot/__init__.py`; it is an upstream-protected file.

**Step 4: Extend stage detection and warm-start provenance**

In `pure_rl_eval_common.py`, preserve `longitudinal_stage_for_task()` and add `spatial_stage_for_task()` plus `curriculum_stage_for_task()`. Add the three C3 IDs to CPU-native backend ownership.

In `train_and_watch.py`, use the generic curriculum stage for source validation and extend the source map. Preserve the existing checkpoint digest because it is curriculum provenance evidence, not a routine duplicate hash.

**Step 5: Run tests and an import-only registry check**

```bash
./isaaclab.sh -p -m pytest \
  tests/test_pure_rl_spatial_task_registration.py \
  tests/test_pure_rl_eval_common.py \
  tests/test_train_and_watch.py -q
./isaaclab.sh -p -c "import isaaclab_tasks, gymnasium as gym; print(gym.spec('Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C3a-Direct-v0').id)"
```

Expected: tests pass and the final command prints the exact C3a task ID.

**Step 6: Commit**

```bash
git add \
  source/flapping_bot/flapping_bot/direct/__init__.py \
  scripts/flapping_rl/pure_rl_eval_common.py \
  scripts/flapping_rl/train_and_watch.py \
  tests/test_pure_rl_spatial_task_registration.py \
  tests/test_pure_rl_eval_common.py \
  tests/test_train_and_watch.py
git commit -m "feat(rl): register spatial curriculum tasks"
```

### Task 6: Implement fixed C3 evaluation grids and watcher integration

**Files:**

- Create: `scripts/flapping_rl/pure_rl_spatial_eval.py`
- Create: `tests/test_pure_rl_spatial_eval.py`
- Modify: `scripts/flapping_rl/eval_suites.py`
- Modify: `scripts/flapping_rl/watch_and_eval.py`
- Modify: `scripts/flapping_rl/pure_rl_eval_common.py`
- Modify: `scripts/flapping_rl/train_and_watch.py`
- Modify: `tests/test_eval_suites.py`
- Modify: `tests/test_watch_and_eval.py`
- Modify: `tests/test_train_and_watch.py`

**Step 1: Write failing grid and aggregation tests**

Define immutable evaluation cases and assert exact counts:

```python
assert len(build_spatial_evaluation_grid("c3a")) == 96
assert len(build_spatial_evaluation_grid("c3b")) == 112
assert len(build_spatial_evaluation_grid("c3c")) == 96
```

Verify unique case IDs, held-out deterministic schedules, C3a left/right coverage, all seven C3b families with both mirrors, and all three severities crossed with four C3c sign pairs.

Test aggregation and the frozen hard gates: 95% survival/completion, 90% overall success, 0.5 m mean errors, 1.5 m P95 errors, 1% reverse motion, 25 degree P95 roll, zero roll-limit terminations, finite metrics, and stage-specific slice success.

**Step 2: Verify failure**

```bash
./isaaclab.sh -p -m pytest tests/test_pure_rl_spatial_eval.py tests/test_eval_suites.py -q
```

Expected: spatial evaluation APIs and suite choices are absent.

**Step 3: Implement pure evaluation logic**

Add:

```python
@dataclass(frozen=True)
class PureRLSpatialEvaluationCase: ...

@dataclass(frozen=True)
class PureRLSpatialPromotionGate: ...

def build_spatial_evaluation_grid(stage_id: str) -> tuple[PureRLSpatialEvaluationCase, ...]: ...
def summarize_spatial_evaluation(... ) -> dict[str, object]: ...
def row_meets_spatial_promotion_gate(... ) -> bool: ...
```

Fail closed on incomplete grids, duplicate cases, wrong metadata, missing slices, non-finite samples, or ambiguous booleans. Keep diagnostic metrics separate from hard gates.

**Step 4: Wire suites and runtime collection**

Add `pure_rl_spatial_c3a_v1`, `c3b_v1`, and `c3c_v1` to `eval_suites.py`. Carry fixed template, roll, slope, turn-sign, heading, and flap-phase schedules into the environment config.

Extend `read_pure_rl_step_metrics()` only when a spatial path is active. In `watch_and_eval.py`, assert the reset schedule, collect per-step errors, tangent velocity, roll, event completion, and termination causes, then call `summarize_spatial_evaluation`. Set the default environment/episode count to the exact grid count. Do not run C1/C2 retention inside the watcher; those remain separate evidence inputs to promotion.

**Step 5: Run focused script tests**

```bash
./isaaclab.sh -p -m pytest \
  tests/test_pure_rl_spatial_eval.py \
  tests/test_eval_suites.py \
  tests/test_watch_and_eval.py \
  tests/test_train_and_watch.py \
  tests/test_pure_rl_eval_common.py -q
```

Expected: all fixed-grid, aggregation, and launcher tests pass.

**Step 6: Commit**

```bash
git add \
  scripts/flapping_rl/pure_rl_spatial_eval.py \
  scripts/flapping_rl/eval_suites.py \
  scripts/flapping_rl/watch_and_eval.py \
  scripts/flapping_rl/pure_rl_eval_common.py \
  scripts/flapping_rl/train_and_watch.py \
  tests/test_pure_rl_spatial_eval.py \
  tests/test_eval_suites.py \
  tests/test_watch_and_eval.py \
  tests/test_train_and_watch.py \
  tests/test_pure_rl_eval_common.py
git commit -m "feat(rl): evaluate spatial curriculum checkpoints"
```

### Task 7: Add fail-closed C3 promotion and retention

**Files:**

- Create: `scripts/flapping_rl/pure_rl_spatial_promotion.py`
- Create: `tests/test_pure_rl_spatial_promotion.py`
- Modify: `scripts/flapping_rl/pure_rl_retention.py`
- Modify: `tests/test_pure_rl_retention.py`

**Step 1: Write failing promotion tests**

Cover:

- minimum 614,400 transitions and 307,200-transition evaluation spacing;
- 256 environments mapping to iteration 50 minimum and 25 iteration spacing;
- two adjacent current-stage passes promote the later checkpoint;
- C3a requires C1 and C2c rows;
- C3b additionally requires C3a;
- C3c additionally requires C3a and C3b;
- a prior-stage gate failure or success-rate drop greater than five points blocks promotion;
- missing, duplicate, wrong-checkpoint, wrong-stage, non-finite, or nonadjacent evidence fails closed;
- the retention matrix orders `c1_straight`, `c2c`, `c3a`, `c3b`, and `c3c` correctly.

**Step 2: Verify failure**

```bash
./isaaclab.sh -p -m pytest tests/test_pure_rl_spatial_promotion.py tests/test_pure_rl_retention.py -q
```

Expected: spatial promotion APIs are absent.

**Step 3: Implement promotion as a pure evidence operation**

Add:

```python
def evaluate_spatial_promotion(
    evaluation_rows: Sequence[Mapping[str, object]],
    *,
    retention_rows: Sequence[Mapping[str, object]],
    source_baselines: Mapping[str, Mapping[str, object]],
    stage_id: str,
    minimum_ppo_iteration: int,
    evaluation_interval: int,
) -> dict[str, object]: ...
```

Reuse `build_sample_equivalent_promotion_schedule()` and the existing retention matrix representation. Require every prior suite to pass its frozen gate and compare success rate to the matching promoted source baseline. Do not invent automatic rehearsal-probability changes or fallback promotion paths.

**Step 4: Run promotion and C2 regressions**

```bash
./isaaclab.sh -p -m pytest \
  tests/test_pure_rl_spatial_promotion.py \
  tests/test_pure_rl_retention.py \
  tests/test_pure_rl_longitudinal_promotion.py -q
```

Expected: all tests pass and C2 promotion behavior is unchanged.

**Step 5: Commit**

```bash
git add \
  scripts/flapping_rl/pure_rl_spatial_promotion.py \
  scripts/flapping_rl/pure_rl_retention.py \
  tests/test_pure_rl_spatial_promotion.py \
  tests/test_pure_rl_retention.py
git commit -m "feat(rl): gate spatial curriculum promotion"
```

### Task 8: Run the CPU-native integration gate and close the implementation handoff

**Files:**

- Create: `tests/test_native_cpu_pure_rl_spatial_runtime_isaac.py`
- Modify: `docs/PROJECT_STATE.md`
- Create: `docs/handoffs/2026-08-11-pure-rl-curriculum3.md`

**Step 1: Write the runtime smoke before running Isaac Sim**

Use six environments with deterministic C3a level-left/right, C3b sequential, and C3c climb/descent-turn schedules. Assert:

- CPU device, disabled physics replication, and active native holonomic constraints;
- 555-value finite observations;
- exact fixed reset schedules;
- finite and orthonormal route frames;
- curved and coupled preview geometry signs;
- partial resets change only selected path rows and reset only their progress;
- eight zero-action steps remain finite;
- spatial reward and termination telemetry exists;
- 25 degree soft boundary and 35 degree termination are visible in pure helper calls;
- no unintended wind or GPU path is enabled.

**Step 2: Run all non-Isaac focused tests first**

```bash
./isaaclab.sh -p -m pytest \
  tests/test_pure_rl_spatial_path.py \
  tests/test_pure_rl_spatial_env_contract.py \
  tests/test_pure_rl_spatial_task_registration.py \
  tests/test_pure_rl_spatial_eval.py \
  tests/test_pure_rl_spatial_promotion.py \
  tests/test_pure_rl_reward_contract.py \
  tests/test_pure_rl_observation_contract.py \
  tests/test_pure_rl_longitudinal_path.py \
  tests/test_pure_rl_longitudinal_env_contract.py \
  tests/test_pure_rl_longitudinal_eval.py \
  tests/test_pure_rl_longitudinal_promotion.py \
  tests/test_pure_rl_retention.py \
  tests/test_eval_suites.py \
  tests/test_watch_and_eval.py \
  tests/test_train_and_watch.py -q
```

Expected: all focused tests pass.

**Step 3: Run the fresh-process CPU-native runtime smoke**

```bash
./isaaclab.sh -p -m pytest tests/test_native_cpu_pure_rl_spatial_runtime_isaac.py -q
```

Expected: one runtime test passes. Inspect the pytest result itself because the repository wrapper may not propagate the child exit status reliably.

**Step 4: Run final static checks and review the diff**

```bash
./isaaclab.sh -p -m py_compile \
  source/flapping_bot/flapping_bot/direct/flapping_bot/pure_rl_spatial_path.py \
  source/flapping_bot/flapping_bot/direct/flapping_bot/pure_rl_reward.py \
  source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py \
  scripts/flapping_rl/pure_rl_spatial_eval.py \
  scripts/flapping_rl/pure_rl_spatial_promotion.py
git diff --check
git status --short
git diff --stat
```

Expected: compilation succeeds, `git diff --check` is silent, and only the approved project, script, test, and documentation files are modified.

**Step 5: Update state and handoff with observed evidence only**

Record the implemented task IDs, exact tests and results, remaining limitation that no PPO training was run, the requirement to promote C2c before C3a, and the exact C3a warm-start command template. Do not claim convergence, controller performance, or real-flight validation.

**Step 6: Commit the runtime gate and handoff**

```bash
git add \
  tests/test_native_cpu_pure_rl_spatial_runtime_isaac.py \
  docs/PROJECT_STATE.md \
  docs/handoffs/2026-08-11-pure-rl-curriculum3.md
git commit -m "test(rl): close spatial curriculum runtime gate"
```

**Step 7: Final verification**

```bash
git status --short --branch
git log -8 --oneline --decorate
```

Expected: the worktree is clean and the C3 implementation is represented by focused commits. Do not launch C3 training until an accepted C2c checkpoint exists.
