# Estimated-State Wind RL Ramp Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move the flapping robot from the current controller-validated, non-RL DeLaurier baseline to a trainable RL stack that uses realistic observations, actuator dynamics, and wind disturbances without breaking the existing path-tracking teacher baseline.

**Architecture:** Reuse the existing `FlappingBotStraightFlightEnv` and `FlappingBotPathTrackingEnv` as the only RL environment families. Pull the already-existing `SensorStateEstimator` from `source/flapping_bot/flapping_bot/px4_like/state_estimation.py` into the RL observation pipeline instead of building a second sensor model. Keep the PX4-like controller as the teacher prior and non-RL benchmark, but make both teacher and policy operate on the same wind / state semantics so RL does not learn on privileged truth while evaluation later uses estimated state.

**Tech Stack:** IsaacLab direct RL envs, Isaac Sim, RSL-RL PPO, existing `state_estimation.py`, existing path-tracking teacher, `watch_and_eval.py`, `eval_suites.py`, pytest.

---

## Recommended path

### Option A: Estimated-state first, then wind, then teacher-off RL (**Recommended**)

Use the existing RL envs, switch the actor observation to estimated-state variables derived from simulated sensors, keep light actuator realism enabled, verify no-wind first, then add wind curriculum, then weaken and remove teacher guidance.

Why this is recommended:

- It reuses the existing sensor and estimator code already proven in `scripts/flapping_px4/fly_straight_line.py` and `scripts/flapping_px4/fly_loiter.py`.
- It avoids the known trap of training on privileged truth and discovering late that the policy depends on signals the real vehicle will not have.
- It keeps debugging localized: if performance drops, the first suspect is observation realism or disturbance robustness, not reward churn.

### Option B: Train RL now on truth-state, then retrofit realism later

This is faster to start but likely to produce a policy that must be relearned once estimated-state, actuator lag, and wind are introduced.

Trade-off:

- faster initial learning curve
- higher probability of a second large retraining cycle

### Option C: Feed raw sensor packets directly to the policy now

This is the most sim2real-pure direction, but it is too expensive for the next step because the current repo does not yet have a stable recurrent or history-heavy policy contract around asynchronous GPS / baro / airspeed updates.

Trade-off:

- highest realism
- highest implementation and training risk

**Chosen path:** Option A.

---

## Scope

This plan covers the next practical milestone only:

- actor observations move from truth-state toward estimated-state
- teacher and policy stop disagreeing about wind semantics
- actuator dynamics are no longer idealized in RL
- wind becomes part of the training and evaluation distribution
- path-tracking RL is trained only after the above are in place

This plan does **not** yet include:

- full raw-sensor-only actor inputs
- recurrent policy architecture changes
- domain randomization beyond wind, sensor noise/bias/delay, and actuator realism
- reward redesign unrelated to realism

---

### Task 1: Freeze the realism contract for RL observations and evaluation

**Files:**
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py`
- Modify: `scripts/flapping_rl/eval_suites.py`
- Test: `tests/test_path_tracking_env_contract.py`
- Test: `tests/test_eval_suites.py`

**Step 1: Write the failing tests**

Add contract tests that require:

- a policy observation mode switch, e.g. `truth` vs `estimated`
- identical observation dimension between supported modes
- an estimated-state evaluation suite for path tracking

Example:

```python
def test_path_tracking_estimated_obs_matches_declared_shape():
    cfg = FlappingBotPathTrackingWeakTeacherRLEnvCfg()
    cfg.policy_state_source = "estimated"
    assert cfg.observation_space == EXPECTED_DIM


def test_eval_suite_contains_estimated_wind_cases():
    cases = build_eval_cases("path_tracking_estimated_wind_v1")
    case_names = {case["name"] for case in cases}
    assert "path_est_nowind" in case_names
    assert "path_est_crosswind_steady" in case_names
    assert "path_est_crosswind_ou" in case_names
```

**Step 2: Run tests to verify they fail**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_path_tracking_env_contract.py tests/test_eval_suites.py -q
```

Expected: FAIL because the estimated-state RL contract and new eval suite are not yet defined.

**Step 3: Write the minimal implementation**

Add env config fields in both RL env families for:

- `policy_state_source: "truth" | "estimated"`
- `policy_use_wind_truth: bool`
- `sensor_noise_scale`
- `sensor_bias_scale`
- `sensor_delay_scale`

Add one new eval suite aimed at the future RL milestone:

- `path_tracking_estimated_nowind_v1`
- `path_tracking_estimated_wind_v1`

Do not change action space.

**Step 4: Run tests to verify they pass**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_path_tracking_env_contract.py tests/test_eval_suites.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py \
        source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py \
        scripts/flapping_rl/eval_suites.py \
        tests/test_path_tracking_env_contract.py \
        tests/test_eval_suites.py
git commit -m "feat: freeze estimated-state rl env contract"
```

---

### Task 2: Reuse the existing sensor-estimator block inside RL envs

**Files:**
- Modify: `source/flapping_bot/flapping_bot/px4_like/state_estimation.py`
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py`
- Create: `tests/test_sensor_state_estimator.py`
- Test: `tests/test_path_tracking_observations.py`

**Step 1: Write the failing tests**

Add tests that require:

- estimator outputs stay finite and shape-stable
- zero-noise / zero-bias / zero-delay configuration tracks truth closely
- RL observation builder can consume estimated state instead of truth state

Example:

```python
def test_sensor_state_estimator_zero_noise_tracks_truth():
    est = SensorStateEstimator(
        sensor_cfg=SensorSuiteCfg(
            gps_delay_s=0.0,
            baro_delay_s=0.0,
            airspeed_delay_s=0.0,
            mag_delay_s=0.0,
            gps_pos_noise_std_m=0.0,
            gps_vel_noise_std_mps=0.0,
            baro_alt_noise_std_m=0.0,
            airspeed_noise_std_mps=0.0,
            mag_heading_noise_std_deg=0.0,
            gyro_noise_std_dps=0.0,
            accel_noise_std_mps2=0.0,
        ),
        estimator_cfg=StateEstimatorCfg(),
        num_envs=1,
        device=torch.device("cpu"),
        control_dt_s=0.02,
    )
    # push one deterministic sample and assert finite close outputs
```

**Step 2: Run tests to verify they fail**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_sensor_state_estimator.py tests/test_path_tracking_observations.py -q
```

Expected: FAIL because the estimator is not yet wired into RL env observation generation.

**Step 3: Write the minimal implementation**

In both RL env families:

- instantiate one vectorized `SensorStateEstimator`
- reset estimator state on env reset
- step estimator once per control step using the same truth kinematics already available to the env
- build actor observations from estimated quantities, not truth, when `policy_state_source == "estimated"`

Keep the first estimated-state actor observation compact:

- estimated position / height error signals
- estimated velocity
- estimated Euler attitude
- estimated body rates
- estimated airspeed
- estimated wind estimate
- previous action
- path geometry preview

Do **not** feed raw asynchronous sensor packets to the actor in this task.

**Step 4: Run tests to verify they pass**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_sensor_state_estimator.py tests/test_path_tracking_observations.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add source/flapping_bot/flapping_bot/px4_like/state_estimation.py \
        source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py \
        source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py \
        tests/test_sensor_state_estimator.py \
        tests/test_path_tracking_observations.py
git commit -m "feat: add estimator-backed rl observations"
```

---

### Task 3: Remove wind-truth leakage and align teacher with policy semantics

**Files:**
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py`
- Test: `tests/test_rl_teacher_guidance.py`
- Test: `tests/test_path_tracking_teacher_mask.py`

**Step 1: Write the failing tests**

Add tests that require:

- teacher can be configured to consume estimated wind instead of world-truth wind
- recovery teacher and nominal teacher use the same state source switch
- no silent fallback to wind truth when `policy_state_source == "estimated"`

Example:

```python
def test_teacher_guidance_uses_estimated_wind_when_truth_disabled():
    cfg = FlappingBotPathTrackingWeakTeacherRLEnvCfg()
    cfg.teacher_guidance_use_wind_truth = False
    cfg.policy_state_source = "estimated"
    assert _resolve_teacher_wind_source(cfg) == "estimated"
```

**Step 2: Run tests to verify they fail**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_rl_teacher_guidance.py tests/test_path_tracking_teacher_mask.py -q
```

Expected: FAIL because teacher guidance still allows privileged wind semantics.

**Step 3: Write the minimal implementation**

Make teacher-side state selection explicit:

- truth-state teacher path for controller-only debugging remains available
- RL env default becomes estimated-state teacher semantics once realism mode is enabled
- diagnostics log which source is used for teacher state and wind

Do not remove the truth-mode teacher from the repo. Keep it for non-RL regression and forensic debugging.

**Step 4: Run tests to verify they pass**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_rl_teacher_guidance.py tests/test_path_tracking_teacher_mask.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py \
        source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py \
        tests/test_rl_teacher_guidance.py \
        tests/test_path_tracking_teacher_mask.py
git commit -m "fix: align rl teacher guidance with estimated-state wind semantics"
```

---

### Task 4: Reintroduce actuator realism for RL and make it schedulable

**Files:**
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/flapping_env.py`
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/direct/flapping_bot/agents/rsl_rl_ppo_straightflight_cfg.py`
- Test: `tests/test_path_tracking_env_contract.py`
- Test: `tests/test_path_tracking_ppo_cfg.py`

**Step 1: Write the failing tests**

Add tests that require:

- path-tracking RL configs define nonzero actuator realism defaults
- a weak-reality schedule exists for early training and a stronger one for later training

Example:

```python
def test_path_tracking_cfg_enables_nonzero_action_filtering():
    cfg = FlappingBotPathTrackingWeakTeacherRLEnvCfg()
    assert cfg.act_lpf_tau_s > 0.0
    assert cfg.act_rate_limit_per_s > 0.0
```

**Step 2: Run tests to verify they fail**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_path_tracking_env_contract.py tests/test_path_tracking_ppo_cfg.py -q
```

Expected: FAIL because current path-tracking RL env defaults are still idealized at `0.0`.

**Step 3: Write the minimal implementation**

Set conservative but nonzero realism defaults for RL tasks, for example:

- early stage: small `act_lpf_tau_s`, moderate `act_rate_limit_per_s`
- later stage: same or slightly stronger realism, but do not exceed the plant’s observed useful bandwidth

Expose them as explicit config knobs so training scripts can stage them.

**Step 4: Run tests to verify they pass**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_path_tracking_env_contract.py tests/test_path_tracking_ppo_cfg.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add source/flapping_bot/flapping_bot/direct/flapping_bot/flapping_env.py \
        source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py \
        source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py \
        source/isaaclab_tasks/isaaclab_tasks/direct/flapping_bot/agents/rsl_rl_ppo_straightflight_cfg.py \
        tests/test_path_tracking_env_contract.py \
        tests/test_path_tracking_ppo_cfg.py
git commit -m "feat: add actuator realism defaults for rl tasks"
```

---

### Task 5: Expand RL evaluation suites before large training runs

**Files:**
- Modify: `scripts/flapping_rl/eval_suites.py`
- Modify: `scripts/flapping_rl/watch_and_eval.py`
- Modify: `scripts/flapping_rl/eval_path_tracking_checkpoint.py`
- Test: `tests/test_eval_path_tracking_suites.py`
- Test: `tests/test_watch_and_eval.py`

**Step 1: Write the failing tests**

Add tests that require the watch/eval stack to score:

- estimated-state no-wind
- estimated-state steady crosswind
- estimated-state OU gust

Example:

```python
def test_watch_and_eval_supports_estimated_wind_suite():
    assert "path_tracking_estimated_wind_v1" in get_eval_suite_choices()
```

**Step 2: Run tests to verify they fail**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_eval_path_tracking_suites.py tests/test_watch_and_eval.py -q
```

Expected: FAIL because the new RL readiness suites are not yet wired through the evaluator.

**Step 3: Write the minimal implementation**

Add two suites:

- `path_tracking_estimated_nowind_v1`
- `path_tracking_estimated_wind_v1`

Each suite should include fixed-seed straight, turn, loiter, and mixed-mission cases. The wind suite should include at least:

- one mild steady crosswind case
- one stronger steady crosswind case
- one OU gust case

**Step 4: Run tests to verify they pass**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_eval_path_tracking_suites.py tests/test_watch_and_eval.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add scripts/flapping_rl/eval_suites.py \
        scripts/flapping_rl/watch_and_eval.py \
        scripts/flapping_rl/eval_path_tracking_checkpoint.py \
        tests/test_eval_path_tracking_suites.py \
        tests/test_watch_and_eval.py
git commit -m "feat: add estimated-state wind eval suites for path tracking"
```

---

### Task 6: Run staged RL training instead of one monolithic training job

**Files:**
- Modify: `scripts/flapping_rl/train_and_watch.py`
- Modify: `scripts/flapping_rl/eval_path_tracking_checkpoint.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/direct/flapping_bot/agents/rsl_rl_ppo_straightflight_cfg.py`
- Create: `docs/analysis/flapping_rl/estimated_state_wind_rl_baseline.md`
- Test: `tests/test_train_and_watch.py`

**Step 1: Write the failing tests**

Add tests that require a named staged training recipe, for example:

- stage A: estimated-state, no-wind, weak teacher
- stage B: estimated-state, wind curriculum, weak teacher
- stage C: estimated-state, wind curriculum, pure-RL continuation

Example:

```python
def test_train_and_watch_supports_estimated_state_recipe():
    args = parse_args(["--recipe", "path_tracking_estimated_wind_v1"])
    assert args.recipe == "path_tracking_estimated_wind_v1"
```

**Step 2: Run tests to verify they fail**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_train_and_watch.py -q
```

Expected: FAIL because the staged recipe does not exist yet.

**Step 3: Write the minimal implementation**

Define one recommended training ladder:

1. `teacher-guided estimated-state no-wind`
2. `teacher-guided estimated-state wind`
3. `weak-teacher estimated-state wind`
4. `pure-rl continuation on the same env`

Keep the first production target modest:

- actor uses estimated-state only
- no privileged wind in the actor
- no raw sensor packet actor input

**Step 4: Run tests to verify they pass**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_train_and_watch.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add scripts/flapping_rl/train_and_watch.py \
        scripts/flapping_rl/eval_path_tracking_checkpoint.py \
        source/isaaclab_tasks/isaaclab_tasks/direct/flapping_bot/agents/rsl_rl_ppo_straightflight_cfg.py \
        docs/analysis/flapping_rl/estimated_state_wind_rl_baseline.md \
        tests/test_train_and_watch.py
git commit -m "feat: add staged estimated-state wind rl training recipe"
```

---

### Task 7: Verification gate before calling the stack RL-ready

**Files:**
- Validate: `scripts/flapping_px4/fly_path_mission.py`
- Validate: `scripts/flapping_rl/eval_path_tracking_checkpoint.py`
- Validate: `scripts/flapping_rl/watch_and_eval.py`
- Create: `docs/analysis/flapping_rl/estimated_state_wind_rl_readiness_report.md`

**Step 1: Run the non-RL guardrail checks**

Run:

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py \
  --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 \
  --phase level_straight \
  --steps 2200 \
  --headless
```

Run:

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py \
  --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 \
  --phase level_turn \
  --steps 2400 \
  --headless
```

Run:

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py \
  --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 \
  --phase level_loiter \
  --steps 2800 \
  --headless
```

Expected: the controller-only baseline still works and remains the non-RL comparison anchor.

**Step 2: Run RL evaluation checkpoints on the new suites**

Run:

```bash
./isaaclab.sh -p scripts/flapping_rl/eval_path_tracking_checkpoint.py \
  --task Isaac-FlappingBot-PathTracking-DeLaurier-WeakTeacherRL-Direct-v0 \
  --eval_suite path_tracking_estimated_nowind_v1 \
  --checkpoint <best_checkpoint.pt> \
  --headless
```

Run:

```bash
./isaaclab.sh -p scripts/flapping_rl/eval_path_tracking_checkpoint.py \
  --task Isaac-FlappingBot-PathTracking-DeLaurier-PureRL-Direct-v0 \
  --eval_suite path_tracking_estimated_wind_v1 \
  --checkpoint <best_checkpoint.pt> \
  --headless
```

**Step 3: Write the readiness report**

Record:

- whether estimated-state training remains stable
- how much degradation wind introduces
- whether pure-RL continuation keeps acceptable straight / turn / loiter completion
- which failure mode appears first: altitude loss, lateral divergence, or termination

**Step 4: Commit**

```bash
git add docs/analysis/flapping_rl/estimated_state_wind_rl_readiness_report.md
git commit -m "docs: record estimated-state wind rl readiness"
```

---

## Acceptance criteria

Do not call the stack ready for large RL training until all of the following hold:

- Non-RL path-tracking teacher baseline still passes the current DeLaurier regression checks.
- RL actor observation can run in `estimated` mode without NaNs or shape drift.
- Teacher and actor no longer rely on wind truth in realism mode.
- RL action execution includes nonzero actuator realism.
- Estimated-state no-wind training can produce a stable checkpoint.
- The same policy family still functions under mild-to-moderate wind evaluation.

## Recommended execution order

1. Task 1
2. Task 2
3. Task 3
4. Task 5
5. Task 4
6. Task 6
7. Task 7

Reason for this order:

- observation realism and teacher-state semantics are the first blockers
- evaluation must be fixed before expensive training
- actuator realism should be in place before the main training recipe is frozen

