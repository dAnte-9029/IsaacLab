# Isaac Sim IMU Dual-Teacher Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Introduce an Isaac Sim IMU-backed estimated-state path and explicit `teacher_truth` / `teacher_estimated` modes without breaking the current non-RL teacher regression workflow.

**Architecture:** Keep the existing `SensorStateEstimator` as the only estimator output producer. Add a repo-local IMU provider abstraction whose first non-synthetic backend is Isaac Sim built-in IMU. Split teacher runtime state sourcing into explicit truth and estimated modes so controller debugging and RL realism can coexist without ambiguity.

**Tech Stack:** IsaacLab direct envs, Isaac Sim built-in IMU extension, existing `SensorStateEstimator`, PX4-like teacher controllers, pytest.

---

### Task 1: Freeze the dual-teacher runtime contract

**Files:**
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py`
- Modify: `scripts/flapping_px4/fly_straight_line.py`
- Modify: `scripts/flapping_px4/fly_loiter.py`
- Test: `tests/test_rl_teacher_guidance.py`
- Test: `tests/test_path_tracking_env_contract.py`

**Step 1: Write the failing test**

Add tests that require:

- an explicit teacher state-source enum or string field, such as `truth` and `estimated`
- `teacher_truth` to remain selectable even when estimated observation support exists
- RL-facing configs to permit `teacher_estimated` without wind truth leakage

Example:

```python
def test_teacher_state_source_accepts_truth_and_estimated():
    cfg = FlappingBotPathTrackingWeakTeacherRLEnvCfg()
    cfg.teacher_state_source = "truth"
    assert cfg.teacher_state_source == "truth"
    cfg.teacher_state_source = "estimated"
    assert cfg.teacher_state_source == "estimated"
```

**Step 2: Run test to verify it fails**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_rl_teacher_guidance.py tests/test_path_tracking_env_contract.py -q
```

Expected: FAIL because teacher state source is not yet an explicit first-class contract.

**Step 3: Write the minimal implementation**

Add config fields and helpers for:

- `teacher_state_source`
- `policy_state_source`
- `imu_source`

Constrain behavior:

- `teacher_truth` may use truth state for non-RL debugging
- `teacher_estimated` must use estimated state and must not silently fall back to truth wind

**Step 4: Run test to verify it passes**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_rl_teacher_guidance.py tests/test_path_tracking_env_contract.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py \
        source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py \
        scripts/flapping_px4/fly_straight_line.py \
        scripts/flapping_px4/fly_loiter.py \
        tests/test_rl_teacher_guidance.py \
        tests/test_path_tracking_env_contract.py
git commit -m "feat: add explicit truth and estimated teacher modes"
```

---

### Task 2: Add a repo-local IMU provider abstraction

**Files:**
- Create: `source/flapping_bot/flapping_bot/px4_like/imu_provider.py`
- Create: `tests/test_imu_provider.py`

**Step 1: Write the failing test**

Add tests that require:

- one abstract IMU provider interface
- one synthetic IMU implementation for existing behavior
- one provider output contract with fields for body rates and body acceleration

Example:

```python
def test_synthetic_imu_provider_returns_required_fields():
    provider = SyntheticImuProvider()
    meas = provider.build_from_truth(
        ang_vel_body=torch.zeros(1, 3),
        specific_force_body=torch.zeros(1, 3),
    )
    assert set(meas.keys()) == {"gyro_rad_s", "accel_mps2"}
```

**Step 2: Run test to verify it fails**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_imu_provider.py -q
```

Expected: FAIL because the provider abstraction does not exist yet.

**Step 3: Write the minimal implementation**

Create:

- `ImuMeasurement` contract or equivalent simple dict contract
- `SyntheticImuProvider`
- a small factory for selecting `synthetic` vs `isaacsim`

Keep it minimal and deterministic.

**Step 4: Run test to verify it passes**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_imu_provider.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add source/flapping_bot/flapping_bot/px4_like/imu_provider.py tests/test_imu_provider.py
git commit -m "feat: add imu provider abstraction"
```

---

### Task 3: Add the Isaac Sim IMU adapter backend

**Files:**
- Create: `source/flapping_bot/flapping_bot/px4_like/isaacsim_imu_adapter.py`
- Modify: `source/flapping_bot/flapping_bot/px4_like/imu_provider.py`
- Test: `tests/test_imu_provider.py`

**Step 1: Write the failing test**

Add tests that require:

- the Isaac Sim adapter module to import safely when Isaac runtime is unavailable
- backend selection to fall back cleanly in headless unit tests

Example:

```python
def test_imu_provider_factory_accepts_isaacsim_backend_name():
    provider = build_imu_provider("isaacsim")
    assert provider.backend_name == "isaacsim"
```

**Step 2: Run test to verify it fails**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_imu_provider.py -q
```

Expected: FAIL because the Isaac Sim backend is not implemented yet.

**Step 3: Write the minimal implementation**

Create a repo-local adapter that:

- creates or binds to one IMU sensor on the robot base body
- reads body-frame angular velocity
- reads body-frame linear acceleration or specific force
- normalizes output into the same measurement contract as the synthetic provider

Do **not** modify Isaac Sim `site-packages`.

**Step 4: Run test to verify it passes**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_imu_provider.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add source/flapping_bot/flapping_bot/px4_like/isaacsim_imu_adapter.py \
        source/flapping_bot/flapping_bot/px4_like/imu_provider.py \
        tests/test_imu_provider.py
git commit -m "feat: add isaac sim imu backend adapter"
```

---

### Task 4: Let the estimator consume external IMU measurements

**Files:**
- Modify: `source/flapping_bot/flapping_bot/px4_like/state_estimation.py`
- Create: `tests/test_sensor_state_estimator.py`

**Step 1: Write the failing test**

Add tests that require:

- estimator step supports externally supplied `gyro` and `accel` measurements
- synthetic internal IMU generation remains available as a fallback

Example:

```python
def test_estimator_prefers_external_imu_measurements_when_supplied():
    state, diag = estimator.step(
        pos_local_true=pos,
        vel_local_true=vel,
        roll_true=roll,
        pitch_true=pitch,
        yaw_true=yaw,
        ang_vel_body_true=ang_vel,
        airspeed_true=airspeed,
        imu_meas={
            "gyro_rad_s": ext_gyro,
            "accel_mps2": ext_accel,
        },
    )
    assert torch.allclose(state["ang_vel_body"], ext_gyro)
```

**Step 2: Run test to verify it fails**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_sensor_state_estimator.py -q
```

Expected: FAIL because the estimator does not yet accept external IMU measurements.

**Step 3: Write the minimal implementation**

Add an optional `imu_meas` input to estimator stepping. Behavior:

- if `imu_meas` is supplied, use it for `gyro_meas` and `accel_meas`
- if not supplied, keep the current synthetic truth-to-IMU path

Do not change output names.

**Step 4: Run test to verify it passes**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_sensor_state_estimator.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add source/flapping_bot/flapping_bot/px4_like/state_estimation.py tests/test_sensor_state_estimator.py
git commit -m "feat: allow estimator to consume external imu measurements"
```

---

### Task 5: Wire IMU source selection into non-RL teacher scripts

**Files:**
- Modify: `scripts/flapping_px4/fly_straight_line.py`
- Modify: `scripts/flapping_px4/fly_loiter.py`
- Test: `tests/test_path_tracking_baseline_suite.py`

**Step 1: Write the failing test**

Add tests that require:

- `--imu_source synthetic`
- `--imu_source isaacsim`
- `--teacher_state_source truth`
- `--teacher_state_source estimated`

to be accepted by the script argument parsers.

**Step 2: Run test to verify it fails**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_path_tracking_baseline_suite.py -q
```

Expected: FAIL because these CLI surfaces do not yet exist.

**Step 3: Write the minimal implementation**

In both scripts:

- instantiate the chosen IMU provider
- route IMU measurements into `SensorStateEstimator`
- select teacher inputs from truth or estimated state explicitly

Default conservatively:

- `imu_source=synthetic`
- `teacher_state_source=truth`

This keeps current debugging behavior unchanged unless the user opts in.

**Step 4: Run test to verify it passes**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_path_tracking_baseline_suite.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add scripts/flapping_px4/fly_straight_line.py \
        scripts/flapping_px4/fly_loiter.py \
        tests/test_path_tracking_baseline_suite.py
git commit -m "feat: expose imu source and teacher state source in teacher scripts"
```

---

### Task 6: Verify truth teacher, synthetic estimated teacher, and Isaac Sim estimated teacher

**Files:**
- Validate: `scripts/flapping_px4/fly_straight_line.py`
- Validate: `scripts/flapping_px4/fly_loiter.py`
- Create: `docs/analysis/flapping_px4/imu_dual_teacher_bringup.md`

**Step 1: Run straight-flight comparison**

Run:

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_straight_line.py \
  --state_source truth \
  --teacher_state_source truth \
  --imu_source synthetic \
  --headless
```

Run:

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_straight_line.py \
  --state_source estimated \
  --teacher_state_source estimated \
  --imu_source synthetic \
  --headless
```

Run:

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_straight_line.py \
  --state_source estimated \
  --teacher_state_source estimated \
  --imu_source isaacsim \
  --headless
```

Expected:

- all runs complete
- no NaNs
- no obvious sign mismatch in roll / pitch response

**Step 2: Run loiter comparison**

Run the same three-mode comparison using:

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_loiter.py --headless
```

with the corresponding argument combinations.

Expected:

- `teacher_truth` remains the forensic baseline
- `teacher_estimated + synthetic`
- `teacher_estimated + isaacsim`

stay qualitatively consistent and do not introduce a new instability class.

**Step 3: Write the bring-up note**

Record:

- which mode is the default forensic baseline
- whether Isaac Sim IMU changed phase lag or sign conventions
- whether estimated loiter degraded relative to the synthetic IMU path

**Step 4: Commit**

```bash
git add docs/analysis/flapping_px4/imu_dual_teacher_bringup.md
git commit -m "docs: record isaac sim imu dual teacher bringup"
```

---

### Task 7: Fold the new contract into the RL ramp

**Files:**
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py`
- Modify: `scripts/flapping_rl/eval_suites.py`
- Modify: `scripts/flapping_rl/train_and_watch.py`
- Test: `tests/test_train_and_watch.py`
- Test: `tests/test_eval_suites.py`

**Step 1: Write the failing test**

Add tests that require:

- RL envs to use `teacher_estimated` in realism mode
- eval suites to distinguish truth-forensics from estimated-realism runs

**Step 2: Run test to verify it fails**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_train_and_watch.py tests/test_eval_suites.py -q
```

Expected: FAIL because the RL ramp does not yet know about the dual-teacher contract.

**Step 3: Write the minimal implementation**

Make the RL ramp assume:

- `teacher_truth` is for debugging only
- `teacher_estimated` is the training/evaluation baseline once realism is enabled

Do not remove the truth path from the repo.

**Step 4: Run test to verify it passes**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_train_and_watch.py tests/test_eval_suites.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py \
        source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py \
        scripts/flapping_rl/eval_suites.py \
        scripts/flapping_rl/train_and_watch.py \
        tests/test_train_and_watch.py \
        tests/test_eval_suites.py
git commit -m "feat: fold dual teacher contract into rl ramp"
```

---

## Acceptance Criteria

- `teacher_truth` remains available and unchanged enough for controller forensics.
- `teacher_estimated` exists as an explicit mode, not an implicit side effect.
- Isaac Sim IMU can drive the current estimator without sign or frame breakage.
- `SensorStateEstimator` remains the single estimator output contract.
- RL realism mode no longer depends on truth wind or truth teacher state.
- Full PX4 `EKF2` remains out of scope for this phase.

## Recommended Execution Order

1. Task 1
2. Task 2
3. Task 3
4. Task 4
5. Task 5
6. Task 6
7. Task 7

Reason:

- contract first
- IMU source abstraction second
- estimator integration before rollout wiring
- teacher-script validation before RL integration

