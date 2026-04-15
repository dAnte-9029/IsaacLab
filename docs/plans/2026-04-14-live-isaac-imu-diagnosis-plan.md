# Live Isaac IMU Diagnosis Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Find why `teacher_state_source=estimated` performs materially worse with `imu_source=isaacsim` than with `imu_source=synthetic`.

**Architecture:** Keep controller code unchanged and isolate the problem by comparing the two IMU paths under the same runtime/controller settings. Use trajectory-level telemetry to test three hypotheses: live IMU accelerometer semantics/install point mismatch, live IMU timing/phase lag, and estimator accel-gate collapse under live measurements.

**Tech Stack:** Isaac Lab rollout scripts, repo-local estimator/controller code, CSV/JSON telemetry, Python analysis.

---

### Task 1: Confirm the live-vs-synthetic data path difference

**Files:**
- Read: `source/flapping_bot/flapping_bot/px4_like/imu_provider.py`
- Read: `source/flapping_bot/flapping_bot/px4_like/isaacsim_imu_adapter.py`
- Read: `source/flapping_bot/flapping_bot/px4_like/state_estimation.py`
- Read: `scripts/flapping_px4/fly_straight_line.py`
- Read: `scripts/flapping_px4/fly_loiter.py`

**Step 1:** Verify how `synthetic` IMU is built from truth.

**Step 2:** Verify how `isaacsim` IMU is created, where it is attached, and when it is sampled.

**Step 3:** Record the candidate failure points:
- accelerometer install-point / body-origin mismatch
- IMU sample timing / phase lag
- accel-gate collapse due to live acceleration norm leaving the 1 g corridor

### Task 2: Run a straight-flight live-vs-synthetic pair on the current baseline

**Files:**
- Run: `scripts/flapping_px4/fly_straight_line.py`
- Inspect: `logs/flapping_px4/straight_line/<timestamp>/summary.json`
- Inspect: `logs/flapping_px4/straight_line/<timestamp>/trajectory_env0.csv`

**Step 1:** Run `estimated + synthetic + 0.95kg`.

**Step 2:** Run `estimated + isaacsim + 0.95kg`.

**Step 3:** Compare:
- `mean_abs_track_error_m`
- `mean_est_pos_xy_err_m`
- `mean_est_vel_xyz_err_mps`
- `mean_est_yaw_err_deg`
- `sensor_accel_norm_mps2`
- `sensor_accel_gate_lpf_norm_mps2`
- `sensor_att_corr_gain`

**Decision rule:** If live IMU pushes accel norm outside the gate corridor much more often and `sensor_att_corr_gain` collapses, the estimator is seeing less usable gravity information from live IMU than from synthetic.

### Task 3: Quantify whether the live accelerometer looks physically different, not just noisier

**Files:**
- Inspect: the two straight-flight `trajectory_env0.csv` files from Task 2

**Step 1:** Compute per-run accel norm statistics: mean, p05, p50, p95.

**Step 2:** Compute the fraction of samples inside the hard gate band `[0.9 g, 1.1 g]`.

**Step 3:** Compute the fraction of samples with non-trivial accel correction gain, e.g. `sensor_att_corr_gain > 0.01`.

**Decision rule:** If the live run has much lower in-band fraction and much lower correction-gain fraction, the problem is mainly estimator/live-accel consistency, not generic controller weakness.

### Task 4: Optional sanity check on the complex-path failure mode

**Files:**
- Inspect: `logs/flapping_px4/estimated_path_tracking_complex_suite/live_imu_spotcheck/20260413_213757/*`
- Run if needed: `scripts/flapping_px4/fly_path_mission.py`

**Step 1:** Use the straight-flight result to decide whether a path-mission rerun is necessary.

**Step 2:** If necessary, rerun one worst-seed case only to confirm the same estimator signature appears before timeout.

**Decision rule:** Only do this if the straight-flight evidence is not already decisive.

### Task 5: Write the diagnosis

**Files:**
- Summarize evidence in the final response

**Step 1:** Separate evidence-backed findings from inference.

**Step 2:** Rank the causes by likelihood.

**Step 3:** State what remains unproven and what the next smallest experiment would be if more certainty is needed.
