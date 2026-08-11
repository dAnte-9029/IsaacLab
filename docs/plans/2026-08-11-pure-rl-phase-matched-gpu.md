# PureRL Phase-Matched GPU Implicit Drive Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an explicit GPU implicit wing-drive variant whose frequency/phase reference is identical to the
CPU-native reference, then rerun the unchanged drive and paired-plant gates.

**Architecture:** Reuse `IdealFrequencyPhaseState` and `step_ideal_frequency_phase` for reference generation while
retaining `IdealCoupledFlappingBotCfg` for GPU-compatible implicit joint tracking. Preserve the current GPU task as
the baseline and expose separate phase-matched C1/C2a task IDs.

**Tech Stack:** Python, PyTorch tensor helpers, Isaac Lab config classes and Gymnasium registration, pytest, Isaac
Sim CPU-native and direct-GPU fresh-process evaluations.

---

### Task 1: Freeze the new drive and task contracts

**Files:**
- Modify: `tests/test_pure_rl_gpu_implicit_task_contract.py`
- Modify: `tests/test_pure_rl_longitudinal_task_registration.py`
- Modify: `tests/test_run_pure_rl_paired_plant_gate.py`

**Step 1: Write failing tests**

Require:

- `PHASE_MATCHED_IMPLICIT_WING_DRIVE` is a valid, distinct variant;
- new C1/C2a configs use CUDA, replicated physics, `IdealCoupledFlappingBotCfg`, actual per-wing-link aerodynamics,
  and the new drive variant;
- existing GPU configs still select `IDEAL_COUPLED_WING_DRIVE`;
- new task IDs map to the new configs;
- paired-plant worker jobs can explicitly select the phase-matched task pair.

**Step 2: Run the focused tests and verify RED**

Run:

```bash
source /home/zn/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
TERM=xterm ./isaaclab.sh -p -m pytest -q \
  tests/test_pure_rl_gpu_implicit_task_contract.py \
  tests/test_pure_rl_longitudinal_task_registration.py \
  tests/test_run_pure_rl_paired_plant_gate.py
```

Expected: failure because the constant, configs, task IDs, and backend selection do not exist.

### Task 2: Implement the phase-matched implicit reference path

**Files:**
- Modify: `source/flapping_bot/flapping_bot/assets/ideal_coupled_drive.py`
- Modify: `source/flapping_bot/flapping_bot/assets/__init__.py`
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/__init__.py`
- Modify: `source/flapping_bot/flapping_bot/__init__.py`

**Step 1: Add the minimal variant constant**

Add `PHASE_MATCHED_IMPLICIT_WING_DRIVE = "phase_matched_implicit_drive"` to `WING_DRIVE_VARIANTS` and exports.
Reuse `IdealCoupledFlappingBotCfg`; do not create new actuator gains.

**Step 2: Reuse the CPU phase state**

Include the new variant wherever the environment:

- validates measured multibody drives and actual-motion aero compatibility;
- allocates `IdealInverseDynamicsPhaseDriveConfig` and `IdealFrequencyPhaseState`;
- maps governor output to `_phase_target_frequency_hz` and `_phase_throttle`;
- calls `step_ideal_frequency_phase` and obtains `q_cmd`, `qd_cmd`, and `qdd_cmd`;
- sends implicit position/velocity targets using the current `IDEAL_COUPLED_WING_DRIVE` branch;
- commits the returned phase/frequency state after each physics step;
- resets the shared phase state.

Do not enter the generalized inverse-dynamics or native-holonomic branches.

**Step 3: Add phase-matched config classes**

Subclass the existing GPU C1 config, changing only `wing_drive_variant`. Add the C2a subclass by changing only
`pure_rl_longitudinal_stage_id` through inheritance.

**Step 4: Run focused contract tests and verify GREEN**

Run the Task-1 pytest command. Expected: all pass.

### Task 3: Register explicit phase-matched tasks

**Files:**
- Modify: `source/isaaclab_tasks/isaaclab_tasks/direct/flapping_bot/__init__.py`
- Modify: `tests/test_pure_rl_longitudinal_task_registration.py`

**Step 1: Add C1 and C2a Gymnasium registrations**

Register:

- `Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-GpuPhaseMatched-Direct-v0`;
- `Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C2a-GpuPhaseMatched-Direct-v0`.

Both reuse `FlappingBotStraightFlightPPORunnerCfg`. Existing registrations remain unchanged.

**Step 2: Run registration tests**

Expected: both new mappings and all old mappings pass.

### Task 4: Make paired evaluation select the candidate explicitly

**Files:**
- Modify: `scripts/flapping_rl/pure_rl_eval_common.py`
- Modify: `scripts/flapping_rl/pure_rl_backend_eval_worker.py`
- Modify: `scripts/flapping_rl/run_pure_rl_paired_plant_gate.py`
- Modify: corresponding focused tests under `tests/`

**Step 1: Extend backend task recognition**

Recognize both old and phase-matched GPU task IDs as `gpu_implicit_candidate`, without changing CPU task IDs.

**Step 2: Add an explicit candidate selector to the paired-gate parent**

Add `--gpu-candidate {baseline,phase_matched}` with `baseline` as the backward-compatible default. Build phase-
matched worker jobs only when explicitly selected, and record the selection in the manifest.

**Step 3: Run focused evaluator tests**

Expected: old job commands remain unchanged by default; phase-matched selection uses both new task IDs.

### Task 5: Run inexpensive validation

**Files:** no new files.

**Step 1: Run focused unit and contract tests**

Run all affected PureRL GPU, backend worker, diagnostics, and paired-gate tests.

**Step 2: Run syntax and whitespace checks**

Run `py_compile` for changed scripts and `git diff --check`.

Expected: all pass.

### Task 6: Run the unchanged physical gates

**Files:** generated artifacts under `/home/zn/temp`; do not track them.

**Step 1: Run the fixed-root 4/5 Hz gate with the phase-matched candidate**

Use fresh CPU and GPU processes, seed 0, 480 Hz, 1 s. Expected: all frozen drive thresholds pass.

**Step 2: Run the five-case paired-plant gate**

Use the same checkpoint, case IDs, action capture/replay, seed, and thresholds as stage 3, with only the selected
GPU task changed. Expected acceptance: every case passes every unchanged threshold.

**Step 3: Stop and report**

If the paired gate fails, report the residual metrics and stop before changing acceleration source, gains, or
thresholds. If it passes, record the new authoritative artifact and request approval before stage 4.

No commit step is included because repository instructions prohibit automatic commits without explicit user
authorization.
