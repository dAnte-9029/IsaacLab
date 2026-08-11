# PureRL GPU Implicit Zero-Shot Qualification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Evaluate the promoted CPU-native PureRL `model_1500.pt` zero-shot on explicit GPU-implicit C1 and C2a tasks, with conditional action replay and artifacts that explain failures.

**Architecture:** Add two explicit screening-only GPU task configs that preserve the shared policy contract while replacing only the backend. A fresh-process worker performs closed-loop or fixed-action rollout, a pure-Python diagnostic module selects and compares cases, and a small orchestrator runs CPU/GPU stages and writes the final report. Existing CPU-native tasks and promotion gates remain authoritative and unchanged.

**Tech Stack:** Python 3.11, PyTorch, Isaac Lab DirectRLEnv/configclass, Gymnasium, RSL-RL inference, NumPy NPZ, CSV/JSON, Matplotlib, pytest.

---

### Task 1: Add explicit GPU-implicit C1 and C2a task contracts

**Files:**
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/__init__.py`
- Modify with explicit user approval: `source/isaaclab_tasks/isaaclab_tasks/direct/flapping_bot/__init__.py`
- Modify: `scripts/flapping_rl/pure_rl_eval_common.py`
- Create: `tests/test_pure_rl_gpu_implicit_task_contract.py`
- Modify: `tests/test_pure_rl_longitudinal_task_registration.py`
- Modify: `tests/test_pure_rl_eval_common.py`

**Step 1: Write failing configuration tests**

Require two new configs:

```python
FlappingBotStraightFlightDeLaurierMeasuredPureRLGpuImplicitEnvCfg
FlappingBotStraightFlightDeLaurierMeasuredPureRLC2aGpuImplicitEnvCfg
```

Assert that both preserve the shared 480 Hz/60 Hz, four-action, 555-observation, 0--5 Hz, 2 Hz/s governor, actual
tail-joint, reward, and termination contract. Assert that only the backend selects CUDA, replicated physics,
`ideal_coupled_drive`, `actual_per_wing_link`, measured mass properties, and
`retain_accelerations=False`. Assert that C2a selects only `pure_rl_longitudinal_stage_id="c2a"`.

**Step 2: Run RED**

```bash
source /home/zn/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
TERM=xterm ./isaaclab.sh -p -m pytest -q \
  tests/test_pure_rl_gpu_implicit_task_contract.py \
  tests/test_pure_rl_longitudinal_task_registration.py \
  tests/test_pure_rl_eval_common.py
```

Expected: failures because the GPU config classes and task IDs do not exist.

**Step 3: Implement the minimal explicit configs**

Subclass the measured PureRL config so the policy contract remains one source of truth, then override the complete
backend surface explicitly:

```python
@configclass
class FlappingBotStraightFlightDeLaurierMeasuredPureRLGpuImplicitEnvCfg(
    FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg
):
    wing_drive_variant: str = IDEAL_COUPLED_WING_DRIVE
    wing_aero_coupling_mode: str = ACTUAL_PER_WING_LINK
    # Explicit CUDA SimulationCfg, replicated scene, and IdealCoupledFlappingBotCfg
    # with retain_accelerations=False.

@configclass
class FlappingBotStraightFlightDeLaurierMeasuredPureRLC2aGpuImplicitEnvCfg(
    FlappingBotStraightFlightDeLaurierMeasuredPureRLGpuImplicitEnvCfg
):
    pure_rl_longitudinal_stage_id: str = "c2a"
```

Do not change implicit-drive stiffness, damping, effort, velocity, aerodynamic coefficients, mass, inertia,
reward weights, termination thresholds, or CPU defaults.

**Step 4: Register only the two approved screening tasks**

Add exactly:

```text
Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-GpuImplicit-Direct-v0
Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C2a-GpuImplicit-Direct-v0
```

Both reuse `FlappingBotStraightFlightPPORunnerCfg`. Update `pure_rl_eval_common.py` so task-family, stage, and backend
resolution are explicit rather than inferred from loose substring matching.

**Step 5: Run GREEN and regression contracts**

```bash
TERM=xterm ./isaaclab.sh -p -m pytest -q \
  tests/test_pure_rl_gpu_implicit_task_contract.py \
  tests/test_pure_rl_curriculum_contract.py \
  tests/test_pure_rl_longitudinal_env_contract.py \
  tests/test_pure_rl_longitudinal_task_registration.py \
  tests/test_pure_rl_eval_common.py \
  tests/test_ideal_coupled_drive_contract.py
```

Expected: all pass; existing CPU task IDs still resolve to the native configs.

**Step 6: Review checkpoint**

Run `git diff --check` and inspect only the files above. Do not commit unless the user explicitly requests it.

### Task 2: Implement pure diagnostic schemas, selection, and attribution

**Files:**
- Create: `scripts/flapping_rl/pure_rl_backend_diagnostics.py`
- Create: `tests/test_pure_rl_backend_diagnostics.py`

**Step 1: Write failing artifact-schema tests**

Define test fixtures for C1 success, C2a path failure, nonfinite failure, and closed-loop/replay disagreement.
Require validation to reject duplicate case IDs, missing backend IDs, misaligned time arrays, nonfinite required
aggregate metrics, and action sequences with width other than four.

**Step 2: Write failing deterministic-selection tests**

Test that `select_diagnostic_cases(..., maximum_cases=8)`:

- returns every failure when at most eight fail;
- otherwise covers distinct termination causes and level/climb/descent before filling by earliest failure and path
  error;
- adds at most two worst successful cases only when slots remain;
- returns identical ordering for identical inputs.

**Step 3: Write failing attribution tests**

Require evidence labels for:

```text
runtime_or_nonfinite
wing_drive_tracking
aero_load_shift
policy_feedback_shift
actuator_or_governor_limit
flight_state_instability
path_tracking_only
unattributed
```

The returned record must contain label, rank, raw evidence, first event time, closed-loop outcome, replay outcome,
and an explicit `causal_claim=false` field.

**Step 4: Run RED**

```bash
TERM=xterm ./isaaclab.sh -p -m pytest -q tests/test_pure_rl_backend_diagnostics.py
```

Expected: import failure because the module does not exist.

**Step 5: Implement minimal pure-Python logic**

Use dataclasses and mappings only; do not import Isaac Sim. Keep hard task pass/fail separate from diagnostic
ranking. Compare traces only over their common time interval, record raw deltas, and classify the closed-loop
versus replay pattern without claiming physical causality.

**Step 6: Run GREEN**

```bash
TERM=xterm ./isaaclab.sh -p -m pytest -q tests/test_pure_rl_backend_diagnostics.py
```

Expected: all pass.

**Step 7: Review checkpoint**

Run `git diff --check`. Do not commit unless requested.

### Task 3: Add a fresh-process backend evaluation worker

**Files:**
- Create: `scripts/flapping_rl/pure_rl_backend_eval_worker.py`
- Create: `tests/test_pure_rl_backend_eval_worker.py`
- Modify only if a missing tensor is demonstrated: `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`

**Step 1: Write failing CLI and trace-contract tests**

Require explicit arguments for task, backend ID, checkpoint, suite, output directory, rollout mode
(`closed_loop` or `action_replay`), selected case IDs, and optional action directory. Reject CPU task/GPU backend
mismatches and replay without aligned four-channel action files.

**Step 2: Run RED**

```bash
TERM=xterm ./isaaclab.sh -p -m pytest -q tests/test_pure_rl_backend_eval_worker.py
```

Expected: import failure because the worker does not exist.

**Step 3: Implement checkpoint and schedule setup**

Reuse `eval_suites.py`, `pure_rl_eval_common.py`, and `pure_rl_longitudinal_eval.py`. Load the saved agent config and
the checkpoint through the current RSL-RL inference path. Assert action width 4, observation width 555, the exact
fixed reset schedule, and the selected backend contract before stepping.

**Step 4: Implement closed-loop and replay stepping**

Closed loop calls the policy at 60 Hz. Replay reads one CPU action row per policy step and never calls the policy.
Both collect existing environment evaluation buffers and debug tensors. Add a new environment debug buffer only
if the required value cannot be read from `_q_cmd`, `_qd_cmd`, robot joint state, or existing `_debug_last_*`
buffers.

**Step 5: Implement bounded outputs**

Always write `manifest.json`, `summary.json`, and `per_case_metrics.csv`. Write NPZ traces and CPU action NPZ only
for requested selected cases. Sample traces at the policy rate; do not add unconditional 480 Hz logging.

**Step 6: Run GREEN and syntax checks**

```bash
TERM=xterm ./isaaclab.sh -p -m pytest -q \
  tests/test_pure_rl_backend_eval_worker.py \
  tests/test_pure_rl_backend_diagnostics.py
TERM=xterm ./isaaclab.sh -p -m py_compile \
  scripts/flapping_rl/pure_rl_backend_eval_worker.py \
  scripts/flapping_rl/pure_rl_backend_diagnostics.py
```

Expected: all pass.

**Step 7: Run a four-environment GPU checkpoint-load smoke**

Use a fresh process and the GPU C1 task for a short finite-state rollout. Require:

- checkpoint loads without missing/unexpected actor parameters;
- `env.device` is CUDA;
- ideal coupled drive and replicated physics are active;
- observation shape is `(4, 555)` and action shape is `(4, 4)`;
- wing, tail, root state, reward, and diagnostic tensors stay finite;
- process exits successfully and the worker manifest reports `completed=true`.

Do not run the full grid until this smoke passes.

**Step 8: Review checkpoint**

Run `git diff --check`. Do not commit unless requested.

### Task 4: Add the qualification orchestrator and report renderer

**Files:**
- Create: `scripts/flapping_rl/run_pure_rl_gpu_qualification.py`
- Create: `tests/test_run_pure_rl_gpu_qualification.py`

**Step 1: Write failing command-construction tests**

Require the orchestrator to launch each Isaac worker in a fresh subprocess, use the native extension only for CPU,
use CUDA only for GPU, allocate separate portable roots, and propagate nonzero worker status. Do not rely on the
repository wrapper's child exit status.

**Step 2: Write failing workflow tests**

With fake worker artifacts, require this state machine:

```text
validate checkpoint
  -> GPU C1 closed loop
  -> GPU C2a closed loop unless GPU runtime/nonfinite failure
  -> select diagnostic cases when any task case fails
  -> CPU selected-case closed loop with action capture
  -> GPU selected-case CPU-action replay
  -> offline comparison and plots
```

A task-gate failure still proceeds to diagnostics. A startup, nonfinite, or corrupt-artifact failure stops before
additional task runs and writes a failed manifest.

**Step 3: Run RED**

```bash
TERM=xterm ./isaaclab.sh -p -m pytest -q tests/test_run_pure_rl_gpu_qualification.py
```

Expected: import failure because the orchestrator does not exist.

**Step 4: Implement orchestration and artifact joins**

Join only on suite version plus case ID. Write:

```text
backend_comparison_summary.json
backend_comparison_summary.csv
per_case_metrics.csv
failure_report.json
failure_report.md
plots/<case_id>.png
```

Use one four-panel plot per selected case: flight/path state, actions and frequency, wing target/actual tracking,
and aerodynamic force/moment. Plot CPU closed loop, GPU closed loop, and GPU CPU-action replay when available.

**Step 5: Run GREEN and focused regression tests**

```bash
TERM=xterm ./isaaclab.sh -p -m pytest -q \
  tests/test_run_pure_rl_gpu_qualification.py \
  tests/test_pure_rl_backend_eval_worker.py \
  tests/test_pure_rl_backend_diagnostics.py \
  tests/test_watch_and_eval.py \
  tests/test_pure_rl_longitudinal_eval.py
```

Expected: all pass.

**Step 6: Review checkpoint**

Run `git diff --check` and inspect generated-output ignore behavior. Do not commit unless requested.

### Task 5: Execute the approved model_1500 zero-shot qualification

**Files:**
- Generated only: `logs/pure_rl_gpu_qualification/model_1500_zero_shot/`

**Step 1: Verify the checkpoint and current CPU evidence**

Require this file:

```text
/home/zn/IsaacLab/.worktrees/native-multibody-rl/logs/rsl_rl/flapping_bot_straight_flight/2026-08-10_19-45-14_pure_rl_c2a_seed0/model_1500.pt
```

Read the existing CPU C1/C2a JSON only as an aggregate reference. The orchestrator must rerun selected CPU cases
when detailed paired traces or CPU action sequences are needed.

**Step 2: Run the full qualification**

```bash
cd /home/zn/IsaacLab/.worktrees/native-multibody-rl
source /home/zn/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

TERM=xterm ./isaaclab.sh -p scripts/flapping_rl/run_pure_rl_gpu_qualification.py \
  --checkpoint logs/rsl_rl/flapping_bot_straight_flight/2026-08-10_19-45-14_pure_rl_c2a_seed0/model_1500.pt \
  --cpu-c1-reference logs/rsl_rl/flapping_bot_straight_flight/2026-08-10_19-45-14_pure_rl_c2a_seed0/eval/c1_retention_1400_1500_v2/eval/model_1500.json \
  --cpu-c2a-reference logs/rsl_rl/flapping_bot_straight_flight/2026-08-10_19-45-14_pure_rl_c2a_seed0/eval/model_1500.json \
  --cuda-device cuda:0 \
  --output-dir logs/pure_rl_gpu_qualification/model_1500_zero_shot \
  --headless
```

Expected: a completed C1 16-case grid and C2a 80-case grid, or a failed manifest plus the applicable diagnostic
workflow. No PPO optimizer or training process is started.

**Step 3: Validate artifacts**

Require:

- complete and unique per-case rows;
- unchanged C1 and C2a hard gates;
- finite aggregate metrics or an explicit runtime/nonfinite failure record;
- selected-case trace/action alignment;
- conditional replay whenever a GPU task case fails;
- every failure-report label to include raw evidence and `causal_claim=false`;
- diagnostic throughput labeled as evaluation throughput, not PPO training throughput.

**Step 4: Review results before any correction**

Classify the outcome as zero-shot pass, task failure with usable diagnostics, runtime/nonfinite failure, or
unattributed failure. Do not change reward, plant, implicit-drive gains, or policy based on the result in this task.

**Step 5: Final validation and report**

```bash
TERM=xterm ./isaaclab.sh -p -m pytest -q \
  tests/test_pure_rl_gpu_implicit_task_contract.py \
  tests/test_pure_rl_backend_diagnostics.py \
  tests/test_pure_rl_backend_eval_worker.py \
  tests/test_run_pure_rl_gpu_qualification.py
git diff --check
git status --short --branch
```

Report modified files, exact commands, test results, GPU C1/C2a outcome, diagnostic attribution, limitations,
baseline preservation, and unverified items. Do not commit generated artifacts. Do not commit source changes unless
the user explicitly requests it.
