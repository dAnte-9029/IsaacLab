# PureRL Tail-Servo Backend Gate Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add and run an isolated CPU-native versus phase-matched GPU-implicit tail-servo step-response gate.

**Architecture:** A new parent/worker runner launches four fresh Isaac processes and writes aligned tail command, joint-state and tail-wrench traces. Pure comparison logic reuses the existing step-response summarizer and applies the thresholds frozen in the approved design without changing either plant configuration.

**Tech Stack:** Python, NumPy, PyTorch, Isaac Lab direct environments, pytest.

---

### Task 1: Lock the pure paired-comparison contract

**Files:**
- Modify: `scripts/flapping_rl/pure_rl_backend_diagnostics.py`
- Modify: `tests/test_pure_rl_backend_diagnostics.py`

1. Write failing tests for aligned command validation, per-surface angle metrics,
   transition-timing deltas, aerodynamic tail-moment metrics and frozen gate
   failures.
2. Run the focused test and confirm it fails because the comparison API is
   absent.
3. Implement the smallest pure comparison function and constants needed by the
   tests, reusing `summarize_step_response`.
4. Run the focused test and confirm it passes.

### Task 2: Add the fresh-process runner contract

**Files:**
- Create: `scripts/flapping_rl/run_pure_rl_tail_servo_gate.py`
- Create: `tests/test_run_pure_rl_tail_servo_gate.py`

1. Write failing tests for the four worker jobs, CPU/GPU device assignments,
   explicit phase-matched selection, fixed modes, and overwrite refusal.
2. Run the focused test and confirm it fails because the runner is absent.
3. Implement parent argument parsing, worker command construction, provenance,
   worker result collection and fail-closed manifests.
4. Run the focused test and confirm it passes.

### Task 3: Implement the isolated Isaac worker

**Files:**
- Modify: `scripts/flapping_rl/run_pure_rl_tail_servo_gate.py`
- Modify: `tests/test_run_pure_rl_tail_servo_gate.py`

1. Write failing source-contract tests for the exact CPU/GPU config classes,
   `480/60 Hz` cadence, three direct tail channels, `20 deg` sequence, and
   `0/8 m/s` aerodynamic modes.
2. Run the focused test and confirm the expected contract failure.
3. Implement fixed-root environment configuration, the three-surface command
   sequence, finite checks, aligned trace capture and per-worker summaries.
4. Run the focused tests, `py_compile`, and `git diff --check`.

### Task 4: Execute and diagnose the physical gate

**Files:**
- Output only: `/home/zn/temp/pure_rl_tail_servo_gate_20260811_phase_matched_v1/`

1. Run the parent runner in `env_isaaclab` with `cuda:0` and headless mode.
2. Inspect all worker manifests and the paired summary.
3. Report whether the no-aero and loaded cases pass independently and identify
   the first failing component boundary.
4. Stop without tuning gains or changing the plant.

### Task 5: Final verification and handoff

**Files:**
- Review all files changed in Tasks 1-3.

1. Run all affected backend diagnostic, runner, drive-gate and paired-plant
   tests.
2. Run `py_compile` on affected Python files and `git diff --check`.
3. Review `git status` and the final diff, preserving the unrelated asset
   configuration change.
4. Report behavior, evidence, artifacts, limitations and baseline preservation.

No commit is included because repository instructions require explicit user
authorization before committing.
