# PureRL Frequency Governor and Retention Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a frequency-only physical slew governor and Hz/s reward term, then add stage-wise straight-flight rehearsal and a fail-closed retention matrix.

**Architecture:** Keep pure Tensor contracts separate from the Isaac environment. Integrate the frequency governor only in the measured PureRL configuration, preserve direct tail actions, and expose physical telemetry. Extend the existing mission curriculum sampler with stage-specific rehearsal probabilities. Build retention analysis as an offline, deterministic CSV/JSON utility keyed by explicit curriculum stages.

**Tech Stack:** Python, PyTorch, Isaac Lab direct environments, pytest, CSV/JSON command-line utilities.

---

### Task 1: Pure Tensor frequency governor

**Files:**
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/action_contract.py`
- Test: `tests/test_flapping_action_contract.py`

Add failing tests for asymmetric physical slew limits, no overshoot, batch/device/dtype preservation, and invalid inputs. Implement a typed result and pure Tensor governor.

### Task 2: Physical Hz/s reward

**Files:**
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/pure_rl_reward.py`
- Test: `tests/test_pure_rl_reward_contract.py`

Replace the normalized frequency-delta penalty with a squared physical Hz/s term. Preserve normalized tail-delta regularization and term-by-term telemetry.

### Task 3: Measured PureRL integration

**Files:**
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`
- Modify: `scripts/flapping_rl/pure_rl_eval_common.py`
- Modify dependent validation scripts and tests under `scripts/flapping_rl/` and `tests/`

Add opt-in governor configuration and state buffers. Enable the 2.0 Hz/s symmetric default only for measured PureRL, reset it from the configured reset frequency, and expose requested/applied frequency, physical slew, and limited fraction. Update evaluation telemetry without removing normalized full-action delta diagnostics.

### Task 4: Stage-wise rehearsal

**Files:**
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py`
- Test: relevant path-tracking curriculum tests under `tests/`

Add a stage-aligned straight-rehearsal probability tuple, validate it, and allow straight-flight samples at every opted-in PureRL stage. Preserve non-PureRL and explicitly configured baselines.

### Task 5: Retention matrix

**Files:**
- Add: `scripts/flapping_rl/pure_rl_retention.py`
- Add: `scripts/flapping_rl/build_pure_rl_retention_matrix.py`
- Add: `tests/test_pure_rl_retention.py`

Implement strict lower-triangular completeness and uniqueness checks, diagonal score baselines, score-drop diagnostics, and all-prior-stage retention gates. Provide CSV input and CSV/JSON output through a thin CLI.

### Task 6: Verification

Run targeted pure Tensor tests first, then environment contract tests and relevant project regression tests in `env_isaaclab` through `./isaaclab.sh`. Run `git diff --check`, inspect the complete diff, and report any Isaac runtime tests that were not run.
