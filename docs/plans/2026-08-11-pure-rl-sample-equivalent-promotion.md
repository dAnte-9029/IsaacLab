# PureRL Sample-Equivalent Promotion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Preserve the longitudinal promotion sample thresholds when CPU-native PPO scales from 64 to 256 environments, then validate the 256-environment candidate with sequential retention evaluation.

**Architecture:** Keep the existing 64-environment 200/100 defaults intact. Add one exact schedule builder based on the frozen reference transition counts, record the decision in a superseding ADR, then run a fresh 256-environment train-only validation and evaluate its checkpoints in fresh processes.

**Tech Stack:** Python 3.11, pytest, Isaac Lab, RSL-RL PPO, CPU PhysX, TensorBoard.

---

### Task 1: Add the sample-equivalent schedule contract

**Files:**
- Modify: `tests/test_pure_rl_longitudinal_promotion.py`
- Modify: `scripts/flapping_rl/pure_rl_longitudinal_promotion.py`

**Step 1:** Add failing tests for 64 -> 200/100, 128 -> 100/50 and 256 -> 50/25.

**Step 2:** Add failing tests for nonpositive inputs and a rollout size that cannot exactly represent the frozen sample interval.

**Step 3:** Run the focused tests and confirm they fail because the schedule builder is missing.

**Step 4:** Implement the minimal exact-integer schedule builder and export it.

**Step 5:** Run the focused promotion tests and confirm they pass.

### Task 2: Record the superseding decision

**Files:**
- Create: `docs/decisions/ADR-2026-08-11-pure-rl-sample-equivalent-promotion-cadence.md`
- Modify: `docs/PROJECT_STATE.md`
- Modify: `docs/audits/2026-08-11-pure-rl-cpu-native-training-acceleration.md`

**Step 1:** Record the frozen 614,400-transition minimum and 307,200-transition interval.

**Step 2:** State that 64-env remains 200/100 and 256-env uses 50/25; no plant, PPO loss or promotion metric changes.

**Step 3:** Update project state from pending decision to approved validation schedule.

### Task 3: Run the 256-environment convergence validation

**Files:**
- Generated run only: `logs/rsl_rl/flapping_bot_straight_flight/`
- Generated evidence only: `/home/zn/temp/pure_rl_cpu_native_256_validation_20260811/`

**Step 1:** Run CPU-native, train-only, 256 environments, 16 minibatches, save interval 25, warm-started from the correctly staged checkpoint.

**Step 2:** Confirm finite training telemetry and saved checkpoints at the approved cadence.

**Step 3:** Evaluate the selected adjacent checkpoints sequentially on C1 and the target longitudinal suite.

**Step 4:** Run promotion with the derived 256-environment 50/25 schedule and save the evidence JSON outside the repository.

### Task 4: Verify and report

**Files:**
- Modify: `docs/PROJECT_STATE.md`
- Modify: `docs/audits/2026-08-11-pure-rl-cpu-native-training-acceleration.md`

**Step 1:** Run the focused promotion and launcher tests, `py_compile`, JSON validation and `git diff --check`.

**Step 2:** Inspect final status and preserve all unrelated dirty work.

**Step 3:** Report the validation outcome and whether 256 environments can become the recommended CPU-native curriculum configuration.

No commit is performed without explicit user authorization.
