# RL Baseline Alignment And Smoke Plan Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align the RL launcher and default evaluation behavior with the current `estimated + synthetic + 0.95kg` baseline so short smoke training can start without manual config drift.

**Architecture:** Keep the existing RL pipeline intact. Make only small launcher/eval hygiene changes: let `train_and_watch.py` pass a mass override into training, default primitive path-tracking tasks to the primitive estimated suite, and update tests that still assume truth-default path-tracking evaluation.

**Tech Stack:** Python 3.11, argparse, Isaac Lab RL entrypoints, pytest.

---

### Task 1: Add Mass Override Support To `train_and_watch.py`

**Files:**
- Modify: `scripts/flapping_rl/train_and_watch.py`
- Test: `tests/test_train_and_watch.py`

**Step 1: Write the failing test**

Add a test that builds a train command with `mass_kg_override=0.95` and asserts the generated command includes `total_mass_kg_override=0.95`.

**Step 2: Run test to verify it fails**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_train_and_watch.py -k mass_override -q
```

Expected: FAIL because `train_and_watch.py` does not yet accept or forward the mass override.

**Step 3: Write minimal implementation**

Add one CLI flag, thread it into `_build_train_cmd`, and only append the Hydra override when provided.

**Step 4: Run test to verify it passes**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_train_and_watch.py -k mass_override -q
```

Expected: PASS.

**Step 5: Commit**

Deferred unless explicitly requested.

### Task 2: Default Primitive Path Tasks To Primitive Estimated Eval Suites

**Files:**
- Modify: `scripts/flapping_rl/train_and_watch.py`
- Modify: `scripts/flapping_rl/path_tracking_eval_common.py`
- Test: `tests/test_train_and_watch.py`
- Test: `tests/test_watch_and_eval.py`
- Test: `tests/test_path_tracking_eval_common.py`

**Step 1: Write the failing tests**

Add tests asserting:
- primitive path-tracking tasks resolve `straight_standard` to `path_tracking_estimated_primitives_nowind_v1`
- generic path-tracking tasks resolve `straight_standard` to `path_tracking_estimated_nowind_v1`

**Step 2: Run tests to verify they fail**

Run:

```bash
./isaaclab.sh -p -m pytest \
  tests/test_train_and_watch.py \
  tests/test_watch_and_eval.py \
  tests/test_path_tracking_eval_common.py -q
```

Expected: FAIL on the new primitive-resolution assertions and on the existing stale truth-resolution assertions.

**Step 3: Write minimal implementation**

Extend the suite resolver helper so primitive path tasks map to the primitive estimated suite, and other path tasks keep mapping to the generic estimated suite.

**Step 4: Run tests to verify they pass**

Run the same command as Step 2.

Expected: PASS.

**Step 5: Commit**

Deferred unless explicitly requested.

### Task 3: Run Targeted Verification For Phase 0 RL Readiness

**Files:**
- Check: `scripts/flapping_rl/train_and_watch.py`
- Check: `scripts/flapping_rl/path_tracking_eval_common.py`
- Check: `tests/test_train_and_watch.py`
- Check: `tests/test_eval_suites.py`
- Check: `tests/test_rl_teacher_guidance.py`
- Check: `tests/test_watch_and_eval.py`
- Check: `tests/test_path_tracking_eval_common.py`

**Step 1: Run targeted verification**

Run:

```bash
./isaaclab.sh -p -m pytest \
  tests/test_train_and_watch.py \
  tests/test_eval_suites.py \
  tests/test_rl_teacher_guidance.py \
  tests/test_watch_and_eval.py \
  tests/test_path_tracking_eval_common.py -q
```

Expected: PASS.

**Step 2: Record outcome**

Use the passing result as the repo-side readiness check before any short smoke RL run.

**Step 3: Commit**

Deferred unless explicitly requested.
