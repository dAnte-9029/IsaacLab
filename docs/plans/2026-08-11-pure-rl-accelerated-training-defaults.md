# PureRL Accelerated Training Defaults Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the validated 256/16/500/25 CPU-native configuration the default for measured PureRL training.

**Architecture:** Keep task-aware defaults in `train_and_watch.py`, which is already the project authority for
CPU-native measured PureRL launches. Resolve defaults only when the corresponding CLI option is omitted, so
explicit overrides and non-measured tasks retain their existing behavior.

**Tech Stack:** Python, argparse, pytest, IsaacLab launcher wrapper.

---

### Task 1: Freeze the launcher contract

**Files:**

- Modify: `tests/test_train_and_watch.py`
- Test: `tests/test_train_and_watch.py`

**Step 1: Write the failing tests**

Update the measured PureRL default test to require `--num_envs 256`, `--max_iterations 500`,
`agent.save_interval=25`, and `agent.algorithm.num_mini_batches=16`. Keep the explicit-override tests and extend
the non-measured default test to require the legacy 512/2000/100 behavior with no minibatch override.

**Step 2: Verify RED**

Run:

```bash
TERM=xterm ./isaaclab.sh -p -m pytest -q \
  tests/test_train_and_watch.py::test_measured_pure_rl_defaults_to_accelerated_cpu_training \
  tests/test_train_and_watch.py::test_non_measured_task_keeps_legacy_launcher_defaults
```

Expected: the measured test fails on the old 64/2000/100/no-minibatch defaults.

### Task 2: Implement task-aware defaults

**Files:**

- Modify: `scripts/flapping_rl/train_and_watch.py`
- Test: `tests/test_train_and_watch.py`

**Step 1: Implement the minimal resolution logic**

Define measured defaults 256/500/25/16 and legacy defaults 512/2000/100. Make the parser use `None` for the
task-aware options. Resolve each value when building the train command, validate it is positive, and retain any
explicit CLI value.

**Step 2: Verify GREEN**

Run the two focused tests, followed by all `tests/test_train_and_watch.py` tests.

### Task 3: Synchronize authoritative documentation

**Files:**

- Modify: `docs/PROJECT_STATE.md`
- Modify: `docs/handoffs/2026-08-10-pure-rl-curriculum2.md`
- Create: `docs/decisions/ADR-2026-08-11-pure-rl-accelerated-training-defaults.md`
- Modify: `docs/audits/2026-08-11-pure-rl-cpu-native-training-acceleration.md`

**Step 1: Record the new default decision**

Create a superseding ADR rather than rewriting the accepted sample-equivalent cadence ADR. Record that only
measured PureRL launcher defaults change and explicit overrides remain available.

**Step 2: Correct operational commands**

Update C1/C2 commands and current-state language from 64/4/2000/100 to 256/16/500/25. Mark historical evidence
as historical rather than changing its recorded parameters.

### Task 4: Verify and review

**Files:**

- Verify all files above.

**Step 1: Run focused tests and compilation**

```bash
TERM=xterm ./isaaclab.sh -p -m pytest -q tests/test_train_and_watch.py
TERM=xterm ./isaaclab.sh -p -m py_compile scripts/flapping_rl/train_and_watch.py
```

**Step 2: Inspect repository state**

Run `git diff --check`, inspect the scoped diff, and report unrelated pre-existing worktree changes separately.

**Step 3: Commit boundary**

Do not commit automatically. The repository instructions require explicit user authorization before committing.
