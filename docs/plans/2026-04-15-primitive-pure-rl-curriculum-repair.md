# Primitive Pure RL Curriculum Repair Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Repair primitive pure RL so later curriculum stages keep training `turn` instead of silently collapsing into `straight` vs `loiter` only, then verify the repaired curriculum with targeted tests and a smoke training run.

**Architecture:** Keep the current path-tracking env, PPO entrypoints, estimated-state defaults, and reward stack unchanged. Make one minimal behavior fix inside mission sampling: expose weighted primitive sampling, let the primitive curriculum express `turn + loiter` mixtures with optional straight rehearsal, and verify that the watcher-driven checkpoint selection still evaluates the repaired training run correctly.

**Tech Stack:** Python 3.11, IsaacLab direct envs, existing path-tracking mission sampler, RSL-RL PPO, pytest, `train_and_watch.py`, `watch_and_eval.py`.

---

## Scope

This plan covers only the narrow repair needed for primitive pure RL continuation:

- keep `estimated + synthetic + 0.95kg` as the training baseline
- preserve current reward / actor / teacher structure
- repair curriculum-driven primitive sampling so `turn` is not starved in later stages
- verify with targeted tests and one short watcher-backed smoke run

This plan does not cover:

- controller retuning
- IMU plumbing changes
- reward rewrites
- teacher-student redesign
- new evaluation backends

### Task 1: Lock In The Broken Behavior With Failing Tests

**Files:**
- Modify: `tests/test_mission_primitives.py`
- Modify: `tests/test_path_tracking_env_contract.py`

**Step 1: Write the failing tests**

Add tests that require:

- weighted primitive sampling to exclude zero-weight kinds
- curriculum modes to support `turn + loiter` mixtures during loiter stages
- primitive pure RL config to use the new mixed stage names instead of the old turn-dropping ones

**Step 2: Run tests to verify they fail**

Run:

```bash
./isaaclab.sh -p -m pytest \
  tests/test_mission_primitives.py \
  tests/test_path_tracking_env_contract.py -q
```

Expected: FAIL because the current sampler has no weighting support and the current primitive curriculum still resolves later stages to `loiter`-only or `straight + loiter`.

**Step 3: Commit**

Deferred unless explicitly requested.

### Task 2: Implement Minimal Weighted Sampling And Mixed Primitive Stages

**Files:**
- Modify: `source/flapping_bot/flapping_bot/path_tracking/mission_primitives.py`
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py`

**Step 1: Write minimal implementation**

In `mission_primitives.py`:

- add optional per-kind weights to `MissionGeneratorCfg`
- sample from enabled primitives using deterministic weighted choice
- keep default behavior backward compatible with unit weights

In `path_tracking_env.py`:

- replace the current single-segment `straight` vs `loiter` special case with weighted mission sampling
- add mixed curriculum modes that preserve `turn` in loiter stages
- keep existing explicit single-primitive eval cases valid
- update primitive pure RL stage names to the mixed modes

**Step 2: Run targeted tests**

Run:

```bash
./isaaclab.sh -p -m pytest \
  tests/test_mission_primitives.py \
  tests/test_path_tracking_env_contract.py -q
```

Expected: PASS.

**Step 3: Commit**

Deferred unless explicitly requested.

### Task 3: Run Broader Regression Verification

**Files:**
- Check: `source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py`
- Check: `source/flapping_bot/flapping_bot/path_tracking/mission_primitives.py`
- Check: `tests/test_path_tracking_rewards.py`
- Check: `tests/test_eval_suites.py`
- Check: `tests/test_train_and_watch.py`

**Step 1: Run regression tests**

Run:

```bash
./isaaclab.sh -p -m pytest \
  tests/test_mission_primitives.py \
  tests/test_path_tracking_env_contract.py \
  tests/test_path_tracking_rewards.py \
  tests/test_eval_suites.py \
  tests/test_train_and_watch.py -q
```

Expected: PASS.

**Step 2: Commit**

Deferred unless explicitly requested.

### Task 4: Run Primitive Pure RL Smoke Training With Watcher Eval

**Files:**
- Check: `scripts/flapping_rl/train_and_watch.py`
- Check: `scripts/flapping_rl/watch_and_eval.py`
- Output: `logs/rsl_rl/flapping_bot_path_tracking/<new_run_name>/`

**Step 1: Launch a short repaired run**

Run:

```bash
./isaaclab.sh -p scripts/flapping_rl/train_and_watch.py \
  --task Isaac-FlappingBot-PathTracking-DeLaurier-PrimitivePureRL-Direct-v0 \
  --run-name rl_phase2_primitive_pure_curriculum_repair_smoke \
  --train-device cuda:0 \
  --eval-device cuda:1 \
  --num-envs 64 \
  --max-iterations 100 \
  --save-interval 25 \
  --episodes 3 \
  --poll-s 30 \
  --mass-kg-override 0.95 \
  --headless
```

Expected:

- watcher produces `eval/summary.csv`
- `best_checkpoint.json` exists
- later checkpoints no longer collapse solely because `turn` disappeared from the curriculum

**Step 2: Compare against the prior unstable run**

Check whether the repaired run:

- still reaches a strong straight score
- preserves non-zero turn/loiter progress deeper into training
- does not rely on the final checkpoint being the best checkpoint

**Step 3: Record outcome**

Treat the repair as successful only if the smoke run plus watcher eval show the curriculum no longer starves `turn` in the later stages.
