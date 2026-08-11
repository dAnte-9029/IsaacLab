# PureRL CPU-Native Training Acceleration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Promote the existing C2a CPU-native candidate if it retains C1, then measure a faster CPU-native PPO configuration without changing the plant or total benchmark work.

**Architecture:** Keep CPU PhysX, the native holonomic constraint, 480 Hz physics, 60 Hz policy and current curriculum contracts authoritative. Add only the missing C1-suite retention adapter and explicit launcher controls needed to disable concurrent evaluation and preserve PPO minibatch size while scaling environments. Evaluate the current candidates first, then compare 64/128/256 environments at equal transitions, optimizer steps and minibatch size.

**Tech Stack:** Python 3.11, pytest, Isaac Lab, RSL-RL PPO, TensorBoard event files, CPU PhysX.

---

### Task 1: Add the C1 suite-to-retention adapter

**Files:**
- Modify: `scripts/flapping_rl/pure_rl_longitudinal_promotion.py`
- Modify: `tests/test_pure_rl_longitudinal_promotion.py`

**Step 1: Write the failing tests**

Add tests showing that a current `pure_rl_curriculum1_v2` suite row becomes a `c1_straight` retention row, with `timeout_rate` used as episode success rate, and that non-suite/nonfinite evidence fails closed.

**Step 2: Verify RED**

Run:

```bash
source /home/zn/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
./isaaclab.sh -p -m pytest tests/test_pure_rl_longitudinal_promotion.py -q
```

Expected: failure because `build_c1_retention_row` does not exist.

**Step 3: Implement the minimal adapter**

Implement `build_c1_retention_row()` and `build_c1_source_baseline()` using only the checkpoint, suite identity, timeout/termination rates, score, mean cross-track error and mean height error. Reject missing, nonfinite or non-suite input.

**Step 4: Verify GREEN**

Run the same targeted test and expect all tests to pass.

### Task 2: Add an explicit train-only launcher path

**Files:**
- Modify: `scripts/flapping_rl/train_and_watch.py`
- Modify: `tests/test_train_and_watch.py`

**Step 1: Write failing tests**

Add tests for `--train-only` behavior selection and for an optional positive `--agent-num-mini-batches` value being forwarded as `agent.algorithm.num_mini_batches=<N>`. Add a rejection test for nonpositive minibatch counts.

**Step 2: Verify RED**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_train_and_watch.py -q
```

Expected: failures because the new controls do not exist.

**Step 3: Implement the minimal controls**

When `--train-only` is selected, launch training and write curriculum provenance normally, but do not create or launch a watcher and do not perform final evaluation. Forward only the explicit positive minibatch override; preserve all existing defaults when omitted.

**Step 4: Verify GREEN and regression**

Run both targeted test modules and `git diff --check`.

### Task 3: Evaluate current C2a candidates on C1 and decide promotion

**Files:**
- Create generated artifacts only under: `/home/zn/temp/pure_rl_c2a_retention_20260811/`
- Read: `logs/rsl_rl/flapping_bot_straight_flight/2026-08-10_19-45-14_pure_rl_c2a_seed0/eval/summary.csv`

**Step 1: Select adjacent C2a passes**

Use `model_1400.pt` and `model_1500.pt`; both pass the 80-case C2a grid and are exactly 100 PPO iterations apart. Include the source C1 `model_1300.pt` to establish a same-contract baseline.

**Step 2: Run one fresh-process C1 evaluation**

Create a temporary evaluation directory containing symlinks to those three checkpoints, then run `watch_and_eval.py --once --no_saved_cfg` on the authoritative C1 task, 16 environments and the registered C1 suite. Keep all generated output outside the repository.

**Step 3: Build promotion evidence**

Read the three C1 suite rows, convert them with the new adapter, combine them with the existing C2a suite rows and run `evaluate_longitudinal_promotion()`. Write the inputs and result as JSON under the temporary artifact root.

**Step 4: Gate continuation**

If promotion passes, record `model_1500.pt` as the C2a source for C2b. If it fails, report the exact C1 metric and do not start C2b.

### Task 4: Benchmark CPU-native PPO environment scaling

**Files:**
- Create generated runs under: `logs/rsl_rl/flapping_bot_straight_flight/`
- Create summary artifact: `/home/zn/temp/pure_rl_cpu_native_benchmark_20260811/summary.json`

**Step 1: Freeze comparable work**

Use C2a, seed 0, the source C1 `model_1300.pt`, 48 steps per environment, four PPO epochs, and 245,760 transitions per run. Preserve 768 samples per minibatch and 1,280 optimizer steps:

| Environments | Iterations | Minibatches | Transitions | Optimizer steps |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 80 | 4 | 245,760 | 1,280 |
| 128 | 40 | 8 | 245,760 | 1,280 |
| 256 | 20 | 16 | 245,760 | 1,280 |

Use `--train-only`, headless mode and a save interval larger than the run.

**Step 2: Run each case in a fresh process**

Run cases sequentially so they do not contend for CPU resources. Do not launch evaluators during timing.

**Step 3: Analyze TensorBoard timing**

Discard startup outliers, then report median and p95 `Perf/total_fps`, collection time and learning time, total wall time, transitions per second, and peak resident memory if available.

**Step 4: Select a candidate, not a new default**

Choose the fastest stable configuration by fixed-work wall time. Do not change production defaults until that environment count completes a short convergence/retention validation because larger rollout batches change policy update cadence.

### Task 5: Record results and hand off the next curriculum action

**Files:**
- Modify: `docs/PROJECT_STATE.md`
- Create: `docs/audits/2026-08-11-pure-rl-cpu-native-training-acceleration.md`

**Step 1: Record evidence**

Document governor state, C2a promotion outcome, benchmark commands, fixed-work contract, timing results and the selected candidate configuration.

**Step 2: Preserve boundaries**

State explicitly that CPU-native plant behavior did not change, GPU implicit remains rejected, and environment scaling is not a promoted training default without convergence evidence.

**Step 3: Validate final diff**

Run targeted tests, `git diff --check`, and inspect `git status`. Do not commit because the user has not requested a commit.
