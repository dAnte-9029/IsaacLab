# PureRL CPU-Native Training Acceleration Audit

Date: 2026-08-11

## Outcome

The CPU-native plant remains the sole PureRL training and promotion authority. The existing C2a run is now
formally promoted: `model_1400.pt` and `model_1500.pt` both pass the 80-case C2a grid and the current C1 v2
retention suite, so `model_1500.pt` is the accepted source for C2b.

For training throughput, 256 environments with 16 minibatches is now the recommended CPU-native training
configuration. At equal transitions, minibatch size, optimizer steps and checkpoint-free training work, it
achieved 2.61 times the throughput of 64 environments. A subsequent 80-iteration resumed C2a run processed
983,040 transitions and passed the approved sample-equivalent C2a plus C1 retention gate at adjacent
`model_1550.pt` and `model_1575.pt` checkpoints. The latter is the accepted source for C2b.

## C2a promotion evidence

The current C1 suite adapter maps suite `timeout_rate` to episode success rate and fails closed for missing,
non-suite or nonfinite evidence. A single fresh CPU-native evaluation process compared the C1 source and two
adjacent C2a checkpoints:

| Checkpoint | C2a survival | C2a cross-track mean (m) | C2a height mean (m) | C1 success | C1 score | C1 cross-track mean (m) | C1 height mean (m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 source `model_1300.pt` | n/a | n/a | n/a | 1.0 | 89.8650 | 0.10765 | 0.46447 |
| C2a `model_1400.pt` | 1.0 | 0.07899 | 0.07930 | 1.0 | 98.2650 | 0.09896 | 0.06279 |
| C2a `model_1500.pt` | 1.0 | 0.05140 | 0.08234 | 1.0 | 98.8331 | 0.04962 | 0.05809 |

The generated decision record is
`/home/zn/temp/pure_rl_c2a_retention_20260811/promotion_evidence.json` and reports `promoted: true`.

## Fixed-work throughput contract

All cases used C2a, seed 0, the same C1 `model_1300.pt` warm start, CPU PhysX, the native holonomic constraint,
480 Hz physics, 60 Hz policy, 48 steps per environment per iteration, four PPO epochs, headless train-only mode
and no concurrent evaluator. Each case processed 245,760 transitions, preserved a 768-sample minibatch and
performed 1,280 optimizer steps.

| Environments | Iterations | Minibatches | Median fps | p05 fps | Median collection (s) | p95 collection (s) | Median learning (s) | p95 iteration (s) | Fixed-work iteration time (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 80 | 4 | 838.0 | 817.6 | 3.556 | 3.648 | 0.107 | 3.756 | 292.812 |
| 128 | 40 | 8 | 1462.0 | 1427.5 | 4.063 | 4.167 | 0.131 | 4.302 | 167.868 |
| 256 | 20 | 16 | 2185.5 | 2143.7 | 5.256 | 5.337 | 0.402 | 5.730 | 112.204 |

The first `max(2, ceil(10 percent))` iterations were omitted from steady-state quantiles. Fixed-work time is the
sum of all reported collection and learning times. Peak RSS was not instrumented, so it is not claimed. The
machine governor reported `performance` for all 32 logical CPUs during this benchmark.

The machine-readable summary is `/home/zn/temp/pure_rl_cpu_native_benchmark_20260811/summary.json`.

## Launcher findings

At the time of this audit, `train_and_watch.py` had opt-in `--train-only`, `--agent-num-mini-batches` and
`--disable-kit-fs-watcher` controls while the watcher default remained unchanged. A wrapper defect was also exposed by the host's
exhausted inotify allocation: after parsing the RSL-RL run name, the parent waited for the run directory without
reading its child's piped stdout. The high warning volume filled the pipe and stopped the child before directory
creation. A background output forwarder now drains the pipe throughout startup and training. A regression test
writes 200 kB before creating a child run directory and confirms the wait completes.

On 2026-08-12, an authority C2b launch exposed that unchanged default as inconsistent with the benchmark
contract: a concurrent CPU-native watcher competed with training and materially increased iteration time.
`ADR-2026-08-12-pure-rl-sequential-evaluation-default.md` corrects the launcher policy. Measured PureRL tasks now
default to train-only execution, `--concurrent-eval` is the explicit opt-in, and non-measured defaults are
unchanged.

The host still has `fs.inotify.max_user_watches=65536`, all of which were occupied primarily by two VS Code file
watchers. Kit therefore still logs `errno=28` warnings. The wrapper no longer deadlocks on those warnings, but a
host-level increase remains recommended for clean logs.

## 256-environment convergence and retention gate

The validation resumed promoted C2a `model_1500.pt` on the same C2a task with 256 environments, 16 minibatches,
80 additional iterations and a 25-iteration save interval. It completed iterations 1500--1579 in 7 minutes
37 seconds. This was a same-stage resume, not a cross-stage curriculum transition. C2a and C1 were evaluated
sequentially in separate fresh CPU-native processes.

| Checkpoint | C2a score | C2a survival | Level / climb / descent success | C2a cross-track mean (m) | C2a height mean (m) | C1 score | C1 success | C1 cross-track mean (m) | C1 height mean (m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `model_1550.pt` | 86.4256 | 1.0 | 1.0 / 1.0 / 1.0 | 0.07472 | 0.08234 | 98.1575 | 1.0 | 0.11154 | 0.06581 |
| `model_1575.pt` | 87.8909 | 1.0 | 1.0 / 1.0 / 1.0 | 0.06032 | 0.07746 | 98.6661 | 1.0 | 0.07020 | 0.06115 |

Both checkpoints pass the target C2a grid and C1 retention. Their 25-iteration separation is exactly 307,200
transitions, so the promotion builder returns `promoted: true` and selects `model_1575.pt`. Machine-readable
evidence is at `/home/zn/temp/pure_rl_cpu_native_256_validation_20260811/promotion_evidence.json`; raw C2a and C1
summaries are in the sibling `c2a_eval/eval/summary.csv` and `c1_eval/eval/summary.csv` files.

## Selected configuration and boundary

Use 256 environments and 16 minibatches for subsequent CPU-native curriculum training. To preserve the sample
budget of a 64-environment, 2,000-iteration run, use 500 iterations; convert a 100-iteration save interval to 25.
Run training without a concurrent evaluator, then evaluate saved checkpoints sequentially in fresh processes.

The sample-equivalent cadence is approved by
`docs/decisions/ADR-2026-08-11-pure-rl-sample-equivalent-promotion-cadence.md`. The frozen minimum and interval are
614,400 and 307,200 transitions. Existing 64-environment evidence remains 200/100; 256-environment validation
uses 50/25 explicitly through `build_sample_equivalent_promotion_schedule()`.

The validated configuration is promoted to the measured PureRL launcher default by
`docs/decisions/ADR-2026-08-11-pure-rl-accelerated-training-defaults.md`: 256 environments, 16 minibatches, 500
iterations, a 25-iteration save interval, and sequential checkpoint evaluation as clarified by
`docs/decisions/ADR-2026-08-12-pure-rl-sequential-evaluation-default.md`. Explicit overrides and all non-measured
task defaults remain available. No plant, action, observation, reward, termination, curriculum geometry, physics
rate or policy rate changed in this work. The rejected GPU implicit route remains rejected.
