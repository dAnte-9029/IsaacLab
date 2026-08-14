# PureRL C3a retention-aware sampling audit

## Scope and status

This audit records the completed 200-iteration, seed-0 C3a experiment that combined the default-disabled
retention-aware task scheduler with actor-only distillation coefficient `0.05`. The run started weights-only from
promoted C2c `model_550.pt`; it did not restore optimizer or iteration state. C3a remains unpromoted.

Run directory:

```text
logs/rsl_rl/flapping_bot_straight_flight/
2026-08-14_09-18-54_pure_rl_c3a_adaptive_sampling_distill005_c2c_seed0_200iter
```

Training completed 200 PPO iterations and 2,457,600 transitions. At the final iteration, the scheduler reported
C1/C2c/C3a probabilities `0.2413/0.4494/0.3093` and strong-climb probability `0.5757`. These online signals are
diagnostic only; promotion decisions below use the frozen fresh-process grids.

## Fixed-grid evidence

Each saved candidate was evaluated in three separate CPU-native processes at 1/480 s: the 96-case C3a v2 grid,
the 16-case C1 v2 retention grid, and the 112-case C2c v2 grid. The C1 tail-limit gate is at most 10 percent. The
C2c hard gates include at least 95 percent overall survival and at least 90 percent climb success.

| Iteration | C3a success / score | C1 tail limit | C1 gate | C2c survival | C2c climb | `+12` success | C2c gate |
| ---: | ---: | ---: | :---: | ---: | ---: | ---: | :---: |
| 50 | 100% / 70.09 | 22.22% | fail | 83.93% | 62.50% | 2/16 | fail |
| 75 | 100% / 67.15 | 16.99% | fail | 91.07% | 79.17% | 6/16 | fail |
| 100 | 100% / 65.37 | 13.00% | fail | 91.96% | 81.25% | 7/16 | fail |
| 125 | 100% / 63.93 | 10.21% | fail | 91.07% | 79.17% | 7/16 | fail |
| 150 | 100% / 69.29 | 4.39% | pass | 92.86% | 83.33% | 8/16 | fail |
| 175 | 100% / 70.18 | 5.10% | pass | 93.75% | 85.42% | 9/16 | fail |
| 199 | 100% / 71.95 | 4.91% | pass | 91.96% | 81.25% | 7/16 | fail |

## Interpretation

The scheduler recovered the C1 tail-action gate by iteration 150 while preserving 100 percent C3a success. C2c
improved through iteration 175, but it never passed the frozen survival or climb gates and regressed at iteration
199. `model_175.pt` is the best three-suite diagnostic compromise in this run, but it is not a promotion candidate.
No adjacent checkpoint pair has complete passing C3a, C1, and C2c evidence.

The fixed-grid result also shows why online scheduler telemetry cannot replace retention evaluation. The scheduler
reduced strong-climb allocation late in the run while the held-out `+12` degree slice still had seven to nine tilt
terminations. A follow-up must align the online strong-climb success signal more closely with the frozen C2c
episode-success contract before another long run.

## Validation

The final worktree passed 224 focused non-Isaac unit and contract tests. The C3 geometry/rehearsal runtime gate and
the adaptive-sampling update runtime gate each passed in separate fresh CPU-native Isaac Sim processes. Python
compilation and `git diff --check` also passed.

## Limitations and decision

This is one seed and one scheduler/distillation setting. It demonstrates partial retention recovery, not C3a
promotion, convergence, robustness, or real-flight validity. Do not start C3b from any checkpoint in this run.
