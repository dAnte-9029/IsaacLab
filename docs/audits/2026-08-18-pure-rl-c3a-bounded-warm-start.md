# PureRL C3a bounded warm-start audit

## Scope and status

This audit records the seed-0 controlled C2c-to-C3a weights-only experiment using the bounded PPO warm-start
guard. The source was promoted C2c `model_550.pt` from
`2026-08-12_17-12-48_pure_rl_c2c_seed0_resume500_to550`. Training used the measured CPU-native plant, 256
environments, 16 mini-batches, the fixed 15/35/50 percent C1/C2c/C3a mixture, C2c strong-climb probability
`0.5`, no actor distillation, and no adaptive task sampling.

The initial run directory was:

```text
logs/rsl_rl/flapping_bot_straight_flight/
2026-08-14_15-07-43_pure_rl_c3a_bounded_warm_start_c2c_seed0_500iter
```

The original guard incorrectly applied its `0.10` actor-update limit after warm-up and stopped at iteration 288
when the update norm reached `0.100094`. After restricting that fail-closed limit to the ten warm-up updates, the
experiment resumed with optimizer state from `model_275.pt`, a fixed `5e-5` learning rate, and four learning
epochs. The terminal checkpoint is `model_499.pt` in:

```text
logs/rsl_rl/flapping_bot_straight_flight/
2026-08-14_15-51-03_pure_rl_c3a_bounded_warm_start_c2c_seed0_resume275_to500
```

The resumed run reached iteration 499 without another warm-start-limit failure. This is diagnostic evidence;
C3a remains unpromoted.

## Startup result

The bounded guard removed the previously observed immediate cold-optimizer collapse. In the CPU-native smoke,
checkpoints 0--2 preserved the source actor and zero optimizer steps. The first update at checkpoint 3 used one
epoch at `1e-5`, included active C2c/turn states, and displaced the actor by `0.00808`, below the `0.10` warm-up
limit. This addresses the synchronized entry-only first update, not long-horizon continual-learning interference.

## C3a fixed-grid trajectory

The unchanged 96-case C3a v2 grid was first sampled every 100 iterations and then refined at 450 and 475.

| Iteration | C3a success | Left | Right | Roll-limit terminations | Gate |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 100 | 96/96 | 100% | 100% | 0 | pass |
| 200 | 96/96 | 100% | 100% | 0 | pass |
| 300 | 96/96 | 100% | 100% | 0 | pass |
| 400 | 96/96 | 100% | 100% | 0 | pass |
| 450 | 96/96 | 100% | 100% | 0 | pass |
| 475 | 96/96 | 100% | 100% | 0 | pass |
| 499 | 60/96 | 50% | 75% | 36 | fail |

C3a was therefore learned and retained through iteration 475, then regressed during the final 24 labeled
iterations. Actor displacement from `model_475.pt` to `model_499.pt` was `0.37794`. The bounded warm start did not
cause an immediate C3a failure, but it also did not prevent late PPO policy drift.

## C1 and C2c retention

The two adjacent C3a-passing checkpoints at iterations 450 and 475 were evaluated in fresh CPU-native processes
on the 16-case C1 v2 grid and 112-case C2c v2 grid.

| Iteration | C1 survival | C1 tail limit | C1 gate | C2c survival | C2c climb | `+12` success | C2c gate |
| ---: | ---: | ---: | :---: | ---: | ---: | ---: | :---: |
| 450 | 16/16 | 61.53% | fail | 85.71% | 66.67% | 0/16 | fail |
| 475 | 16/16 | 54.54% | fail | 87.50% | 70.83% | 2/16 | fail |

Both checkpoints passed all C2c slices except positive 12 degrees. At iteration 450, the `+12` slice had 13 tilt
and three cross-track terminations. At iteration 475, it had 12 tilt and two cross-track terminations. C1 had zero
terminations and acceptable mean position errors, but both policies exceeded the 10 percent tail-limit gate by a
large margin.

For comparison, terminal `model_499.pt` failed all three suites: C1 tail-limit fraction was 48.34 percent, C2c
survival/climb success were 82.14/58.33 percent with 0/16 success at `+12` degrees, and C3a success was 62.5
percent.

## Evidence locations

```text
logs/rsl_rl/flapping_bot_straight_flight/2026-08-14_c3a_bounded_coarse_eval/eval/summary.csv
logs/rsl_rl/flapping_bot_straight_flight/2026-08-14_c3a_bounded_refine450_eval/eval/summary.csv
logs/rsl_rl/flapping_bot_straight_flight/2026-08-14_c3a_bounded_refine475_eval/eval/summary.csv
logs/rsl_rl/flapping_bot_straight_flight/2026-08-18_c3a_bounded_retention_c1_eval/eval/summary.csv
logs/rsl_rl/flapping_bot_straight_flight/2026-08-18_c3a_bounded_retention_c2c_eval/eval/summary.csv
```

## Decision and limitation

The bounded warm start successfully removes the immediate cold-optimizer update failure, but the fixed rehearsal
mixture alone does not retain promoted C1/C2c behavior over long C3a training. Neither `model_450.pt`,
`model_475.pt`, nor `model_499.pt` is a promotion candidate. Do not start C3b from this run.

This is one seed and one fixed schedule. The next bounded diagnostic should evaluate C1/C2c at iterations
100/200/300/400 to localize when old-task retention first falls, before changing the optimizer or launching
another long training run.
