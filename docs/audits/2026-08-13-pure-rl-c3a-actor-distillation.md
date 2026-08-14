# PureRL C3a actor distillation experiment

## Scope

This audit records a bounded actor-only policy distillation experiment on the measured CPU-native simulation plant.
It is continual-RL controller evidence, not aerodynamic-model validation or real-flight evidence.

## Experiment

- Source: promoted C2c `model_550.pt` from
  `2026-08-12_17-12-48_pure_rl_c2c_seed0_resume500_to550`.
- Run: `2026-08-13_15-50-50_pure_rl_c3a_actor_distill_c2c_seed0_100iter`.
- Training: weights-only, seed 0, 256 environments, 16 mini-batches, 100 iterations, 1,228,800 transitions.
- Distillation: frozen source actor, coefficient 0.05, C1/C2c observations only; critic and action standard deviation
  unconstrained.
- Curriculum: the same event-balanced C3a settings as Experiment 1, including 0.5 strong-climb quota and successful
  C2c recovery recycling.

Training completed with return code 0. The actor and critic each retained a 555-value input. The auxiliary actor
loss remained finite, with mean 0.04294 and final value 0.05343. Mean throughput was 1,667.6 steps/s.

## Fixed-grid results

| Checkpoint | C3a success | C1 termination | C1 tail limit | C2c survival | C2c climb | `+12 deg` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 100% | 0% | 22.32% | 84.82% | 64.58% | 3/16 |
| 75 | 100% | 0% | 16.86% | 89.29% | 75.00% | 4/16 |
| 99 | 100% | 0% | 11.64% | 95.54% | 89.58% | 11/16 |

All three C3a checkpoints passed the unchanged 96-case current-stage gate. At iteration 99, distillation improved
C2c relative to the matched event-balanced run: overall survival increased from 89.29 to 95.54 percent, climb
success from 75.00 to 89.58 percent, and `+12` degree success from 4/16 to 11/16. C1 tail-limit exposure decreased
from 17.45 to 11.64 percent.

The result remains below the frozen gates. C2c has five `+12` degree tilt terminations and does not pass promotion.
C1 has zero terminations but its tail-limit fraction remains above 10 percent. No checkpoint is eligible for C3a
promotion and C3b must not start.

Evaluation evidence is under the run-local `eval_actor_distill/` directory. The three suites were run sequentially
in fresh CPU-native processes with the saved policy checkpoints and unchanged v2 contracts.

## Validation

The focused unit and contract suite passed:

```text
110 passed, 3 warnings in 8.35s
```

This command included the CPU-native C3 runtime gate. A separate 32-environment, two-iteration training smoke also
completed with return code 0 and finite actor distillation loss.

## Conclusion

Actor-only distillation at coefficient 0.05 materially reduced late-stage forgetting without blocking C3a turn
learning, but it did not fully retain promoted C1/C2c performance in 100 iterations. This is a positive bounded
result, not a promotion. The next controlled experiment should tune distillation strength before introducing a
larger continual-learning mechanism.
