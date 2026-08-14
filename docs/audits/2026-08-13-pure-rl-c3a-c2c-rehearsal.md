# PureRL C3a C2c rehearsal audit

## Scope

This audit diagnoses C2c retention loss in the first C3a policy, records the exact-rehearsal implementation, and
reports one bounded continuation experiment. It is controller evaluation evidence on the CPU-native simulation
plant. It is not aerodynamic-model validation or real-flight evidence.

## Diagnosis

Evaluated checkpoint:

`2026-08-13_09-24-10_pure_rl_c3a_seed0_resume499_to550/model_550.pt`

The unchanged 112-case C2c v2 promotion grid produced 92.86 percent overall survival and 83.33 percent climb
success. Signed-slope diagnostics were:

| Slope | Success | Termination causes |
| ---: | ---: | --- |
| -12 deg | 16/16 | none |
| -8 deg | 16/16 | none |
| -4 deg | 16/16 | none |
| 0 deg | 16/16 | none |
| +4 deg | 16/16 | none |
| +8 deg | 16/16 | none |
| +12 deg | 8/16 | 8 tilt |

The separate diagnostic achieved 16/16 at `-15` degrees and 0/16 at `+15` degrees; all positive failures were
tilt terminations. Evidence is under:

`/home/zn/temp/pure_rl_c3a_c2c_detail_YpE8eG/eval/model_550.json`

## Implementation

- `pure_rl_longitudinal_eval.py` now emits `slope_breakdown` with per-slope success and termination counts.
- `watch_and_eval.py` records the four pre-reset termination causes for every C2 episode.
- C3 rows assigned to C2c rehearsal now use the authoritative longitudinal sampler and query instead of the smooth
  C3 event approximation.
- C3a sampling is 15 percent C1, 35 percent C2c, and 50 percent C3a. C2c rehearsal is weighted 2:1 toward climb.
- Plant parameters, action and observation contracts, reward weights, C2c grid, and promotion thresholds are
  unchanged.

## Bounded continuation experiment

The existing C3a `model_550.pt` and optimizer were resumed for 52 launcher iterations, producing cadence
checkpoints 575 and 600:

`2026-08-13_10-12-25_pure_rl_c3a_rehearsal_fix_resume550_to600`

Both checkpoints passed C1 retention. `model_600.pt` also passed the C3a 96-case grid with 100 percent survival,
event completion, and success. Mean horizontal and vertical errors were 0.2712 m and 0.1143 m.

Neither checkpoint recovered C2c:

| Checkpoint | Overall survival | Climb success | +8 deg | +12 deg |
| ---: | ---: | ---: | ---: | ---: |
| 575 | 85.71% | 66.67% | 12/16, 4 tilt | 4/16, 12 tilt |
| 600 | 85.71% | 66.67% | 12/16, 4 tilt | 4/16, 12 tilt |

All level and descent slices remained 16/16. Evaluation evidence is under:

`/home/zn/temp/pure_rl_c3a_rehearsal_fix_eval_uvuuru`

## Validation

The focused non-Isaac suite passed:

```text
146 passed in 2.43s
```

The fresh CPU-native C3 runtime gate passed:

```text
1 passed, 3 warnings in 6.50s
```

The runtime gate forced a C2c rehearsal row and matched preview, route errors, tangent, and recovery against a
direct longitudinal query. The warnings were existing Gym registration and headless shutdown warnings.

## Conclusion

The evaluator and rehearsal contract are corrected, but the bounded continuation did not restore strong climb.
C3a is not promoted. Further same-optimizer continuation is not justified by this experiment. The next authority
run should initialize weights-only from the promoted C2c `model_550.pt` and train C3a under the corrected curriculum
from its first iteration.
