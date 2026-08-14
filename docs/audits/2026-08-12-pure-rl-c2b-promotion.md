# PureRL C2b promotion audit

- Date: 2026-08-12
- Branch: `feat/native-multibody-rl`
- Training run: `2026-08-12_10-45-45_pure_rl_c2b_seed0`
- Authority plant: CPU-native measured multibody
- Evaluation contract: `pure_rl_longitudinal_c2b_v1`

## Observed evaluation evidence

After replacing the remote ground asset with project-local geometry, fresh-process evaluation completed the
112-case C2b fixed grid for three late checkpoints. All three checkpoints passed the fixed C2b hard gate:

| Checkpoint | Survival | Level / climb / descent success | Recovery | Mean cross-track (m) | Mean height (m) | P95 height (m) | Reverse | Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `model_450.pt` | 1.000 | 1.000 / 1.000 / 1.000 | 1.000 | 0.06355 | 0.13867 | 0.28778 | 0.000 | 83.1798 |
| `model_475.pt` | 1.000 | 1.000 / 1.000 / 1.000 | 1.000 | 0.07799 | 0.11855 | 0.28910 | 0.000 | 83.5744 |
| `model_499.pt` | 1.000 | 1.000 / 1.000 / 1.000 | 1.000 | 0.07894 | 0.14767 | 0.32150 | 0.000 | 81.5254 |

The adjacent sample-equivalent candidates `model_450.pt` and `model_475.pt` were then evaluated on the frozen
16-case C1 retention suite:

| Checkpoint | C1 success | Termination | Mean cross-track (m) | Mean height (m) | Score | Retention |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `model_450.pt` | 1.000 | 0.000 | 0.04700 | 0.12290 | 98.1326 | pass |
| `model_475.pt` | 1.000 | 0.000 | 0.05300 | 0.09922 | 98.3457 | pass |

The separate signed 10-degree diagnostic was run but is not promotion evidence. Its survival rates were below the
fixed-suite gate and do not alter the approved C2b distribution or the promotion result.

Evaluation artifacts were written outside the repository at:

- `/home/zn/temp/pure_rl_c2b_eval_20260812_XvHIqj/eval/summary.csv`
- `/home/zn/temp/pure_rl_c2b_c1_retention_20260812_SUGzVz/eval/summary.csv`
- `/home/zn/temp/pure_rl_c2b_eval_20260812_XvHIqj/promotion_evidence.json`

The promotion JSON passed `python -m json.tool` validation. The repository helper
`scripts/flapping_rl/pure_rl_longitudinal_promotion.py::evaluate_longitudinal_promotion()` returned
`promoted=true` for the adjacent iterations 450 and 475.

The full suites exercised the project-local ground and current registered task configuration. After those runs,
the watcher interface was tightened so both automated and direct measured PureRL evaluation use a writable
conversion cache rather than the robot source directory. That final cache-routing revision passed focused command,
parser, and configuration tests. A repeat 16-environment runtime smoke could not be started from the later
restricted execution container because CUDA was unavailable there and the request for host execution was rejected
when the approval service returned HTTP 503; this does not replace or invalidate the completed full-suite results.

## Interpretation

`model_475.pt` is the accepted C2b checkpoint and the required weights-only source for C2c. It is selected as the
later member of the adjacent passing pair, not because it is the final training checkpoint. `model_499.pt` passes
the fixed C2b grid but is not part of the required 25-iteration adjacent evidence pair used here.

This is task-level controller evaluation in simulation. It does not validate the aerodynamic model, establish
wind robustness, demonstrate C2c or C3 convergence, or provide real-flight evidence.

## Remaining work

- Train C2c from the accepted `model_475.pt` using weights-only initialization.
- Evaluate two adjacent C2c checkpoints on the 144-case C2c grid and matching C1 retention suite.
- Do not start C3a until C2c promotion is complete.
