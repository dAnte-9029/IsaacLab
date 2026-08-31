# ADR-2026-08-27: L1 total variation for the PureRL frequency request

- Status: Accepted for controlled experiment
- Date: 2026-08-27
- Scope: one default-disabled PureRL reward mode and one C1-initialized C3a joint-training route

## Context

The completed `c3a_joint_from_c1_requested_frequency_smoothness_v1` experiment added a squared penalty on the
clipped normalized frequency request before the physical governor. Frozen evaluation still rejected every C2c
checkpoint. Its best near-gate checkpoint, `model_200.pt`, passed survival, climb, descent, recovery and
cross-track gates, but mean and p95 absolute height errors were `0.54884 m` and `1.57906 m`, above the unchanged
`0.50 m` and `1.50 m` limits.

On the same fixed heading-0, reset-phase-0, `+12 deg` trajectory, the squared-penalty checkpoint reduced clipped
request RMS delta from `0.3632` to `0.3247`, raw saturation from `65.29%` to `58.59%`, vertical-response delay from
`0.8667 s` to `0.7667 s`, and full-trajectory height MAE from `1.558 m` to `1.156 m` relative to the matched
unregularized joint `model_200.pt`. It did not remove the switching failure: requested-frequency sign flips only
changed from 29 to 28. The promoted C2c actor produced no sign flips on the same trajectory.

A squared increment cost can make a fixed total request change cheaper by distributing it across more, smaller
increments. It therefore controls amplitude but does not directly price total variation or the count of small
reversals.

## Decision

Retain the existing squared route and add a selectable `absolute` mode for the same default-disabled reward term:

```text
requested_frequency_action_delta_penalty
    = abs(0.5 * (requested_frequency_action[t]
                 - requested_frequency_action[t-1]))
```

Accept one explicit launcher route,
`c3a_joint_from_c1_requested_frequency_total_variation_v1`, with penalty weight `0.10`. The route uses
`--c3a-joint-requested-frequency-total-variation` and otherwise reuses the complete frozen
`c3a_joint_from_c1_v1` recipe: selected C1 `model_1300.pt`, weights-only load, fresh optimizer, seed 0, 256
environments, 16 minibatches, 201 iterations, checkpoint interval 25, task-aware PPO, the 15/35/50 task mixture,
strong-climb probability 0.5, `[256,128]` actor/critic, and train-only execution.

The default mode remains `squared` and the default weight remains `0.0`. Existing tasks, the original joint route,
and the `0.05` squared-smoothness route remain reproducible.

## Alternatives considered

- Increasing the squared weight was not selected because it would strengthen amplitude suppression without
  changing the incentive to split variation across many small increments.
- Penalizing sign flips alone was not selected because it is discontinuous around zero and can be evaded by
  oscillating without crossing zero.
- Removing phase observations, slowing the physical governor, or adding a separate frequency head remain larger
  observation, plant-interface, or architecture changes and are not part of this controlled reward experiment.

## Consequences

- For a monotone change with fixed endpoints, the L1 cost is independent of how many smaller steps divide that
  change; repeated reversals add directly to the cost.
- A constant climb request has zero steady penalty, while genuine maneuver onset incurs one bounded transition
  cost.
- The experiment changes reward optimization, not plant dynamics, action limits, observation, governor, task
  sampling, network, PPO schedule, or frozen promotion gates.
- Frozen C1, C2c, and C3a suites remain the only promotion authority; reduced flip telemetry alone cannot promote
  a checkpoint.

## Assumptions

- The clipped pre-governor normalized request remains the correct actor-facing boundary.
- Weight `0.10` makes a full-range reversal contribute `0.10` before reward aggregation and is an experimental
  value, not a new task default.
- Reset initialization of the previous request prevents an artificial first-step penalty.

## Validation requirements

- Pure Tensor tests must distinguish squared and absolute modes, cover constant-request zero penalty, reject an
  unknown mode, and preserve default-disabled behavior.
- Launcher tests must prove the three joint routes are mutually exclusive, the new route injects exactly
  `mode=absolute` and `weight=0.10`, and provenance records its distinct route.
- Focused PureRL C1/C2/C3 contract tests and Python compilation must pass before launch.
- Fresh CPU-native startup must confirm the C1 source checkpoint, weights-only loading, fresh optimizer, exact
  reward mode and weight, unchanged joint recipe, warm-start guard, and absence of traceback.
- After training, evaluate checkpoints `50/75/100/125/150/175/200` in separate fresh processes on the unchanged
  C1, C2c v2, and C3a v2 suites, then repeat the fixed `+12 deg` response comparison.

## Reconsideration triggers

Reconsider this term if total variation and sign flips do not materially fall, if height accuracy remains outside
the frozen C2c gates, or if C1/C3a performance or maneuver-onset response regresses. Any frequency-only filter,
slow head, phase-invariant representation, or optimizer-gradient method must be a separately approved experiment.
