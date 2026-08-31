# ADR-2026-08-27: Pre-governor requested-frequency smoothness for PureRL joint training

- Status: Accepted for controlled experiment
- Date: 2026-08-27
- Scope: default-disabled PureRL reward term and one C1-initialized C3a joint-training route

## Context

The completed seed-0 `c3a_joint_from_c1_v1` run learned C3a and retained C1. Its C2c performance improved through
iteration 175 to 100 percent survival and climb success, but every checkpoint failed the frozen C2c path-accuracy
gate. At iteration 175, mean absolute height error was `0.6609 m` and p95 absolute height error was `1.9065 m`,
above the unchanged `0.50 m` and `1.50 m` limits.

A fixed `+12 deg` trajectory comparison between promoted C2c `model_550.pt` and joint `model_175.pt` exposed a
control-interface failure. The joint actor's deterministic requested frequency action changed sign 27 times over
the active slope. The corresponding `3.49 Hz` oscillation matched the `3.43 Hz` mean wingbeat frequency, and a
first-harmonic flap-phase regression explained approximately 76 percent of requested-action variation. The
promoted C2c actor remained positive throughout the same slope. Its mean vertical-velocity response delay was
`0.25 s`, compared with `0.567 s` for the joint actor.

The 555-value actor observation explicitly contains `sin(phase)` and `cos(phase)` in each of 30 sensor-history
frames. This makes phase-conditioned control representable. The existing frequency-slew penalty observes the
2 Hz/s governor output, while action-delta regularization covers only the three tail actions. Opposite saturated
frequency requests can therefore cancel through the governor without receiving a direct requested-action delta
penalty.

## Decision

Add a default-disabled reward term on the clipped normalized frequency request before the frequency governor:

```text
requested_frequency_action_delta_penalty
    = (0.5 * (requested_frequency_action[t]
              - requested_frequency_action[t-1])) ** 2
```

The default penalty weight is `0.0`, preserving all existing tasks and checkpoints. The environment keeps one
previous requested-frequency scalar per environment, initializes it to the reset request, updates it after reward
calculation, and exposes penalty, contribution, absolute-delta, and sign-flip telemetry. The physical frequency
governor, action limits, plant, observation, task sampler, network, PPO settings, and frozen gates are unchanged.

Accept one explicit launcher route,
`c3a_joint_from_c1_requested_frequency_smoothness_v1`. It reuses the complete `c3a_joint_from_c1_v1` recipe and
changes only:

```text
requested_frequency_action_delta_penalty_weight = 0.05
```

The original `--c3a-joint-from-c1` route remains reproducible. The new route is selected by
`--c3a-joint-requested-frequency-smoothness`, starts weights-only from the same selected C1 `model_1300.pt`, uses
a fresh optimizer, runs 201 iterations, and remains train-only.

## Alternatives considered

- Lowering the physical frequency-governor rate was rejected because it would further delay strong-climb
  response without teaching a coherent request.
- Enabling the existing global action low-pass was rejected because it filters all four actions and would also
  reduce tail-control bandwidth.
- Removing phase from the complete observation was rejected because tail and periodic-state estimation may need
  phase information.
- More C2c rehearsal and wider actors were rejected as the first response because neither directly constrains
  within-task, within-wingbeat frequency-request reversal.
- A phase-invariant slow frequency head remains a later option if the direct requested-action penalty is
  insufficient, but it changes architecture and warm-start behavior.

## Consequences

- Full-range `-1 -> +1` frequency-request reversal has unit penalty; a constant nonzero climb request has zero
  steady penalty. This separates persistent wingbeat-synchronous reversal from one maneuver-onset transition.
- The experiment changes reward optimization and cannot be compared as initialization-only evidence. It remains
  comparable with the original joint run as a single added reward term.
- Training telemetry can show whether requested-action reversal decreases, but frozen C1, C2c, and C3a suites
  remain the sole promotion authority.

## Assumptions

- The clipped pre-governor normalized request is the correct boundary for actor smoothness; unbounded inference
  output beyond `[-1, 1]` remains outside the environment action contract.
- Weight `0.05` is large enough to distinguish recurring full-range reversals without materially penalizing one
  slow maneuver transition. This is an experimental value, not a promoted default.
- Reset initialization removes artificial first-step penalties.

## Validation requirements

- Pure Tensor tests must cover a full-range sign reversal, constant-request zero penalty, default-disabled
  behavior, invalid inputs, and Hydra-style configuration restoration.
- Launcher tests must prove the original route has no new override, the new route injects exactly weight `0.05`,
  and provenance records the distinct route.
- A fresh CPU-native startup must confirm the selected C1 source, weights-only loading, fresh optimizer, 256
  environments, `[256,128]` actor/critic, task-aware PPO, warm-start guard, the reward override, and no traceback.
- After training, checkpoints `50/75/100/125/150/175/200` must be evaluated in separate fresh processes on the
  unchanged C3a, C1, and C2c suites. The fixed response comparison should then measure requested-action sign
  flips, total variation, flap-phase harmonic dependence, actual frequency, and height response.

## Reconsideration triggers

Reconsider this reward term if requested-frequency sign reversals remain phase locked, C2c path accuracy does not
improve, C3a or C1 regresses, or coherent maneuver-onset frequency changes become too slow. In that case, test a
frequency-only slow head or frequency-only filter as a separate architecture experiment; do not silently change
the plant, global tail-action bandwidth, or frozen evaluation thresholds.
