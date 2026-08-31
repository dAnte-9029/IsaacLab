# ADR-2026-08-28: Penalize PureRL frequency requests rejected by the governor

- Status: Accepted for controlled experiment
- Date: 2026-08-28
- Scope: one default-disabled PureRL reward term and one C1-initialized C3a joint-training route

## Context

The completed C1-initialized joint experiment learned C1 and C3a but failed the frozen C2c path-accuracy gates.
Both the `0.05` squared request-delta experiment and the `0.10` L1 total-variation experiment retained a
deterministic wingbeat-synchronous frequency request. On the fixed heading-0, reset-phase-0, positive 12-degree
trajectory, L1 checkpoints 150/175/200 produced 24--27 request sign reversals while the promoted C2c actor
produced none.

The PureRL action low-pass and generic action rate limit are disabled. The physical frequency governor remains
enabled with symmetric 2 Hz/s rise and fall limits. The actor observes the governor-applied action history, while
the failed delta penalties depend on the preceding pre-governor request, which is not part of that history. The
governor therefore attenuates the physical effect of the periodic request, and the reward does not directly price
the portion of the request that the governor rejects.

## Decision

Retain the physical frequency governor and add a default-disabled penalty at the normalized action boundary:

```text
requested_applied_frequency_action_gap_penalty
    = abs(clipped_requested_frequency_action
          - governor_applied_frequency_action)
```

Accept one explicit launcher route,
`c3a_joint_from_c1_requested_applied_frequency_gap_v1`, with penalty weight `0.05`. It is selected by
`--c3a-joint-requested-applied-frequency-gap` and otherwise reuses the complete frozen
`c3a_joint_from_c1_v1` recipe: selected C1 `model_1300.pt`, weights-only load, fresh optimizer, seed 0, 256
environments, 16 mini-batches, 201 iterations, checkpoint interval 25, task-aware PPO, the 15/35/50 task
mixture, strong-climb probability 0.5, `[256,128]` actor/critic, and train-only execution.

The existing joint, squared-delta and L1-total-variation routes remain reproducible. Their reward terms remain
default-disabled, and the new route does not enable them.

## Alternatives considered

- Disabling the governor was rejected because it would change the accepted plant/action contract and transmit the
  existing high-frequency request directly into actual wing frequency and aerodynamic loading.
- Increasing the L1 weight was rejected because it retains the hidden preceding-request dependency and does not
  directly identify governor-infeasible requests.
- Removing phase observations or adding a separate slow frequency head remains a larger observation or
  architecture experiment and is not needed for this first interface-aligned test.
- Changing the cubic flap-frequency effort proxy in the same run was rejected because it would confound request
  switching with the separate question of whether the actor selects an insufficient mean frequency.

## Consequences

- A request already reachable by the governor has zero gap penalty.
- A request beyond the one-step governor envelope is penalized only by its rejected magnitude; the term becomes
  zero after the applied command catches up.
- The reward depends on the current action and current applied command, both available at the reward boundary; it
  does not require the actor to reconstruct its preceding raw request.
- Reduced request/applied gap or sign flips are diagnostics only. Frozen C1, C2c and C3a suites remain the sole
  promotion authority.

## Assumptions

- The governor-applied normalized frequency action is the correct actuator-interface boundary.
- Weight `0.05` is sufficient to price the observed rejected request without suppressing necessary maneuver
  onset; it is experimental and does not change the task default.
- The unchanged cubic frequency penalty permits this run to isolate request feasibility from mean-frequency
  selection.

## Validation requirements

- Pure Tensor tests must cover zero gap, positive governor rejection, weighted reward contribution,
  default-disabled behavior and missing-request rejection.
- Launcher tests must prove route exclusivity, exact weight `0.05`, disabled request-delta terms and distinct
  provenance.
- Focused PureRL reward, launcher and C1/C2/C3 contract tests plus Python compilation and `git diff --check` must
  pass before launch.
- Fresh CPU-native startup must confirm the selected C1 source, weights-only loading, fresh optimizer, exact new
  reward weight, unchanged joint recipe, and absence of traceback.
- After training, checkpoints 50/75/100/125/150/175/200 must be evaluated in separate fresh processes on the
  unchanged frozen C1, C2c v2 and C3a v2 suites, followed by the fixed positive 12-degree response comparison.

## Reconsideration triggers

Reconsider the interface or actor representation if the request/applied gap and phase-locked sign flips remain
large. If switching disappears but C2c still fails with low actual frequency, run a separate controlled ablation
of the cubic flap-frequency penalty rather than changing it in this experiment.
