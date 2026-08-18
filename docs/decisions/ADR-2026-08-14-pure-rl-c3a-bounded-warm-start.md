# ADR-2026-08-14: Bounded PPO warm start for PureRL C3a

- Status: Experimental
- Date: 2026-08-14
- Scope: weights-only C2c-to-C3a transfer on the measured CPU-native plant

## Context

A fresh C3a gradient-conflict probe loaded the promoted C2c `model_550.pt` weights without optimizer state. Its
first 48-step rollout had no active turn or C2c strong-climb transitions, but the subsequent PPO iteration changed
the actor by norm `0.38454`, about 9.5 times the later median. Frozen evaluation immediately fell from 15/16 to
0/16 success at `+12` degrees. The source Adam state had 35,392 updates, while the fresh optimizer began from zero;
the adaptive schedule also began at `3e-4` and could raise the first minibatch rate before reacting to KL.

## Decision

Enable a project-local, weights-only warm-start guard for C3a. The guard discards the first three rollout buffers
without updating so the unchanged source policy advances into active maneuver states. It then performs ten PPO
updates using one learning epoch and a fixed learning rate linearly increasing from `1e-5` to `5e-5`. Later updates
restore the configured epoch count and remain at fixed `5e-5`. The guard measures the actor parameter displacement
on every update and, during the ten warm-up updates, fails before checkpoint saving if its norm exceeds `0.10`.

The base environment keeps the feature disabled. The measured C3a configuration enables it, and the hook is
attached only by the weights-only checkpoint-loading path. Fresh random initialization, same-stage resume, C1,
C2, C3b, C3c, evaluation, plant dynamics, rewards, and observations remain unchanged.

## Alternatives considered

- Randomizing only `episode_length_buf` was rejected because it staggers timeout but does not randomize path
  progress or guarantee maneuver-state coverage.
- Loading the prior Adam state was rejected because cross-stage training should not inherit task-specific optimizer
  momentum.
- Changing upstream RSL-RL adaptive-KL code was rejected; a project-local fixed schedule isolates the intervention.
- PCGrad was deferred because the held-out collapse occurred before active climb/turn gradients were present.

## Consequences

- Checkpoints 0--2 are intentionally unchanged copies of the source policy while the environment advances.
- Training logs expose skipped-update status, warm-start update index, learning rate, epoch count, and actor update
  norm. The `0.10` fail-closed limit applies only while the optimizer is in the ten-update warm-up phase.
- Iteration number no longer equals the number of PPO updates during the three-iteration burn-in.
- C3a convergence and retention still require frozen-grid evaluation; the guard is not a promotion result.

## Assumptions

- Three 48-step rollouts are sufficient for the current C3a entry geometry to expose active C2c and turn states.
- Fixed `5e-5` remains plastic enough for C3a while avoiding the cold-start Adam displacement seen at `3e-4`.
- Actor update norm `0.10` is a diagnostic safety boundary, not a universal PPO threshold.

## Validation requirements

- Unit tests must cover skipped storage, the learning-rate ramp, temporary epoch count, default-disabled wiring,
  C3a enablement, and fail-closed actor displacement.
- A CPU-native weights-only smoke must show no actor or optimizer change in checkpoints 0--2 and a bounded first
  update at checkpoint 3 after active maneuver telemetry appears.
- The next controlled run must evaluate early checkpoints on the unchanged C1, C2c, and C3a frozen grids before
  making any retention or promotion claim.

## Reconsideration triggers

Reconsider the burn-in duration or fixed schedule if the first updated rollout lacks active maneuver transitions,
the update norm exceeds `0.10`, C3a fails to learn, or held-out C1/C2c still collapses immediately despite the
bounded first update.
