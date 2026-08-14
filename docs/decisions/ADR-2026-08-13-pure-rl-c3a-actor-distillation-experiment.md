# ADR-2026-08-13: Actor-only policy distillation experiment for PureRL C3a

- Status: Experimental
- Date: 2026-08-13
- Scope: C3a continual-learning retention on the measured CPU-native plant

## Context

The event-balanced cumulative C3a experiment learned the complete turn grid but still forgot the promoted C2c
strong-climb behavior. At iteration 99 it achieved 100 percent C3a success, but C2c survival was 89.29 percent,
climb success was 75 percent, and the `+12` degree slice succeeded in only 4 of 16 cases. Increasing exact old-task
exposure alone was therefore insufficient.

## Decision

Add an explicit, default-disabled actor distillation coefficient. A C3a weights-only warm start copies the loaded
C2c actor and actor observation normalizer into a frozen teacher. PPO remains on-policy. On C1 and C2c rehearsal
observations only, an auxiliary mean-action MSE penalizes deviation between the current actor and the frozen source
actor. Current C3a turn observations receive zero distillation loss. The critic, action standard deviation, reward,
plant, action contract, and 555-value policy observation remain unchanged.

The first bounded experiment uses coefficient `0.05`, the event-balanced C3a sampling contract, 256 environments,
16 mini-batches, 100 PPO iterations, and seed 0. This is an experiment, not a promoted default.

## Alternatives considered

- Additional old-task sampling alone was rejected for this experiment because the matched event-balanced run did
  not restore strong climb.
- Stale trajectory replay was rejected because PPO data should remain on-policy.
- Critic regularization was rejected because the intended intervention is behavior preservation, not value-function
  preservation.
- EWC and expandable policy architectures were deferred until the smaller actor-only intervention is measured.

## Consequences

- The feature is opt-in and requires a C3a weights-only warm start, which defines the frozen teacher unambiguously.
- RSL-RL and Isaac Lab upstream sources are unchanged; the project reuses the existing auxiliary actor-loss hook.
- Logged `Loss/symmetry` represents the actor distillation MSE in this experiment.
- A stronger coefficient or longer run requires a new controlled comparison rather than silently changing this run.

## Validation requirements

- Unit tests must prove that the teacher is copied after checkpoint loading, remains frozen, and masks current C3a
  rows.
- Runtime evidence must prove that actor and critic input dimensions remain 555 and that the loss is finite.
- Checkpoints must be evaluated on the unchanged C3a, C1, and C2c fixed grids. C3a promotion still requires two
  adjacent current-stage passes and complete passing retention evidence.

## Reconsideration triggers

Reconsider the coefficient or use a different continual-learning method if the bounded run blocks C3a learning,
fails to improve strong-climb retention, or still misses the frozen C1/C2c gates.
