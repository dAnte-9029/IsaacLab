# ADR-2026-08-29: Make the C3b loiter reachable and train from promoted C3a

- Status: Accepted for controlled experiment
- Date: 2026-08-29
- Scope: C3b frozen-grid geometry, evaluation evidence and first training route

## Context

The promoted C3a split actor `model_200.pt` was evaluated zero-shot on the 176-case C3b v2 grid. The finite
turn and turn/vertical templates ended at approximately 125--133 m, but the loiter template ended at 277.5 m.
The fixed 20 s episode therefore required 13.875 m/s to complete loiter, above both the 7 m/s command and the
12 m/s path design speed. All 16 loiter cases failed regardless of policy quality, so v2 could not serve as a
valid frozen promotion contract.

## Decision

Keep the 20 s episode and all C3b template, roll, slope, heading and phase slices. Shorten only the two-event
loiter profile from 260 m to 110 m after its entry segment. The deterministic loiter now ends at 127.5 m, which
is comparable to the other C3b templates and requires 6.375 m/s over 20 s. Register this incompatible correction
as `pure_rl_spatial_c3b_v3`; v2 results remain historical evidence and cannot be mixed with v3 promotion rows.

Every spatial evaluation also writes a compact per-case JSON containing the registered template, slope, turn
sign, heading and phase together with termination, event completion and error metrics. Full step traces are not
duplicated.

Random training batches retain the minimum-altitude and nonlocal-clearance gates. Invalid rows are rejected and
resampled deterministically for at most 16 attempts instead of allowing one batch-wide retry; an unresolved row
still fails closed. This removes a batch-size-dependent startup failure without clipping or accepting an invalid
centerline.

The corrected C3b training route is `c3b_split_frequency_actor_v2`. It loads weights only from the formally promoted
C3a split checkpoint:

```text
logs/rsl_rl/flapping_bot_straight_flight/
2026-08-29_10-07-47_pure_rl_c3a_split_frequency_actor_seed0_201iter/model_200.pt
```

The optimizer is fresh. The split frequency/tail actor, 0.05 governor-gap reward, symmetric 2 Hz/s governor and
bounded warm start are preserved. Task-aware PPO uses the registered C1/C2c/C3a/C3b weights
`0.15/0.20/0.15/0.50`; C2c strong-climb probability is 0.5. Training uses seed 0, 256 environments, 16
mini-batches, 251 iterations and a 25-iteration save interval. Distillation, adaptive sampling, PCGrad, GEM and
additional reward changes remain disabled.

The superseded v1 run stopped after collecting the rollout near iteration 59 because only four active
strong-C2c phase samples were present while a diagnostic-only per-rollout check required 16. Task-aware PPO
normalizes C2c as one task and does not require 16 active-phase rows for a mathematically valid update. V2 keeps
the task-level minimum at 32 and continues logging phase counts, but sets the C3b per-rollout phase abort threshold
to zero. C3a retains its original threshold of 16. V2 restarts from the promoted C3a source rather than resuming
the partial v1 optimizer state.

## Zero-shot v3 evidence

The corrected grid completed all 16 loiter cases, proving the former loiter failure was caused by the horizon
contract. Overall zero-shot success increased from 70.45 to 79.55 percent. Templates 4, 6, 8 and 9 passed every
case; template 3 passed 12/16; climb-containing templates 5 and 7 each passed 16/32 and accounted for all 32
terminations. The policy therefore has substantial compositional transfer, while climb/turn sequencing remains
the principal learning target. This result does not pass the C3b promotion gate.

## Promotion requirements

C3b promotion requires two adjacent checkpoints at the frozen 25-iteration cadence that each pass C3b v3 and
the frozen C3a, C2c and C1 suites in fresh CPU-native processes. A C3b-only pass, missing retention evidence or
any result produced under v2 rejects promotion. The 260 m loiter is deferred to a later endurance contract and
is not silently reintroduced into C3b.

## Consequences

- The C3b task becomes learnable within its unchanged 20 s horizon without relaxing performance gates.
- C3a remains an explicit rehearsal group instead of being merged into current C3b samples.
- Three-task C3a task-aware PPO remains behaviorally unchanged; four-task weighting is enabled only by C3b.
- The first run answers whether standard joint optimization from the promoted split actor can acquire sequence
  composition while retaining all earlier skills. More complex optimization is deferred until that result.
