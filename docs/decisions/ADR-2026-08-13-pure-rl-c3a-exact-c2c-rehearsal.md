# ADR-2026-08-13: Exact C2c rehearsal in PureRL C3a

- Status: Accepted
- Date: 2026-08-13
- Scope: C3 longitudinal retention geometry, C3a sampling, and C2 evaluation diagnostics

## Context

The first C3a run passed its 96-case turn grid and later recovered C1 retention, but failed C2c retention. The
accepted C2c source `model_550.pt` had achieved 99.11 percent survival and 97.92 percent climb success on the
112-case v2 grid. After C3a training, checkpoints at iterations 500, 525, and 550 achieved only 89.29--92.86
percent overall survival and 75.00--83.33 percent climb success.

The original C3 generator labeled 25 percent of C3a samples as C2c rehearsal, but represented them with the C3
centerline event profile. That profile used 8--12 m quintic entry and exit transitions around a 20--30 m slope
plateau. The authoritative C2c path contract instead uses a 15--20 m level entry, a direct constant 4--12 degree
slope over 20--30 m, and a direct level recovery. The rehearsal therefore covered similar longitudinal geometry
without reproducing the retained task.

An extended evaluator localized `model_550.pt` failures to positive slope. The `+4` and `+8` degree slices passed
16/16, while `+12` degrees passed 8/16 and all eight failures were tilt terminations. Every descent slice passed,
including all 16 non-promotion `-15` degree diagnostic cases; all 16 `+15` degree diagnostic cases terminated on
tilt.

## Decision

C3 environments retain a separate `PureRLLongitudinalPathBatch` for rows assigned to the C2c rehearsal family.
Those rows use the existing `sample_longitudinal_path_batch()` and `query_longitudinal_path()` implementations for
reset heading, initial velocity, preview, route-relative errors, tangent and normal frames, recovery state, reward
inputs, and termination inputs. The C3-specific 35 degree absolute-roll termination is disabled on those rows so
their termination contract remains C2c. Other rows continue to use the spatial centerline and local projection.

C3a sampling changes from `15/25/60` percent C1/C2c/C3a to `15/35/50` percent. Within the C2c rehearsal family,
zero probability is assigned to level paths because the independent C1 family already supplies level rehearsal;
climb and descent receive `2/3` and `1/3` respectively. This makes strong climb approximately 23.3 percent of all
C3a episodes while keeping 50 percent of episodes dedicated to the current turn task. C3b and C3c retain their
existing family probabilities and balanced C2c climb/descent rehearsal.

The C2 evaluator reports a deterministic signed-slope breakdown with case count, survival, success, recovery, and
ground/tilt/cross-track/height-error termination counts. These diagnostics do not change the frozen v2 grid or
promotion gates.

## Alternatives considered

- Lowering the C2c `+12` degree requirement was rejected because the promoted C2c source passed the same v2 grid.
- Relaxing the 75 degree tilt termination was rejected because the failure represents loss of a previously
  demonstrated capability, not an evaluator-only threshold mismatch.
- Increasing only the old approximate C3 centerline rehearsal was rejected because it would preserve the geometry
  mismatch.
- Continuing the existing C3a optimizer indefinitely was rejected after the bounded 50-iteration experiment failed
  to recover C2c retention.

## Consequences

- C3 training performs one additional batched longitudinal query; the action, 555-value observation, reward
  weights, plant, policy rate, and no-wind boundary are unchanged.
- Existing C3a checkpoints were trained under a different rehearsal contract and cannot establish promotion under
  the corrected curriculum.
- A fresh C3a run should start weights-only from the promoted C2c `model_550.pt`. It must produce two adjacent C3a,
  C1, and C2c passes before C3b starts.

## Assumptions

- The promoted C2c source remains the authority for the 4--12 degree longitudinal envelope.
- The diagnostic concentration at positive 12 degrees justifies temporary climb-heavy rehearsal without changing
  the final symmetric C2c promotion grid.
- C1 rehearsal covers level flight sufficiently when the C2c sub-mixture omits level rows.

## Validation requirements

- Pure logic tests must prove that C3a uses 15/35/50 family probabilities and a 0/2:1 C2c sub-mixture.
- A CPU-native runtime test must compare a forced C2c rehearsal row against direct `query_longitudinal_path()`
  results for preview, errors, tangent, and recovery.
- Checkpoint evaluation must retain the frozen 112-case C2c v2 grid and report signed-slope termination details.
- Promotion still requires two adjacent current-stage passes with complete C1 and C2c retention evidence.

## Reconsideration triggers

Reconsider the C3a mixture if a fresh weights-only run from promoted C2c repeatedly fails either C3a turn learning
or C2c retention, or if per-slope diagnostics show that the failure has moved away from strong positive slope.
