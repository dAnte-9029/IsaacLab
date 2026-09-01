# ADR: Route-heading-canonical attitude observation for PureRL C3b

- Date: 2026-08-31
- Status: Experimental

## Context

The paired global-yaw consistency experiment did not remove the matched heading-0 versus heading-180 action
divergence. At iteration 100 it recovered three C3b frozen cases, but the reset first-action RMS difference became
slightly larger and all remaining failures were concentrated at heading 180. The 555-value PureRL observation
contains 30 world-frame attitude quaternions, while body-frame velocities, rates and path preview already remove
the common global-yaw degree of freedom. The actor must therefore learn a coordinate invariance that the task does
not require.

## Decision

Add a default-disabled observation option that replaces each world-frame actor attitude sample with

```text
q_route_body = q_z(-route_heading) * q_world_body
```

where `route_heading` is the fixed initial tangent heading for the episode. Do not remove the aircraft's current
yaw: yaw deviation relative to the route remains in `q_route_body`. Normalize the result and choose the quaternion
sign with a non-negative scalar component before temporal sign alignment. Store the canonical quaternion in the
existing 30-frame sensor history.

The world quaternion remains authoritative for physics, wind conversion, reward, termination and body-frame path
preview. The observation remains 555 values, so the split actor checkpoint tensor shapes are unchanged. The old
world-quaternion observation remains the default.

Run one seed-0, 101-iteration controlled route,
`c3b_adaptive_sampling_heading_canonical_observation_from_model100_v1`, weights-only from the fixed-mixture C3b
`model_100.pt`. Preserve the split-frequency actor, fresh optimizer, adaptive reset sampler, task-aware PPO
`15/20/15/50`, governor-gap reward, bounded warm start, 256 environments, 16 minibatches and 25-iteration save
cadence. Set paired yaw-consistency loss and distillation to zero.

Training and every frozen evaluation process must enable the observation option explicitly. Frozen C3b v3, C3a,
C2c and C1 gates remain unchanged.

## Alternatives considered

- Removing the current yaw was rejected because it would erase route-relative yaw error.
- Increasing the auxiliary yaw-consistency coefficient was rejected because the 0.05 experiment was non-monotonic,
  slightly degraded C2c and did not reduce matched first-action divergence.
- A yaw-equivariant network was deferred because observation canonicalization is the smaller direct intervention.
- Changing quaternion dimension or replacing it with Euler angles was rejected because it would prevent strict
  architecture-compatible warm start and introduce a new singular representation.

## Consequences

- Equivalent trajectories under a common world-yaw rotation receive the same attitude representation by
  construction, including the heading-180 quaternion boundary.
- The source checkpoint is architecture-compatible but not observation-function-preserving away from heading 0;
  this is the single controlled experimental variable.
- Checkpoints from this route require the explicit canonical-observation flag during deployment and evaluation.
- This is a new opt-in observation semantic, not a replacement for the accepted shared-v1 baseline.

## Assumptions

- `_straight_line_heading_rad` is the initial path tangent for C1, authoritative C2c rehearsal and each C3 path.
- Route-relative attitude remains within the non-negative-scalar quaternion hemisphere before termination, so the
  sign rule does not create an operational discontinuity.
- Body-frame velocity/rate and preview channels already carry the remaining information needed for control.

## Validation requirements

- Pure Tensor tests must prove invariance across route headings and equality for `q` and `-q`, while preserving
  batch, dtype and device checks.
- Contract tests must prove default-disabled behavior, unchanged 555-value dimension, explicit launcher provenance,
  zero yaw-consistency loss and evaluator flag propagation.
- A fresh CPU-native smoke must construct the C3b split actor, load the fixed-mixture `model_100.pt` weights and
  complete the first trainable PPO update without non-finite values or traceback.
- Performance evidence requires fresh-process evaluation of models 50, 75 and 100 on C3b v3, C3a, C2c and C1.

## Reconsideration triggers

Reconsider the representation if heading-0 behavior regresses materially, old-suite retention fails, matched
heading actions remain different under canonical observations, or route-relative quaternion signs still jump in
recorded histories. Do not change the plant, governor, reward or frozen grids to compensate.
