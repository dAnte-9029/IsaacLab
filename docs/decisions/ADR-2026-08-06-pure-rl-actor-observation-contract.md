# ADR: Freeze the first-curriculum PureRL actor observation

- Status: Accepted
- Date: 2026-08-06
- Scope: `FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg`
- Evidence: `docs/audits/2026-08-06-pure-rl-observation-calibration.md`

## Context

The first curriculum must learn closed-loop straight flight without a target
speed input or a precomputed trim controller. The actor nevertheless needs a
temporally useful, physically interpretable state representation that remains
valid when the world-frame line direction changes. The historical 68-value
controller-oriented observation does not contain the selected quaternion,
phase, action-history and path-preview contract.

## Decision

Measured PureRL uses a 555-value actor observation comprising 30 sensor frames,
30 applied direct-action frames and five body-frame preview points. Histories
are sampled at the 60 Hz policy rate and ordered oldest to newest. Reset fills
each history with its first real sample. World `wxyz` quaternion signs are
made continuous across adjacent frames.

Each sensor frame is:

```text
[q_WB_wxyz, ground_velocity_body_xyz, angular_velocity_body_xyz,
 forward_air_velocity_body_x, actual_flap_frequency, sin(phase), cos(phase)]
```

Forward air velocity is ideal truth for curriculum 1. The preview queries an
unbounded level straight line at `0.12--0.60 s`, using current along-track
ground speed limited to `1--12 m/s`. It does not expose commanded speed.

Physical scales are fixed at `12 m/s` for body-x ground and forward air
velocity, `5 m/s` for lateral/vertical ground velocity, `5 rad/s` for all
angular rates, and `5 m` for preview coordinates. Actual frequency maps
affinely from `0--5 Hz` to `-1--1`. Quaternion, phase features and normalized
actions remain unchanged. The assembled vector receives a final `[-5, 5]`
safety clip.

At reset, route heading and flap phase are randomized over a full period.
Route tangent, root yaw and initial world velocity rotate together; root
position remains at the route origin and commanded height. Flap phase, wing
joint position/velocity and native drive state are initialized consistently.
Cross-track reward and termination are route-relative.

The critic uses the same actor observation. Wind, sensor noise, delay, low-pass
filtering and privileged critic state are not part of curriculum 1.

## Consequences

- The policy sees actual flap frequency separately from its prior action, so
  actuator dynamics are observable without confusing target and state.
- Body-frame preview geometry is invariant to random world route direction,
  while the world quaternion preserves absolute attitude information.
- Fixed physical scales are inspectable and stable across runs; calibration
  gates can identify whether the safety clip is masking normal data.
- Thirty frames give 0.5 seconds of policy-rate context but increase actor
  input width and CPU memory traffic.
- The actor does not receive an explicit speed command. Learned equilibrium
  speed is determined by dynamics, reward and action costs.
- Historical 68-value environments and zero-heading/zero-phase reset behavior
  remain available because all new behavior is explicitly selected by the
  Measured PureRL configuration.

## Validation requirements

- Pure tensor tests must verify the 555-value layout, exact component order,
  quaternion continuity, reset filling, preview geometry and scale mapping.
- Static environment tests must verify that only Measured PureRL selects the
  new observation and randomizations.
- Synthetic, randomized-reset, scripted-dynamics and near-boundary gates must
  remain finite and report pre-clip excursions.
- A 64-environment CPU runtime gate must verify observation shape, finite
  values, safety bounds, partial reset behavior and native mechanism tracking.

## Reconsideration triggers

Reconsider scales if representative closed-loop training data repeatedly
reaches the safety clip. Reconsider history length if profiling shows it is a
material CPU bottleneck or ablation shows no control benefit. Add Pitot
imperfections and wind only with a new curriculum decision. Add privileged
critic state only with an explicit asymmetric-critic contract. Any change to
layout, order, scale or reset randomization requires a superseding ADR and new
gate evidence.
