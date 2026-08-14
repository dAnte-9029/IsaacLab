# ADR-2026-08-12: PureRL maneuver envelope v2

- Status: Accepted
- Date: 2026-08-12
- Scope: untrained C2c and C3 longitudinal/spatial curriculum capability targets
- Supersedes: the C2c and C3 slope bounds in the 2026-08-10 C2 and 2026-08-11 C3 designs

## Context

The original longitudinal curriculum was intentionally conservative. C2b ended at 6 degrees and C2c at 8
degrees, while signed 10-degree paths were diagnostic only. The promoted C2b policy demonstrated reliable basic
longitudinal path following, but a 6-degree path at approximately 7 m/s demands only about 0.74 m/s vertical
velocity and does not represent the desired final maneuver capability.

The later curricula had not been trained, so their envelopes can be strengthened without invalidating a promoted
C2c or C3 checkpoint. C2a and C2b are retained as learned foundations. Their task distributions, evaluation
contracts, and promotion evidence remain unchanged.

## Capability progression

- C1 learns stable straight and level path following, direct flap-frequency and tail control, smooth actions, and
  recovery from randomized heading and flap phase.
- C2a learns to interpret vertical preview geometry and enter, hold, and leave a mild climb or descent.
- C2b learns reliable basic longitudinal maneuvering through a level-entry, slope, and level-recovery sequence.
- C2c learns the final straight-line longitudinal envelope, including materially stronger climb and descent while
  retaining C1.
- C3a isolates horizontal turning and lets the policy infer necessary bank from path error without a target-roll
  command.
- C3b learns ordered multi-event behavior: repeated turns, S-turns, loiter, and strong climb or descent before or
  after a turn.
- C3c learns coupled three-dimensional maneuvering with simultaneous turn and climb or descent while retaining all
  earlier capabilities.

## Decision

The accepted v2 maneuver envelope is:

| Stage | Training envelope | Promotion grid or severity |
| --- | --- | --- |
| C2a | 1.5--4 degrees | `0, +/-2, +/-4` degrees; unchanged |
| C2b | 2--6 degrees | `0, +/-2, +/-4, +/-6` degrees; unchanged |
| C2c | 4--12 degrees | `0, +/-4, +/-8, +/-12` degrees |
| C2c diagnostic | n/a | `+/-15` degrees, never promotion eligible |
| C3a | level turns | 9/11/13-degree geometry-roll levels |
| C3b | sequential vertical events at 4--12 degrees | each vertical template at 8 and 12 degrees |
| C3c | coupled vertical magnitude 3--10 degrees | `(roll, slope)` severities `(8,4)`, `(12,7)`, `(6,9)` degrees |

C3c coupled events satisfy:

```text
(phi_geometry / 20 deg)^2 + (abs(gamma) / 10 deg)^2 <= 1
```

Multiple sampled C3c vertical events alternate climb and descent direction. This preserves multi-event spatial
control without allowing repeated same-direction descents to construct a path below the ground plane.

The 20--30 m training slope length and fixed 25 m evaluation slope remain unchanged. A 25 m path at 12 degrees
changes altitude by approximately 5.31 m. At 6.5--7.3 m/s, a 12-degree flight-path angle corresponds to about
1.38--1.55 m/s vertical velocity, which reaches the upper portion of the available real-flight descriptive range.
The 15-degree diagnostic probes margin rather than defining promotion authority.

## Versioning and lineage

The unchanged C2a and C2b evaluation contracts remain v1. C2c becomes
`pure_rl_longitudinal_c2c_v2`. All C3 evaluation contracts become v2 because their C2c rehearsal dependency and
longitudinal envelope changed. Promotion helpers reject rows from the superseded contracts.

Promoted C2b `model_475.pt` remains the required weights-only source for C2c. It is evidence for the basic
longitudinal foundation, not evidence that the v2 C2c envelope has been learned. No C3 training may begin until a
C2c checkpoint passes the v2 grid and C1 retention gate.

## Unchanged boundaries

The four-action and 555-observation policy contracts, CPU-native plant, 60 Hz policy rate, reward weights, path
error thresholds, 25-degree P95 roll gate, 35-degree roll termination, no-wind boundary, training defaults, and
sample-equivalent promotion cadence are unchanged. This decision does not claim the stronger curricula converge;
that requires PPO training and complete promotion evidence.
