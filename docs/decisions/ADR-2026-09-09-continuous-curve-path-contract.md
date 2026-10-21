# ADR: Continuous three-dimensional path engineering contract

- Date: 2026-09-09
- Status: accepted engineering scope (user authorized implementation of steps 1--3)
- Base: `feat/native-multibody-rl`, `41ea8565`
- Scope: path generation, replay, environment integration and smoke validation

## Context

C3 already provides smooth event transitions and simultaneous turn/climb events.
The new goal is tracking a supplied continuously varying three-dimensional path,
including paths not described by a finite event sequence. The original full-joint
`model_175.pt` passed five suites once, but is not promoted. This change neither
loads that policy nor starts a new training run.

## Decision

Use the explicit `MeasuredPureRL-C4-Direct-v0` project-local task. It is an
engineering/replay task without a registered PPO runner. Existing C1--C3 task
defaults and event samplers are preserved. Six-group rehearsal and PPO are a
later stage.

`flapping_bot.path_tracking.continuous_curve` owns recipes, dense geometry,
serialization and sampled kinematic screening. The direct environment's
`pure_rl_spatial_path` adapts these tables to the existing local projection
query; it does not move old C3 geometry into the new module.

### Geometry and reset

- `s` is 3-D arc length in meters. The episode-local world frame is z-up.
- `k = d(heading)/ds`; true horizontal projected curvature is `k/cos(gamma)`.
- `dp/ds = (cos(gamma) cos(heading), cos(gamma) sin(heading), sin(gamma))`.
- Heading and centerline use trapezoidal integration on a 0.25 m grid.
- The table is 300 m. The first 20 m uses a quintic entry envelope. Curves
  continue smoothly beyond 140 m; there is no recovery event or terminal stop.
- Recipes contain biases, two harmonics, independent phases/wavelengths,
  optional chirp (nonconstant wavelength), horizontal/vertical mirrors, initial
  heading, altitude and flap phase. They begin with zero slope and curvature.
- Base wavelengths are 60--120 m, second harmonic fraction 0--0.35 and chirp
  0--0.15. These are candidate recipe bounds, not an acceptance guarantee.
- JSON stores complete recipes, the exact contract and screening reports.
  Consumers rebuild tables at an explicit dtype/device. Cached reports are never
  trusted for acceptance. Dense tables can additionally be exported as NPZ.
- A file owns geometry and reset heading/altitude/flap phase. Conflicting legacy
  evaluation schedules are rejected. Environment row `i` uses recipe
  `i % recipe_count`, including partial resets, rather than reset-batch rank.
- Without a file, resets draw bounded candidate recipes independently. No
  rejected curve is clipped, relabeled or silently replaced. A rejected recipe
  raises with its ID. This is not yet a production training sampler.

### Screening and evidence boundary

The design speed band is 7--12 m/s, not a target-speed reward or speed governor.
At the upper speed, the screen uses `phi_proxy=atan(v^2*k/g)` and checks:

| Quantity | Engineering screen |
| --- | --- |
| Reference altitude over full table | 4--50 m |
| Absolute geometric roll proxy | <=14 degrees |
| Absolute slope | <=8 degrees |
| `(phi_proxy/20deg)^2 + (gamma/10deg)^2` | <=0.9 |
| Absolute `dk/ds` | <=0.002 rad/m² |
| Absolute `dgamma/ds` | <=0.017 rad/m |
| Absolute `v*d(phi_proxy)/ds` | <=0.35 rad/s |
| `v²*sqrt((cos(gamma)*k)^2+(dgamma/ds)^2)` | <=4 m/s² |
| Nonlocal sample separation | >=4 m, on a 1 m grid, excluding arc neighbors within 8 m |

The altitude floor reserves approximately the existing 3 m tracking corridor
above the ground threshold. The ceiling bounds the first engineering domain.
Derivative and acceleration thresholds screen rapidly varying candidates; they
are provisional engineering choices, not identified actuator or aerodynamic
limits. Derivatives include the entry envelope and second harmonic. All
screening covers the complete 300 m table, not only the successful prefix.

These are sampled geometric/kinematic checks, not a proof between samples,
collision-free tracking-tube certification, or dynamic feasibility. Actual
speed outside the assumed band is reported after entry. A future dynamic audit
must check speed, frequency governor, actual tail response and control margins
before any path is called dynamically feasible. The fixed-wing geometric roll
proxy is not a commanded roll or measured attitude requirement.

### Observation, reward and completion

Retain the existing 555-value actor input and 0.12--0.60 s preview. Preview uses
clipped current tangent speed, not a time-indexed reference trajectory. Preserve
the fixed-initial-heading canonical attitude option. The C4 config selects the
existing 0.05 governor-gap reward and existing plant/action/termination limits.

Define the C4 roll-reward input continuously as
`turn_activity=k²/(k²+0.003²)`. It is zero at zero curvature and smooth through
sign reversals; it introduces no event boundary or path-peak normalization. The
scale is an explicit engineering task choice, requiring trace review later.
Reward equations and old-stage inputs are unchanged.

Completion means local projected progress reaches 140 m. The legacy
`final_event_progress_m`/`reached_all_events` fields carry this value only at the
adapter boundary; C4 event count is zero. C4 exposes separate completion and
progress telemetry. Completion does not truncate the episode. The aircraft
must keep tracking the continuation until the 20 s horizon. No timed arrival
or reference-speed tracking is claimed.

Freeze these later evaluation semantics now: report completion by 20 s,
survival through the horizon, and per-path as well as pooled horizontal/vertical
normal errors over the entire rollout. A per-path accuracy pass requires mean
absolute error <=0.50 m and P95 <=1.50 m in both axes, finite values, survival
and completion. Report reverse-motion fraction, roll, and design-speed
violations separately. This step implements geometry and telemetry; the
checkpoint evaluator and promotion integration are not yet implemented.

### Diagnostic grid and generalization

The frozen 96-case grid is three recipes (coupled, quadrature, biased curve),
two horizontal mirrors, two vertical mirrors, four headings and two flap phases.
The exact recipes live in `continuous_curve_diagnostic_grid()`. It is a
diagnostic grid, not 96 distinct shapes or a final generalization test.

Before training, define separate development and final held-out curve sets,
including nonperiodic variation, multiple severities per family and unseen
parameter combinations. Do not use training failures to filter the final set.
Old C1/C2c/C3a/C3b/C3c frozen gates remain unchanged.

## Alternatives considered

- Another event sequence: does not address continuously varying geometry.
- Unconstrained Cartesian control points: require additional regularity and
  feasibility handling. Constrained splines remain a valid later option.
- A full trajectory optimizer or new controller: deferred until the small
  generator and current policy's traces establish the actual missing capability.
- Enlarging actor input or changing network/reward weights: deferred pending
  preview and dynamic-response evidence.
- Time-synchronized tracking: a separate contract requiring velocity/time
  reference inputs and objectives.

## Validation requirements

Verify batch size one, float32/64, Torch CPU/CUDA where available, reproducible
sampling and independent single-reset mirrors, analytic versus numerical
derivatives, unit arc steps, frame handedness, finite data, rejection of bad
recipes, 96-case screening, exact JSON replay, future-path preview distinction,
global-yaw equivalence, local projection continuity and completion boundaries.
Run a fresh six-environment CPU-native smoke with 555 observations, finite
rewards/state/telemetry, cached queries and repeated partial reset isolation.
Retain existing C3 pure and environment contract tests.

## Consequences and reconsideration

The system can now consume reproducible continuous paths through the existing
policy input interface. No tracking success, policy learning, promotion,
multi-seed robustness or real-flight feasibility follows from this engineering
gate. Revisit bounds or preview only with an explicit versioned decision when
dynamic diagnostics identify infeasible demand, insufficient anticipation,
quaternion/projection discontinuity, or systematic undercoverage. Never relax
old retention gates to compensate for C4 failures.
