# ADR: C4 random reset sampling and 1.2-second preview

Status: accepted for implementation by the user's execution request.

## Context

The original C4 engineering adapter sampled candidates but raised on the first
kinematic rejection. A controlled fixed-96 comparison of the unchanged
`model_175.pt` improved success from 56/96 to 96/96 with a 1.2 s preview.
These results establish neither random-path generalization nor old-task
retention under a changed horizon. The original 2026-09-09 path-contract ADR
continues to define geometry, screen thresholds, 300 m tables and 140 m completion.

## Decision

C4 without `pure_rl_continuous_curve_file` samples new recipes at initialization
and each reset. Curves remain constant within an episode. Sampling remains the
existing harmonic/chirped family; this is not arbitrary Cartesian spline sampling.
`ContinuousCurveSamplingConfig` exposes these uniform candidate ranges:

- heading-rate amplitude: 0.005--0.009 rad/m;
- slope amplitude: 2--4 degrees;
- heading and slope wavelengths independently: 80--120 m;
- heading/slope phases and initial yaw independently: 0--2*pi;
- horizontal and vertical mirrors independently equiprobable;
- second harmonic fraction: 0--0.25; chirp: 0--0.05.

The phase relation is unrestricted because both phases are sampled independently.
Heading and slope biases remain zero. Existing environment flap-phase reset
behavior is preserved; file replay retains its explicit phase values.

Screen the full candidate path on the requested device/dtype. Retry only failed
rows up to 32 attempts per row, then raise with failed row IDs; no clipping or
fallback task. Structural errors are not converted into retries. Accepted rows
are drawn from the candidate distribution conditioned on the screen, not from
an unchanged uniform distribution. Seed reproducibility assumes the same device
and random/reset call sequence; different reset order can change future draws.
No new claim of dynamic/aerodynamic feasibility is made.

Fixed JSON loading remains strict and never replaces a rejected recipe. CLI
random export now uses the accepted sampler, with float64 screening for replay.
The environment passes its typed sampling configuration into initialization and
partial reset; the existing row writer preserves unselected rows.

`pure_rl_spatial_preview_times_s` is configurable, finite, positive, increasing
and exactly five entries. C4 defaults to (0.24, 0.48, 0.72, 0.96, 1.20) s.
Old tasks keep (0.12, 0.24, 0.36, 0.48, 0.60) s. The query's existing bounded
speed and table-end interpolation apply. Actor size stays 555; no weights,
reward, plant, termination, training runner or old task defaults change.
The checkpoint evaluator explicitly retains its historical CLI semantics:
`--preview-time-scale 1` means 0.6 s, `2` means 1.2 s regardless of C4 defaults.

## Validation and consequences

99 pure/contract tests passed before the final CLI and device coverage additions;
31 focused generator/evaluator tests passed afterward, including CPU/CUDA and
float32/64. Native C4 test passed (1 test), covering fixed replay followed by two
random partial resets and eight steps of path immutability/finite observations.
Existing C3 native geometry/reward/reset test passed (1 test, 2 deselected).
Runtime emitted existing shutdown/Gym warnings. No PPO was launched.

An independent 96-path sample with seed 104729 passed the geometric screen and
was exported under `logs/continuous_curve/20260910_random_generator/` for a
separate long-preview zero-shot run. Its tracking results are pending at launch.
Old C1--C3 *long-preview checkpoint* retention suites were not run; old defaults
remain short-preview and C3 default runtime regression passed. Changing those
old defaults would require separate retention evidence. Revisit ranges only
with versioned experiments, retaining the original frozen evaluation targets.
