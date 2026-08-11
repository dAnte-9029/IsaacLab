# PureRL GPU training roadmap

## Goal

Use GPU implicit dynamics to train candidate policies efficiently, while keeping CPU-native dynamics as the
authoritative plant for qualification and final promotion.

## Stages

1. **Evaluation contract**: remove terminal/reset mixed samples and freeze trace, statistics, and acceptance
   definitions.
2. **Wing-drive gate**: compare the prescribed 4 Hz and 5 Hz, 1 s trajectories at the 480 Hz physics rate.
3. **Paired plant gate**: capture CPU policy actions, replay them on GPU, and compare actuator, aerodynamic wrench,
   and short-horizon state responses.
4. **Zero-shot policy gate**: run the promoted CPU checkpoint on GPU over the complete C1 and C2a grids, with
   selected failure traces.
5. **Training-efficiency gate**: compare short PPO runs by simulated steps per second, wall time, numerical
   stability, and learning progress.
6. **Transfer pilot**: train C1 on GPU, then evaluate and, if needed, briefly fine-tune on CPU-native.
7. **Curriculum rollout**: only after the pilot passes, apply the GPU-to-CPU process at each curriculum stage and
   retain prior-stage rehearsal and retention evaluation.

Each stage ends with a separate report and decision before the next stage begins. A failed stage is a valid stop
result and does not authorize plant or reward tuning.

Status on 2026-08-11: stages 1 and 2 passed. Stage 3 completed but failed the paired-plant gate. The complete C1/C2a
zero-shot grid in stage 4 is therefore not started pending a decision on whether to revise the GPU candidate plant.
See `docs/audits/2026-08-11-pure-rl-gpu-implicit-paired-plant-gate.md`.

## Frozen evaluation definitions

- Physics-rate drive traces are 480 Hz; policy traces are 60 Hz. Reports must not mix the two.
- Every evaluation uses an explicit seed (default 0), and matched CPU/GPU runs use and record the same seed.
- A post-step state whose environment is `done` is an auto-reset state and is excluded from physical traces. The
  corresponding terminal outcome remains in per-case metrics.
- The drive gate runs each constant 4 Hz and 5 Hz target for 1 s. Steady statistics use `t >= 0.20 s`; the maximum
  absolute error and finite-state check use the full run. RMS and P99 pool both mirrored wing errors. Amplitude and
  phase come from a fundamental-frequency least-squares sine fit over the steady window. The common target uses the
  environment's URDF-derived 0.98-safe amplitude and is evaluated at each post-step state timestamp.
- The drive gate is RMS <= 0.30 deg, P99 <= 0.60 deg, full-run maximum <= 2.0 deg, amplitude error <= 2%, phase
  time error <= one physics step (2.08 ms), mirrored wing-sync RMS <= 0.10 deg, and no nonfinite values.
- Paired plant comparisons use the common valid time interval. Integrated vector error is
  `norm(integral(GPU - CPU) dt) / integral(norm(CPU)) dt`. Waveform RMS and peak errors use the norm of the
  vector difference relative to the RMS and peak CPU vector norm. Limits are 5% for force integral, 10% for
  moment integral, 10% RMS and 15% peak for both wrench waveforms. Over the first 1 s, maximum position,
  linear-velocity, attitude, and angular-velocity differences are limited to 0.25 m, 0.50 m/s, 5 deg, and
  1 rad/s. Policy actions must match within `1e-6`; requested and applied frequency must each match within
  `1e-5 Hz`. Tail and wing state differences are recorded as diagnostics but are not independent gates.
- C1 and C2a keep their existing hard task gates. Diagnostic deltas explain a result but do not relax a gate.
- CPU-native remains authoritative. Passing GPU gates permits candidate training only; final promotion still
  requires CPU-native evaluation.

The accepted backend boundary and artifact schema remain in
`docs/decisions/ADR-2026-08-11-pure-rl-gpu-implicit-zero-shot-qualification.md`. The implementation details remain
in `docs/plans/2026-08-11-pure-rl-gpu-implicit-zero-shot.md`.
