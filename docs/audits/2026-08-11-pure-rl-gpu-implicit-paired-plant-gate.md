# PureRL GPU implicit paired-plant gate

Date: 2026-08-11

## Decision

The phase-matched GPU candidate passes the fixed-root drive gate and the
isolated tail-servo gate, but still fails the frozen free-flight paired-plant
gate. It is not yet qualified for broad GPU policy training. Thresholds were
not changed after observing the results.

The authoritative paired artifact is:

`/home/zn/temp/pure_rl_paired_plant_gate_20260811_phase_matched_v4`

It uses seed 0 and checkpoint
`logs/rsl_rl/flapping_bot_straight_flight/2026-08-10_19-45-14_pure_rl_c2a_seed0/model_1500.pt`.
CPU-native generated the 60 Hz actions and GPU replayed those exact actions in
fresh processes.

## Current results

| Case | Force RMS | Moment RMS | Moment peak | 1 s position | 1 s velocity | 1 s attitude | 1 s angular rate | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `c1_h0_p0` | 6.34% | 10.96% | 16.47% | 0.279 m | 0.376 m/s | 6.07 deg | 0.680 rad/s | fail |
| `c1_h2_p2` | 4.83% | 9.26% | 7.58% | 0.111 m | 0.437 m/s | 6.98 deg | 0.510 rad/s | fail |
| C2a level | 6.34% | 10.96% | 16.47% | 0.279 m | 0.376 m/s | 6.07 deg | 0.680 rad/s | fail |
| C2a descent | 6.22% | 10.96% | 16.47% | 0.279 m | 0.376 m/s | 6.07 deg | 0.680 rad/s | fail |
| C2a climb | 6.35% | 10.97% | 16.47% | 0.279 m | 0.376 m/s | 6.07 deg | 0.680 rad/s | fail |

The frozen limits are 10% force/moment RMS, 15% force/moment peak,
0.25 m position, 0.50 m/s velocity, 5 deg attitude and 1 rad/s angular
velocity over the first second. Integrated force and moment errors remain
inside their 5% and 10% limits in all five cases.

## Diagnostics completed

The phase-matched candidate now uses a current-step aerodynamic reference and
a separately advanced implicit actuator target. An empirically measured
`0.18` physics-step residual lead reduced fixed-root GPU maximum tracking error
to `0.051 deg` at 4 Hz and `0.070 deg` at 5 Hz. In the paired test, maximum
wing-position difference fell from about `0.264 deg` to `0.141 deg` for the
phase-zero cases.

The isolated tail artifact
`/home/zn/temp/pure_rl_tail_servo_gate_20260811_phase_matched_v6` passed. The
maximum CPU/GPU tail-angle difference was `8.54e-6 deg`; at 8 m/s, tail-moment
RMS differences were only `1.2e-5` to `1.5e-5 percent`. Earlier huge tail
moment differences came from retaining non-owning NumPy views of CPU tensors;
the diagnostic runner now records owning snapshots and has a regression test.

Reset pose, velocity, flap phase and joint states are deterministic under the
paired schedule. Raw actions and requested/applied frequency match exactly.
The remaining discrepancy appears within the first wingbeat as root angular
velocity and attitude divergence even when wing and tail angles are close.
This is consistent with different root reaction dynamics between the native
holonomic constraint and the implicit actuator plus mimic relation.

## Consequence

Do not start GPU implicit training or relax the paired thresholds. The route is
rejected by `ADR-2026-08-11-reject-pure-rl-gpu-implicit-training.md`; the
previously proposed free-root reaction-wrench diagnostic is cancelled. Resume
curriculum training on CPU-native and limit acceleration work to changes that
preserve that plant.
