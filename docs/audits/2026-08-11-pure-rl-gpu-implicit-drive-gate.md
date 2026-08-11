# PureRL GPU implicit fixed-root drive gate

Date: 2026-08-11

## Scope

This gate compares only the CPU-native and GPU-implicit wing mechanisms under the shared PureRL plant setup. Each
fresh process runs one 4 Hz or 5 Hz case for 1 s at 480 Hz with a fixed root, 8 m/s body-x relative flow, enabled
per-wing DeLaurier loads, disabled gravity and tail aerodynamics, seed 0, and the same URDF-derived 29.3832 degree
safe stroke amplitude.

The first 0.20 s is excluded from steady RMS, P99, amplitude, phase, and synchronization metrics. Full-run maximum
error and finite-state checks use all 480 samples. The target is the analytical sinusoid evaluated at each
post-step state timestamp; the backend-specific internal command is retained separately in each trace.

## Results

| Backend | Frequency | Full max error | Steady RMS | Steady P99 | Amplitude error | Phase time error | Sync RMS | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CPU native | 4 Hz | 0.0400 deg | 0.0281 deg | 0.0400 deg | 0.136% | 0.0014 ms | <1e-6 deg | pass |
| CPU native | 5 Hz | 0.0625 deg | 0.0440 deg | 0.0625 deg | 0.212% | 0.0015 ms | <1e-6 deg | pass |
| GPU phase-matched implicit | 4 Hz | 0.0510 deg | 0.0379 deg | 0.0505 deg | 0.184% | 0.0015 ms | 0.00067 deg | pass |
| GPU phase-matched implicit | 5 Hz | 0.0699 deg | 0.0516 deg | 0.0690 deg | 0.248% | 0.0056 ms | 0.00093 deg | pass |

All four cases passed the frozen drive thresholds and remained finite. The phase-matched candidate separates the
current-step aerodynamic reference from the implicit actuator target and advances the latter by an additional
`0.18` physics step. Relative to the preceding one-step-only candidate, GPU full maximum error fell from
`0.2886` to `0.0510 deg` at 4 Hz and from `0.3606` to `0.0699 deg` at 5 Hz.

## Claim boundary

The GPU implicit mechanism passes this narrow loaded trajectory gate. This does not establish aerodynamic-wrench,
free-flight, closed-loop policy, PPO-training, or CPU-transfer equivalence. The paired action-replay plant gate is
still required before full zero-shot policy qualification.

Generated evidence is under `/home/zn/temp/pure_rl_drive_gate_20260811_phase_matched_v5/`; `summary.json` and the four NPZ
traces are the numerical source for this audit.
