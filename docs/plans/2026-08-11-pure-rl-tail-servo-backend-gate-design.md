# PureRL Tail-Servo Backend Gate Design

Date: 2026-08-11

Status: completed and passed.

## Objective

Determine whether the residual CPU-native versus GPU-implicit paired-plant
error is dominated by different PhysX tail-joint responses. The experiment is
diagnostic only: it does not tune actuator gains, change the plant, or alter
training defaults.

## Alternatives

1. Add backend selection to the existing CPU action-step implementation. This
   reuses more code, but broadens a previously accepted CPU baseline.
2. Add a dedicated paired tail-servo runner. This keeps the old runner and
   plant unchanged while reusing its pure response summarizer. This is the
   selected approach.
3. Add tail cases to the policy paired-plant evaluator. This would mix actuator
   diagnosis with policy and free-flight divergence and is rejected.

## Experiment contract

Each backend runs in a fresh Isaac process. CPU uses the authoritative native
holonomic configuration. GPU uses the explicit phase-matched implicit
configuration. Both use fixed root, zero gravity, `480 Hz` physics, `60 Hz`
zero-order-held commands, seed zero, disabled randomization, disabled outer
action filtering and disabled outer action rate limiting.

Three environments independently exercise rudder, left elevon and right
elevon. Each surface receives the representative sequence
`0 -> +20 deg -> 0 -> -20 deg -> 0`, with `0.75 s` dwell. Runtime URDF soft
limits remain authoritative. Flapping frequency remains `2.66958 Hz`.

The four cases are:

- CPU-native, aerodynamics disabled;
- phase-matched GPU implicit, aerodynamics disabled;
- CPU-native, wing and tail aerodynamics enabled with `8 m/s` body-forward
  air-relative flow;
- phase-matched GPU implicit with the same aerodynamic condition.

## Metrics and frozen thresholds

Both backends must remain finite, obey the existing soft limits, settle within
`0.50 s`, overshoot by no more than `5%`, and have steady-state error no larger
than `0.25 deg`.

Paired traces must have identical timestamps and commands. Per surface and
mode, the thresholds are:

- maximum absolute actual-angle difference: `0.50 deg`;
- RMS actual-angle difference: `0.25 deg`;
- maximum steady-state actual-angle difference: `0.25 deg`;
- rise-time and settling-time differences: one `60 Hz` policy period
  (`16.67 ms`);
- with aerodynamics, tail-moment waveform relative RMS: `10%`;
- with aerodynamics, tail-moment waveform relative peak: `15%`.

The timing and steady-state limits reuse the accepted direct-action contract.
The aerodynamic moment limits reuse the stage-3 paired-plant thresholds. They
are frozen before evaluating the new traces.

## Artifacts and decision boundary

The runner writes one manifest, one summary, per-worker JSON, and compressed
NPZ traces. It refuses to overwrite an existing output directory and records
the selected GPU candidate. If no-aero cases fail, the next investigation is
the backend-specific implicit actuator response. If only loaded cases fail,
the next investigation is tail aerodynamic load coupling. No fix is included
in this stage.

## Result

The authoritative artifact is
`/home/zn/temp/pure_rl_tail_servo_gate_20260811_phase_matched_v6`. Both modes
passed. Maximum actual-angle difference was `8.54e-6 deg`; rise and settling
times were identical. At 8 m/s, tail-moment waveform RMS difference was
`1.20e-5` to `1.52e-5 percent` and peak difference was below
`4.50e-5 percent`.

An earlier failure was a diagnostic recording error: NumPy arrays created from
CPU tensors retained shared storage, so later tensor updates rewrote previous
samples. The runner now records owning snapshots, and a unit test verifies
that later source-tensor mutation cannot change a saved sample.
