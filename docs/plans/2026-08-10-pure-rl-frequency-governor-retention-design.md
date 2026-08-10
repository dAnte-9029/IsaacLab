# PureRL Frequency Governor and Curriculum Retention Design

## Scope

This design closes two approved gaps in the measured multibody PureRL workflow:

1. prevent the policy from using commanded flapping frequency as a per-wingbeat phase-modulation channel; and
2. make old-task rehearsal and forgetting checks explicit as later curriculum stages are introduced.

Real-flight data is not used to identify actuator dynamics in this change. Flight-envelope, closed-loop behavior, and validation distributions may inform later work, but the governor values below are provisional controller-side limits rather than identified actuator parameters.

## Frequency command contract

The policy continues to run at 60 Hz and outputs a normalized frequency command plus three direct tail-surface commands. Tail commands remain unfiltered. Only the frequency channel receives a slew governor after conversion to physical hertz.

For requested frequency `f_req`, previous applied frequency `f_prev`, policy interval `dt`, and configured rise/fall limits, the governor applies:

```text
delta = clamp(f_req - f_prev, -fall_limit * dt, rise_limit * dt)
f_applied = f_prev + delta
frequency_slew = delta / dt
```

The measured PureRL default is a symmetric 2.0 Hz/s limit. At 60 Hz this permits at most 0.0333 Hz change per policy step. This value is configurable and is explicitly not treated as an ES08MDII identification result. Other environment configurations retain their existing behavior unless they opt in.

The applied physical frequency is converted back to the normalized action representation before entering action history and applied-action telemetry. Requested action remains available separately through the raw policy action buffer.

## Reward contract

The old normalized frequency-action-delta term is replaced by a physical slew term:

```text
frequency_slew_penalty = (frequency_slew_hz_per_s / frequency_slew_scale_hz_per_s) ** 2
```

The default scale is 2.0 Hz/s and the existing weight remains 0.01. Consequently, a command held constant has zero penalty and motion at the governor limit has an unweighted penalty of one. Tail-action delta regularization remains in normalized joint-range units.

Telemetry reports requested and applied frequency, signed physical slew, governor-limited fraction, and the physical-slew penalty. Evaluation continues to retain normalized full-action delta for backward diagnostic value, but it is no longer the frequency reward quantity.

## Rehearsal contract

Each curriculum stage receives a configured probability of sampling the earlier straight-flight task. The PureRL default schedule is:

```text
stage 0: 0.50
stage 1: 0.35
stage 2: 0.25
stage 3: 0.20
```

This schedule prevents a hard distribution switch. It does not guarantee retention by itself, so checkpoint promotion also requires evaluation against all completed stages.

## Retention matrix contract

Evaluation records are keyed by `(training_stage, checkpoint, evaluation_stage)`. For a checkpoint produced at stage `k`, records for every evaluation stage from 0 through `k` are mandatory. Missing or duplicate cells fail closed.

The matrix reports:

- each cell's success-gate result and score;
- the diagonal score for the evaluation task when it was first learned;
- signed and clipped score drop relative to that diagonal;
- a row-level retention result requiring all current and earlier success gates to pass.

Score drop is diagnostic in this change; checkpoint promotion is gated by the established per-stage success gates. A later experiment may add a maximum allowable score drop only after real score distributions are available.

## Compatibility

The frequency governor changes the measured PureRL default action dynamics, so existing checkpoints should be treated as pre-governor artifacts and retrained for comparable results. The general action contract and non-opt-in environments preserve their baseline behavior. The rehearsal and retention utilities are designed to be reused when subsequent trajectory and disturbance curricula are connected to the measured multibody environment.
