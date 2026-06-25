# Pitch Moment Fit and OOD Zero Validation Design

## Decision

Add a low-dimensional pitch-moment correction on top of the current longitudinal effective-force integration:

```text
M_y = k_w M_{y,wing}^{prior} + k_t M_{y,tail}^{prior} + M_0
```

The purpose is not to identify a full six-axis aerodynamic model. The purpose is to make the simulator's pitch response credible enough that the already integrated effective `Fx/Fz` model stays inside its recorded input envelope during the validation cases that will precede reinforcement learning.

The validation target is strict: for the agreed Isaac validation suite, the effective-force model must report zero OOD samples, zero fallback samples, and zero OOD-triggered terminations. This is a validation-suite guarantee, not a global guarantee for arbitrary RL exploration. If the three-parameter moment model cannot meet this target, the implementation should report that failure and stop rather than widening the force envelope or hiding OOD samples.

## Scope

In scope:

- export fixed wing and tail pitch-moment priors from the same log states used by system identification;
- fit `k_w`, `k_t`, and `M_0` from real-log pitch-moment labels;
- select the regularization strength with validation logs and report held-out metrics once;
- integrate the moment correction into Isaac as `disabled`, `shadow`, and `replace_my` modes;
- keep the existing effective-force replacement contract unchanged: replace only body `Fx/Fz`, preserve `Fy/Mx/Mz`, and only optionally replace `My`;
- evaluate whether the corrected plant stays inside the real-flight envelope and produces zero force-model OOD in the validation scenarios.

Out of scope:

- fitting a neural moment residual;
- fitting wing or tail acting points;
- changing the frozen effective `Fx/Fz` model;
- expanding the effective-force feature envelope to make failures disappear;
- starting RL training before the OOD and response gates pass.

## Data and model contract

The fitting data must contain, per sample:

- real-log label `my_b` in the same body convention used by the system-identification data;
- fixed wing prior `M_{y,wing}^{prior}`;
- fixed tail prior `M_{y,tail}^{prior}`;
- log id, split id, phase, cycle-average flapping frequency, body pitch rate, airspeed or equivalent velocity fields, and control inputs needed for diagnostics.

Do not use the current frozen effective-force artifact's `prior_my_b` placeholder as a pitch-moment prior. It is not a fitted moment prior. The exporter must compute the component wing and tail prior moments from the same DeLaurier and tail code paths or parameter contracts used by Isaac.

The fitted artifact must contain:

- coefficient vector `[k_w, k_t, M_0]`;
- body-frame convention and units;
- prior-generation configuration and hashes;
- train/validation/test split provenance;
- selected ridge regularization parameter;
- pointwise fit metrics and response-validation metrics;
- bootstrap coefficient samples or covariance for later domain randomization.

## Fitting objective

Use whole-log splits, not random row splits. Fit on training logs only:

```text
min_theta sum_i w_i (my_i - [my_wing_i, my_tail_i, 1] theta)^2
          + lambda ((k_w - 1)^2 + (k_t - 1)^2)
```

where `theta = [k_w, k_t, M_0]`. Use equal-log weighting so that long logs do not dominate the coefficients. Do not penalize `M_0`. Select `lambda` by validation logs, then report the held-out test once.

Report at least three baselines:

1. raw prior: `M_y = M_{y,wing}^{prior} + M_{y,tail}^{prior}`;
2. bias-only prior: `M_y = M_{y,wing}^{prior} + M_{y,tail}^{prior} + M_0`;
3. fitted model: `M_y = k_w M_{y,wing}^{prior} + k_t M_{y,tail}^{prior} + M_0`.

If wing and tail priors are highly collinear, report the condition number and coefficient uncertainty. In that case, the fitted `M_y` prediction may still be useful, but individual `k_w` and `k_t` should not be overinterpreted.

## Isaac runtime design

The runtime already computes wing and tail moments before applying the effective-force correction. Add an optional pitch-moment correction after those raw components exist:

1. compute the raw simulator force and moment exactly as before;
2. compute or reuse `M_{y,wing}^{prior}` and `M_{y,tail}^{prior}`;
3. evaluate the fitted scalar pitch-moment model;
4. in `shadow`, log the predicted `My` but apply the raw simulator moment;
5. in `replace_my`, replace only body `My`;
6. leave `Mx` and `Mz` unchanged;
7. keep the effective `Fx/Fz` replacement independent from the moment correction mode.

This keeps the experiment interpretable: if OOD disappears, the explanation is improved pitch-rate dynamics, not an altered force envelope.

## Validation gates

The gates are ordered. A later gate does not compensate for a failed earlier gate.

1. **Exporter gate**: component priors are finite, non-placeholder, convention-checked, and aligned with `my_b`.
2. **Fit gate**: fitted model improves validation `My` metrics over the raw prior and does not rely on coefficient values that are numerically unstable.
3. **Offline response gate**: replayed or approximated pitch-rate and pitch response is closer to real logs than the raw prior for selected maneuvers.
4. **Isaac contract gate**: `disabled` and `shadow` apply identical state evolution; `replace_my` changes only `My`; effective `Fx/Fz` replacement still changes only `Fx/Fz`.
5. **OOD-zero gate**: in the agreed Isaac validation suite, effective-force OOD count is zero, fallback count is zero, OOD termination count is zero, and the robot stays inside the real-flight envelope for pitch rate, pitch attitude, speed, and angle-of-attack proxies.

If gate 5 fails, the next allowed action is diagnostic reporting. The first model extension to consider later is a scalar damping term such as `-c_q q`, but that is intentionally outside this design unless explicitly approved.

## Credibility claim after success

If all gates pass, the valid claim is:

> The corrected Isaac plant is credible for the tested pre-RL longitudinal operating envelope: it applies the frozen effective longitudinal force inside its recorded input envelope, uses a low-dimensional fitted pitch-moment correction to avoid pitch-rate-driven OOD, and matches real-flight response trends well enough for controlled RL pretraining experiments.

The claim is not:

> The simulator is a fully validated whole-aircraft aerodynamic model for all RL states.

