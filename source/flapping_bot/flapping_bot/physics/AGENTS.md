# AGENTS.mdPhysics-Model Guidelines

These instructions apply to aerodynamic, force, moment, actuator, and rigid-body model code in this directory.

## 1. Required physical metadata

Every new force or moment output must make the following explicit in code, type definitions, or docstrings:

* physical quantity;
* SI unit;
* coordinate frame;
* application point for force;
* reference point for moment;
* sign convention;
* tensor shape.

Do not use ambiguous names such as `force`, `moment`, `tau`, or `point` when multiple frames or reference points are possible. Prefer names such as:

* `force_b`
* `moment_b_about_cg`
* `strip_point_b_from_cg`
* `force_wing`

## 2. Force and moment consistency

When changing a force model:

* identify whether the corresponding moment must change;
* preserve force-resultant consistency;
* preserve or explicitly redefine the force application line;
* document any moment-closure assumption;
* do not present an assumed load distribution as measured truth.

For strip-based models, test where applicable:

[
\sum_i \mathbf{f}_i = \mathbf{F}
]

and

[
\sum_i \mathbf{r}_i \times \mathbf{f}_i = \mathbf{M}
]

The moment reference point must be explicit.

## 3. Model-evidence boundaries

Distinguish clearly between:

* data-supported quantities;
* analytical-prior quantities;
* calibrated parameters;
* closure assumptions;
* numerical stabilization;
* controller compensation.

Do not describe an unvalidated closure or tuned coefficient as physically identified.

Do not add a coefficient solely because it improves a flight trajectory. Any new empirical coefficient must have:

* a defined physical or numerical role;
* an explicit configuration entry;
* a documented admissible range;
* a validation or sensitivity plan.

## 4. Baseline and ablation support

Physics changes must preserve explicit model variants needed for comparison, including where relevant:

* original prior;
* corrected resultant force;
* corrected force with alternative moment closures.

When correction is disabled, the original prior path should be reproduced within a stated tolerance.

Do not overwrite the only implementation of the original model.

## 5. Numerical behavior

Handle explicitly:

* zero or near-zero airspeed;
* zero or near-zero resultant components;
* denominator singularities;
* sign changes;
* strip-force cancellation;
* batch size one;
* CPU and CUDA devices;
* float32 and relevant float64 tests;
* clipping and smoothing transitions.

Avoid hard discontinuities unless required by the model and documented.

Any clamp, epsilon, smoothing width, or fallback value must be named and configurable or justified as a fixed numerical constant.

## 6. Prohibited coupling

Physics code must not:

* import PX4-like controller modules;
* read PID gains;
* access path-tracking objectives;
* inspect rewards or task success;
* modify mission references;
* tune itself based on whether a trajectory visually appears stable.

## 7. Required validation

For pure-model changes, add targeted tests for applicable properties:

* conservation;
* symmetry or left-right mirror behavior;
* finite outputs;
* frame transformations;
* known hand-calculated cases;
* disabled-feature regression;
* continuity near branch or switching boundaries.

A test must not compute its expected value by calling the same implementation path being tested.

Before implementation, identify the relevant modeling ADR. If no approved ADR exists for a new physical assumption, stop and request a design decision.
