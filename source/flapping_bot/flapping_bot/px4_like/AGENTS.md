
PX4-Like Controller Guidelines

These instructions apply to controller, estimator, control-allocation, tuning-profile, and guidance-integration code in this directory.

1. Controller and plant separation

Controllers may consume states, estimates, references, and configured effectiveness information.

Controllers must not:

modify wing or tail aerodynamic coefficients;
modify mass, inertia, or geometry;
change the aerodynamic model to improve tracking;
silently compensate for a plant-model error by rewriting plant parameters.

Controller gains and controller behavior belong in controller profiles or typed configuration.

2. Preserve controller structure

Unless explicitly approved, preserve the existing PX4-like cascade structure and public interfaces.

Changes to one loop must identify interactions with:

angular-rate loop;
attitude loop;
speed and altitude control;
path or loiter control;
control allocation;
actuator saturation.

Do not retune multiple loop levels simultaneously without recording the sequence and objective.

3. Gain tuning rules

Different plant models may use independently tuned gains, but comparisons must use:

the same controller architecture;
the same tuning objective;
the same gain bounds;
the same actuator constraints;
the same tuning task set;
the same success and failure criteria.

Keep tuning scenarios separate from held-out evaluation scenarios.

Do not tune gains directly on the final reported test cases.

4. Safety and limits

All controller outputs must have explicit:

units;
limits;
sign conventions;
rate limits where applicable;
saturation handling;
anti-windup behavior for integrators.

Reset behavior for integrators, filters, and controller state must be deterministic and tested.

5. Determinism and diagnostics

Controller tests and tuning runs should record:

controller profile;
gains;
references;
measured or simulated states;
unsaturated outputs;
saturated outputs;
saturation duration;
reset state;
random seed when randomness is involved.

Avoid hidden gain scaling or condition-specific gain changes unless they are part of an explicitly documented scheduling method.

6. Required tests

Add applicable tests for:

sign correctness;
zero-error behavior;
saturation;
anti-windup;
reset behavior;
deterministic repeated execution;
known step or impulse responses;
interface compatibility.

Controller tests must use fixed plant assumptions or mocks. Do not change the plant model inside a controller test.
