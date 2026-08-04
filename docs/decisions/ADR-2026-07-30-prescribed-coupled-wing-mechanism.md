# ADR: Prescribed coupled wing mechanism for plant validation

- Status: Accepted
- Date: 2026-07-30
- Scope: Measured-wing plant validation

## Context

The accepted `ideal_coupled_drive` uses a hard opposed PhysX mimic relation,
but its common coordinate remains a finite-stiffness implicit PD degree of
freedom. True-fixed aerodynamic tests show that per-wing force loading can
drive this common coordinate away from its reference and enter a
step-sensitive load/drive feedback loop.

The physical aircraft has one motor and a rigid gear/linkage path, but motor
electrical dynamics, losses, backlash, compliance and torque limits are not
identified. The immediate objective is to validate the measured multibody
plant for a known flap trajectory, not to predict motor feasibility.

## Decision

Add an explicit `prescribed_coupled_drive` variant. It constructs both wing
targets from one common coordinate:

```text
q_left = q_left_mid + q
q_right = q_right_mid - q
```

It removes both wing PD drives. Each physical wing coordinate is prescribed by
a time-varying zero-width PhysX joint-position limit generated from the same
analytical sine reference. The limits are updated through the articulation
tensor API before each physics step; neither physical wing joint state is
overwritten during the step.

The existing hard mimic remains part of `ideal_coupled_drive`, but is not
stacked with the two prescribed limits. An initial implementation that drove
the left limit and propagated the motion through the mimic produced
`0.912 deg` synchronization error in a 2 Hz smoke run even though left
tracking error was only `0.000237 deg`. Prescribing both physical coordinates
avoids this serial constraint-iteration error.

Add a matching `prescribed_per_wing_link` aerodynamic coupling mode. It uses
the analytical prescribed position, velocity and acceleration as DeLaurier
inputs, maps the resulting left/right wrenches to the physical wing links, and
lets PhysX transmit the limit and mimic reactions to the body.

This is an ideal mechanism with unlimited actuation authority. It is not a
motor model and it does not predict motor torque, electrical power, gearbox
efficiency or tracking loss.

The established `kinematic_override`, `ideal_coupled_drive`, actual-motion
diagnostics and default plant remain available and unchanged.

## Alternatives considered

- Increase the current PD gains: rejected because it preserves the identified
  finite-compliance load/drive loop and turns solver tuning into an unmeasured
  actuator assumption.
- Add a detailed DC motor and gearbox model: deferred because those parameters
  are unavailable and are unnecessary for prescribed-trajectory plant
  validation.
- Overwrite both massive wing joint states every step: rejected because it
  does not ask PhysX to resolve the wing constraint reaction.
- Apply only an equivalent aerodynamic wrench to the base: retained as a
  baseline, but it bypasses wing-joint aerodynamic loading.

## Consequences

- Left/right synchronization and the common flap trajectory are generated
  from one coordinate and imposed as solver constraints rather than PD
  tracking objectives.
- Aerodynamic loading cannot change the DeLaurier motion input in this variant.
- PhysX still carries measured wing mass and inertia and applies aerodynamic
  wrenches on the wing links.
- The required ideal actuation effort is not directly exposed by Isaac Lab's
  joint actuator buffer because the wing actuators are passive.
- A moving zero-width joint limit is an experimental PhysX realization. Its
  constraint reaction, time-step convergence and conservation behavior must
  be validated before this variant can replace any default.

## Assumptions

- The real gear/linkage imposes negligible left/right compliance for the
  plant-validation bandwidth.
- The commanded sine trajectory is the intended mechanism output.
- Unlimited ideal mechanism authority is acceptable for plant validation.
- Motor sizing, efficiency and attainable trajectory remain outside this
  variant's claim.

## Validation requirements

- The real articulation initializes with two passive wing actuators and two
  opposed position constraints generated from one common coordinate.
- No per-step wing `write_joint_state_to_sim` and no wing position or velocity
  drive target is used.
- The prescribed common-coordinate error and opposed synchronization error
  are measured at 2, 3, 4 and 5 Hz.
- The true-fixed eight-cycle aerodynamic matrix is bounded and converges with
  time step.
- A no-aerodynamics, no-gravity floating-base test checks linear and angular
  momentum closure for the complete articulation.
- Reaction behavior is compared against the Amini reduced inertial reference
  before default promotion.

## Reconsideration triggers

Replace or supplement this ideal mechanism when measured motor/gearbox
parameters become available, when the real linkage shows meaningful
compliance or backlash, or when the moving-limit implementation fails the
reaction or time-step validation gates.
