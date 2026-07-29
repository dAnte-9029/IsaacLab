# ADR: Ideal coupled drive for the measured wing multibody plant

- Status: Accepted
- Date: 2026-07-29
- Scope: Wing actuation and left/right mechanical coupling

## Context

The measured wing plant must allow PhysX to integrate wing acceleration and
the resulting reaction on the body. Writing wing position and velocity into
the simulation on every physics step bypasses that dynamics path.

The physical aircraft has one motor and a gear train inside the body. The
gear train imposes opposed left/right strokes, but the available measurements
do not identify motor electrical dynamics, gear elasticity, backlash, friction
or torque limits.

## Decision

Add an explicit `ideal_coupled_drive` variant on top of
`measured_wing_multibody`. Retain `kinematic_override` as the default.

The ideal variant has:

- one implicit PD driver on `left_wing`;
- one zero-stiffness, zero-damping passive actuator entry on `right_wing`;
- a hard `PhysxMimicJointAPI` relation
  `q_left + q_right = 0`;
- position and velocity targets sent only to the left driver each physics
  step;
- normal position targets for rudder and tail joints;
- no per-step `write_joint_state_to_sim` for wing motion.

The hard mimic schema is applied by a project-local adapter after the source
robot is spawned and before PhysX initializes or the source environment is
cloned. The upstream Isaac Lab asset is not modified.

The numerical ideal-driver settings are:

```text
stiffness = 2000 N m/rad
damping = 20 N m s/rad
effort_limit = 1000 N m
velocity_limit = 200 rad/s
position iterations = 16
velocity iterations = 4
```

These are solver settings for prescribed-motion approximation, not identified
motor or gearbox parameters. The high effort limit prevents the ideal
reference tracker from being interpreted as the real vehicle's actuator
capacity.

Writing a consistent joint state during environment reset remains permitted.
It establishes the initial condition and is not a per-step kinematic motion
source.

## Consequences

- PhysX now integrates measured wing mass and inertia from a single driven
  generalized coordinate.
- Constraint reactions transmit wing inertia to the body.
- Asymmetric loads cannot make the two wing angles diverge independently.
- The common-coordinate driver torque is available as a diagnostic, but must
  not be treated as real motor shaft torque.
- Baseline configs and their kinematic override remain unchanged.

At the formal `1/240 s` physics step, the 5 Hz no-aerodynamics integration
test measured `0.010942 deg` maximum synchronization error and
`0.289652 deg` maximum left-driver tracking error. The corresponding ideal
joint effort reached approximately `142.43 N m`.

## Alternatives considered

- Two independent PD wing servos plus a synchronization penalty: rejected
  because the physical mechanism has one common drive coordinate and does not
  allow independent wing stroke.
- Keep per-step state writes: rejected because this does not provide a
  force-driven multibody plant.
- Model the physical motor and gear train now: deferred because the required
  motor, ratio, loss, backlash and compliance measurements are unavailable.
- Encode the relation by modifying the upstream URDF: rejected because project
  behavior belongs in the project extension.

## Validation requirements

- The real robot asset must initialize with one driver and one passive right
  wing.
- The measured runtime mass must remain `0.90415 kg`.
- At 2 Hz and at 5 Hz with the formal time step, maximum
  `abs(q_left + q_right)` must remain below `0.1 deg`.
- Body vertical and pitch response must be nonzero with aerodynamics and
  gravity disabled.
- The formal environment must instantiate and step without invoking the wing
  kinematic override.

## Remaining scope

The current aerodynamic model still derives wing kinematics from the command
and applies an equivalent combined wrench to `base_link`. A later change must
use measured joint motion for aerodynamic inputs and apply each wing wrench to
its wing link before motor-load or joint-load conclusions are made.

## Reconsideration triggers

Replace this ideal drive with a physical actuator variant when motor speed and
torque data, gear ratio and efficiency, transmission friction, backlash or
compliance become available.
