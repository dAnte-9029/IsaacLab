# ADR: Power-consistent PhysX coupling for the sinusoidal phase drive

- Status: Accepted
- Date: 2026-07-31
- Scope: Experimental coupling between the virtual mechanism phase and measured PhysX wing links
- Extends: `ADR-2026-07-31-sinusoidal-phase-speed-drive.md`

## Context

The accepted sinusoidal phase-speed model defines a hidden mechanism phase and
exact opposed sine kinematics. PhysX supports the measured wing rigid bodies and
the linear left/right mimic relation, but it does not provide a native nonlinear
joint relation of the form `q=Gamma sin(phi)`. The moving-limit experiment did
not produce a position-velocity-consistent joint state.

A project-side coupling must therefore transmit force between the virtual phase
and the physical wing coordinate without overwriting PhysX joint state. It must
also avoid counting the measured wing inertia both in PhysX and in the virtual
phase equation.

## Decision

Add an explicit experimental `sinusoidal_phase_speed_drive` variant. It uses:

- measured body and wing rigid-body properties;
- the hard PhysX opposed-wing mimic relation;
- passive zero-stiffness PhysX wing actuators;
- actual PhysX common wing position and velocity;
- an explicit project-side nonlinear constraint effort;
- the accepted virtual-throttle phase-speed model;
- no per-step joint-state write and no moving joint limit.

Define the common physical wing coordinate from the left joint:

```text
q = q_left - q_left_mid
q_dot = q_dot_left
```

The hard mimic enforces the right coordinate with opposite sign. The nonlinear
constraint residuals are

```text
C = q - Gamma sin(phi)
C_dot = q_dot - Gamma cos(phi) omega
```

The effort applied to the common physical wing coordinate is

```text
Q_c = -K_c C - D_c C_dot
```

Only the left joint receives `Q_c`; the hard mimic transmits the opposed motion
to the passive right wing. Since `q_right=-q_left`, this left-joint effort is
the common-coordinate generalized effort.

The equal and opposite phase-coordinate reaction is

```text
Q_phi_c = -Gamma cos(phi) Q_c
```

The phase-speed core accepts this reaction by passing `-Q_c` through its
common-joint virtual-work projection.

The coupling is power-consistent:

```text
Q_c q_dot + Q_phi_c omega
  = -d(0.5 K_c C^2)/dt - D_c C_dot^2
```

The constraint gains use the measured two-wing common-coordinate inertia and a
discrete repeated pole at `50 Hz`, with damping ratio one. The
effort is limited by the existing `1000 N m` numerical protection limit. These
are numerical constraint settings, not motor parameters.

For this coupled variant, measured wing inertia belongs exclusively to PhysX.
The virtual phase core therefore disables its transformed-wing inertia term and
retains only an explicit constant numerical phase inertia:

```text
J_phase_numerical = 0.00435938889653 kg m^2
```

This equals the peak transformed measured-wing inertia only to define an
initial numerical scale. It is not identified drivetrain inertia and is not a
second copy of wing inertia in the phase equation. It must receive a
sensitivity test before any physical frequency-response claim.

Aerodynamic kinematics use actual PhysX joint position, velocity and
acceleration. Per-wing DeLaurier wrenches are applied to their physical wing
links. No aerodynamic torque feedforward is used by the phase-speed regulator.

The default plant and all previous drive variants remain unchanged.

## Alternatives considered

- Native PhysX mimic or fixed tendon for the sine relation: unavailable because
  those relations are linear in joint coordinates.
- Moving zero-width joint limits: rejected because the tested joint velocity
  was inconsistent with imposed position.
- Per-step joint position and velocity writes: rejected because they bypass
  force-driven multibody motion.
- Reuse transformed wing inertia in the phase equation while feeding back the
  constraint reaction: rejected because it counts wing inertia twice.
- Omit the phase reaction: rejected because it breaks internal power
  consistency and prevents wing loading from affecting actual frequency.
- Add detailed linkage bodies: deferred until linkage geometry is frozen.

## Consequences

- Actual PhysX joint state remains position-velocity consistent.
- Left/right opposition is still a hard linear constraint.
- The sine relation is approximate at finite constraint bandwidth and time
  step; its residual is a required diagnostic.
- Wing inertia and aerodynamic loading can affect actual phase speed through
  the constraint reaction.
- The numerical phase inertia and constraint bandwidth influence transient
  frequency response and must not be presented as measured actuator behavior.
- Constraint damping intentionally dissipates energy when the sine relation is
  violated.

## Assumptions

- One explicit common-coordinate effort on the left joint is correctly
  transmitted through the hard mimic.
- Fifty hertz is adequate for the initial 0--5 Hz mechanism gate.
- The peak-inertia-scale constant phase inertia is only a numerical closure.
- Reset may write a consistent initial joint state; no per-step state writes are
  permitted.

## Validation requirements

- Pure tests for residual sign, effort sign, saturation and internal power.
- Contract tests confirming passive wings, hard mimic, explicit effort, no
  moving limits and no per-step wing state write.
- No-aerodynamics, no-gravity, fixed-root tests at 2 and 5 Hz.
- Maximum left/right synchronization error below `0.1 degree`.
- Maximum sine-constraint position error below `0.5 degree`.
- PhysX joint velocity versus finite-difference position RMS error below five
  percent.
- Bounded phase rate, joint state, effort and constraint energy.
- A later floating-base test must check whole-articulation momentum closure.
- A later aerodynamic 2/3/4/5 Hz time-step matrix must use actual joint
  kinematics and per-wing link wrenches.

## Initial gate evidence

The initially proposed one-percent constant phase inertia,
`4.35938889653e-5 kg m^2`, failed the 2 Hz, `1/480 s` fixed-root gate. Actual
phase frequency grew to thousands of hertz and the effort saturated at
`1000 N m`. This value is prohibited for the coupled PhysX variant.

Using `0.00435938889653 kg m^2` made the same coupling bounded. At 2 Hz with a
30 Hz constraint, the same-time sine error was `0.170 degree`, left/right
synchronization error was numerically zero, and joint-velocity finite-difference
RMS disagreement was `1.35 percent`. At 5 Hz with a 50 Hz constraint, the sine
error was `0.464 degree`, joint-velocity disagreement was `3.49 percent`, and
maximum effort was `12.04 N m`. The instantaneous 5 Hz phase frequency still
ranged from `4.349` to `5.988 Hz`; this is evidence of numerical mechanism
behavior, not validated motor response.

## Reconsideration triggers

Reconsider this coupling if its constraint error requires stiffness that is not
time-step convergent, if the numerical phase inertia dominates frequency
response, if internal power does not close, or if explicit linkage geometry
becomes available.
