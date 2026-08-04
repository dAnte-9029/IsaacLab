# ADR: Ideal inverse-dynamics phase drive

- Status: Accepted
- Date: 2026-08-01
- Scope: Experimental measured-wing mechanism drive and aerodynamic coupling
- Supersedes for promotion: `ADR-2026-07-31-sinusoidal-phase-physx-coupling.md`

## Context

The power-consistent sinusoidal phase coupling gives PhysX a consistent wing
position and velocity state, but its numerical phase inertia and finite
penalty constraint allow aerodynamic hinge loading to perturb phase speed and
distort the sinusoidal trajectory. The narrow 2 Hz per-wing aerodynamic gate
became unbounded and was strongly time-step dependent.

The physical aircraft uses one motor and a rigid gear/linkage mechanism, but
motor electrical parameters, gear ratio, reflected drivetrain inertia,
friction, backlash and compliance have not been measured. The current purpose
is a measured body/wing multibody plant with a reasonable ideal mechanism, not
motor sizing or drivetrain identification.

## Decision

Add a new explicit `ideal_inverse_dynamics_phase_drive` variant while retaining
all established drive variants and the current default.

The ideal mechanism owns phase and frequency. Virtual throttle maps to a target
frequency as before. Actual frequency follows a first-order command response
with the accepted 0.15 s two-percent settling-time convention and is not
perturbed by aerodynamic or inertial load:

```text
f_target = f_max clamp(u, 0, 1)
df/dt = (f_target - f) / tau_f
dphi/dt = 2 pi f
tau_f = settling_time / 4
```

The exact mechanism reference remains:

```text
q_ref = Gamma sin(phi)
qdot_ref = Gamma cos(phi) omega
qddot_ref = Gamma [cos(phi) alpha - sin(phi) omega^2]
q_left = q_left_mid + q_ref
q_right = q_right_mid - q_ref
```

PhysX continues to own the measured rigid-body state. Both wing actuators are
passive, the hard linear mimic enforces opposed left/right motion, and no
per-step joint-state write or moving joint limit is used.

The drive reads actual PhysX joint position and velocity. It forms a desired
common-coordinate acceleration from the analytical reference plus a numerical
tracking correction:

```text
qddot_des = qddot_ref
           + kp (q_ref - q)
           + kd (qdot_ref - qdot)
```

PhysX generalized mass, Coriolis/centrifugal compensation and gravity
compensation tensors are reduced onto the opposed common coordinate. For a
fixed base this is a direct joint-space projection. For a floating base the
unactuated base acceleration is eliminated with the Schur complement. The
resulting mechanism effort is:

```text
Q_drive = M_common qddot_des + h_common - Q_external_common
```

`Q_external_common` includes the aerodynamic hinge load already applied to the
physical wing links, so subtracting it is load feedforward rather than removal
of the external body reaction. PhysX still receives each wing force and moment
at the corresponding wing COM and transmits the equal-and-opposite actuator
reaction through the articulation.

The DeLaurier position and velocity inputs remain actual PhysX joint state. Its
acceleration-dependent apparent-mass input uses `qddot_ref`, because this
variant assumes a rigid ideal mechanism and because feeding the explicit
PhysX acceleration back into the aerodynamic force created the failed
step-sensitive acceleration loop. Actual PhysX joint acceleration remains a
diagnostic and is not relabeled as the aerodynamic input.

The numerical tracking bandwidth and 1000 N m effort limit are solver settings,
not identified motor, gearbox or linkage properties. Reported effort is the
required common mechanism output torque. It is not motor-shaft torque,
electrical power or a feasibility guarantee.

## Alternatives considered

- Retune numerical phase inertia or the phase-speed PI loop: rejected because
  those unmeasured parameters would determine the apparent physical response.
- Keep actual joint acceleration in the explicit DeLaurier loop: rejected
  because the narrow gate was unbounded and strongly time-step dependent.
- Use moving zero-width limits: rejected because PhysX did not report velocity
  consistent with the imposed position sequence.
- Write wing state each step: rejected because it bypasses force-driven
  multibody reaction.
- Add a detailed motor and gearbox: deferred until its physical parameters are
  measured and because it is outside the current plant objective.
- Add explicit linkage geometry or a custom nonlinear PhysX constraint:
  deferred until mechanism geometry is frozen or the ideal effort realization
  cannot satisfy the validation gates.

## Consequences

- Frequency commands are smooth and load independent, matching an ideal
  speed-controlled mechanism rather than an unidentified motor.
- Actual PhysX position and velocity remain solver-consistent and aerodynamic
  loads continue to act on the physical wing links.
- Tracking is still an effort-driven numerical approximation, not a native
  nonlinear holonomic constraint; the residual must be reported.
- Required mechanism torque and mechanical power become useful diagnostics,
  but cannot be interpreted as motor-shaft quantities.
- The previous phase-penalty variant remains available as a failed comparison
  baseline. The established default remains unchanged.

## Assumptions

- One phase defines the rigid linkage output over 0--5 Hz.
- The hard mimic correctly represents the opposed left/right relation.
- The analytical acceleration is the appropriate apparent-mass input under the
  ideal rigid-mechanism assumption.
- Joint damping, friction and contact are absent from the PhysX inverse
  dynamics tensors and remain outside the initial gate.
- The base generalized-force ordering and common-coordinate reduction are
  checked against the local PhysX tensor API before floating-base claims.

## Validation requirements

- Pure float32/float64 tests for the first-order frequency step, unwrapped
  phase, analytical kinematics, fixed-base projection, floating-base Schur
  reduction, effort sign, finite outputs and effort saturation.
- Contract tests confirming passive wing actuators, hard mimic, explicit
  effort, prescribed aerodynamic acceleration, no moving limits and no
  per-step wing-state write.
- Fixed-root, no-aerodynamics tests at 2 and 5 Hz with maximum position error
  below 0.5 degree, opposed synchronization error below 0.1 degree, joint
  velocity versus finite-difference RMS disagreement below five percent and no
  effort saturation.
- A narrow fixed-root aerodynamic gate at 2 Hz, 1/480 s, 8 m/s and zero angle
  of attack must remain finite for eight cycles before any frequency or
  time-step matrix is run.
- If the narrow gate passes, run 2/3/4/5 Hz at 1/240, 1/480 and 1/1000 s, then
  a floating-base no-aerodynamics momentum/power-closure test before free-flight
  controller work or default promotion.

## Reconsideration triggers

Reconsider this decision if the 2 Hz aerodynamic gate remains unbounded, if
the required effort saturates, if tracking requires nonconvergent bandwidth,
if floating-base inverse-dynamics reduction fails momentum or power closure,
or when measured drivetrain parameters or explicit linkage geometry become
available.
