# ADR: Native holonomic load and transient diagnostics

- Status: Accepted
- Date: 2026-08-05
- Scope: Diagnostics for the canonical native measured-wing multibody plant
- Depends on: `ADR-2026-08-03-native-holonomic-wing-mechanism.md`

## Context

The native holonomic plant prescribes one opposed sinusoidal wing coordinate
inside the PhysX constraint solve. Validation now needs the corresponding
mechanism load, the sensitivity of acceleration-dependent DeLaurier terms to
the acceleration source, and continuous-frequency time-step evidence.

The custom PhysX row requests force output, but the installed
`IPhysxCustomJoint` callback does not return the generated `PxConstraint`
handle to Python. Consequently, the exact solver constraint multiplier is not
available through the current project interface. A diagnostic must not label a
reconstructed load as that multiplier.

## Decision

Add two opt-in diagnostics. Both remain disabled in the canonical configuration
and neither changes the applied aerodynamic wrench or native target.

The first diagnostic is explicitly named a multibody inverse-dynamics
estimate. Let the opposed joint direction be

\[
\mathbf{s}=[1,-1]^T,
\]

with positive common motion defined as left-wing upstroke and right-wing
opposed motion. For a floating base, eliminate the six unactuated root degrees
of freedom using

\[
\bar{\mathbf{M}}_{jj}
=\mathbf{M}_{jj}
-\mathbf{M}_{jb}\mathbf{M}_{bb}^{-1}\mathbf{M}_{bj},
\]

\[
\bar{\mathbf{h}}_j
=\mathbf{h}_j
-\mathbf{M}_{jb}\mathbf{M}_{bb}^{-1}\mathbf{h}_b,
\]

\[
\bar{\mathbf{Q}}_{ext,j}
=\mathbf{Q}_{ext,j}
-\mathbf{M}_{jb}\mathbf{M}_{bb}^{-1}\mathbf{Q}_{ext,b}.
\]

For a fixed base, the barred quantities equal their joint-space values. The
common-coordinate terms and estimated ideal constraint output are

\[
I_c=\mathbf{s}^T\bar{\mathbf{M}}_{jj}\mathbf{s},
\]

\[
Q_{bias,c}=\mathbf{s}^T\bar{\mathbf{h}}_j,
\]

\[
Q_{ext,c}=\mathbf{s}^T\bar{\mathbf{Q}}_{ext,j},
\]

\[
Q_{constraint,est}
=I_c\ddot q_{ref}+Q_{bias,c}-Q_{ext,c},
\]

\[
P_{constraint,est}=Q_{constraint,est}\dot q_{actual}.
\]

All outputs have batch shape `(N,)`. Inertia is in kg m2, torque terms are in
N m, and power is in W. Root generalized force and moment use the PhysX
world-frame convention; wing hinge loads are projected from link-local FLU
wrenches. The moment reference is the relevant wing hinge after translating
each wing COM wrench. The estimate describes the ideal mechanism output at the
common wing coordinate. It is not the exact PhysX multiplier, motor-shaft
torque, electrical power, or a drivetrain feasibility result.

The second diagnostic recomputes a shadow DeLaurier wrench with actual PhysX
joint acceleration while keeping the same actual joint position, joint
velocity, body state, air velocity and aerodynamic parameters as the applied
calculation. The shadow wrench is recorded but never applied. The canonical
plant continues to apply the prescribed analytical acceleration, including the
instantaneous frequency-rate term.

Continuous-frequency validation uses three profiles that start at 2 Hz: a
2--5 Hz one-second ramp, a 2--5 Hz 0.25-second ramp, and a smooth 2--5 Hz
one-second periodic profile. Fixed-root tests use 8 m/s wind, zero gravity and
no tail. Free-root tests start at 8 m/s in still air and include gravity plus
neutral tail aerodynamics. No flight controller is active. In the free-root
test, `stable` means finite and numerically bounded only; it does not mean
attitude or trajectory stability.

## Alternatives considered

- Report the reconstructed torque as the native solver force: rejected because
  the exact `PxConstraint` handle and multiplier are not exposed.
- Add a motor and gearbox model to obtain shaft quantities: deferred because
  motor, reduction, friction and drivetrain inertia are not identified.
- Apply the actual-acceleration shadow wrench: rejected because it would close
  the known step-delayed acceleration-force loop and change the accepted plant.
- Enable diagnostics unconditionally: rejected because inverse-dynamics tensor
  queries and a second strip calculation add runtime cost.

## Consequences

- The environment can record interpretable mechanism-output load and power
  components without changing default behavior.
- Applied and shadow aerodynamic calculations can be compared at identical
  state inputs.
- Diagnostics require the full wing-link wrench so the reconstructed external
  generalized load matches what PhysX receives.
- Exact solver reaction and motor feasibility remain outside the claim.

## Assumptions

- The common coordinate and hard mimic use left `+1`, right `-1`.
- Non-wing joint acceleration is negligible during the diagnostic projection;
  the transient tests keep the tail neutral.
- PhysX generalized mass, Coriolis/centrifugal and gravity compensation tensors
  follow the installed CPU tensor API ordering.
- The analytical rheonomic acceleration is the plant input; actual PhysX joint
  acceleration is a discrete diagnostic sampled after integration.

## Validation requirements

- Hand-calculated fixed- and floating-base reduction, sign, unit, dtype and
  power tests.
- An Isaac Sim aerodynamic integration gate proving finite nonzero load and
  shadow diagnostics while retaining the native tracking gate.
- Fixed- and free-root continuous-frequency jobs at 1/240, 1/480 and 1/1000 s.
- Record JSON summaries, NPZ traces, Git provenance, configuration, boundedness,
  tracking, synchronization, A/B wrench differences, torque and power.
- Preserve both diagnostic flags as default false.

## Reconsideration triggers

Reconsider the estimator if a supported exact constraint-force handle becomes
available, if tail or other joint motion is included in the mechanism load
claim, if the 1/480 s metrics cease to converge, or when measured drivetrain
parameters justify a motor-shaft model.
