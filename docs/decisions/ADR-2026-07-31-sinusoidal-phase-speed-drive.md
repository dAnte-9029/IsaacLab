# ADR: Sinusoidal phase speed drive

- Status: Accepted
- Date: 2026-07-31
- Scope: Common wing mechanism phase and frequency dynamics

## Context

The measured-wing multibody plant needs a wing mechanism that preserves the
physical sinusoidal stroke while allowing the actual flap frequency to change
under load and track an instantaneous frequency command. The existing
`ideal_coupled_drive` treats wing angle as a finite-stiffness position tracking
problem. The experimental `prescribed_coupled_drive` imposes position through
moving zero-width limits, but PhysX does not report a consistent joint velocity
for that realization.

The physical vehicle has one motor and a rigid gear/linkage path inside the
body. Motor electrical parameters, shaft inertia, gear ratio, efficiency,
friction, backlash and compliance are not measured. A detailed motor model
would therefore introduce unsupported parameters without resolving the
mechanism-state requirement.

## Decision

Add a new explicit `sinusoidal_phase_speed_drive` model in stages. The first
stage is a pure PyTorch phase-drive core. PhysX integration and default
promotion are separate decisions.

The mechanism state is an unwrapped phase `phi`, phase rate `omega` and PI
integral state `xi`. The common physical wing coordinate is

```text
q = Gamma sin(phi)
q_dot = Gamma cos(phi) omega
q_ddot = Gamma [cos(phi) alpha - sin(phi) omega^2]
```

The opposed physical wing coordinates are

```text
q_left = q_left_mid + q
q_right = q_right_mid - q
```

This preserves the accepted convention: phase zero is the neutral pose starting
upstroke, and positive `q` raises both physical wings toward body-FLU `+z`.

Virtual throttle `u` is clamped to `[0,1]` and maps linearly to the instantaneous
frequency setpoint:

```text
f_setpoint = f_max u
omega_setpoint = 2 pi f_setpoint
```

The accepted initial values are `Gamma=30 degree`, `f_max=5 Hz`, zero frequency
at zero throttle and an approximate `0.15 s` two-percent speed settling time.

The speed regulator first computes a desired phase acceleration:

```text
e_omega = omega_setpoint - omega
alpha_reg = k_p e_omega + k_i xi
```

The known mechanism inertia and damping terms convert that desired acceleration
to generalized drive torque:

```text
Q_drive_unsat
  = J_phi alpha_reg
  + 0.5 (dJ_phi/dphi) omega^2
  + B_phi omega
Q_drive = clamp(Q_drive_unsat, -Q_limit, Q_limit)
```

Conditional integration prevents the PI state from winding farther into
saturation. The initial numerical effort limit is `1000 N m`. It is a solver
protection limit and must not be interpreted as real motor or gearbox torque.
The inertia and damping compensation contains only terms already defined by the
ideal mechanism. Aerodynamic load feedforward is not used; a signed aerodynamic
load should first cause frequency droop and the PI loop should recover it.

For damping ratio `zeta` and approximate two-percent settling time `t_s`,

```text
omega_n = 4 / (zeta t_s)
k_p = 2 zeta omega_n
k_i = omega_n^2
```

The measured opposed-wing inertia about the common stroke coordinate is
`I_eq`. Its phase-coordinate contribution is

```text
J_phi(phi) = J_floor + I_eq Gamma^2 cos^2(phi)
dJ_phi/dphi = -2 I_eq Gamma^2 sin(phi) cos(phi)
```

`J_floor` is an explicit numerical regularization because the rotating motor
and transmission inertia has not been measured. The initial floor is one
percent of the peak transformed wing inertia. It is not a measured physical
parameter and must receive a sensitivity check before PhysX promotion.

The signed common joint torque from aerodynamics, `Q_q_aero`, is projected into
phase space by virtual work:

```text
Q_phi_aero = Gamma cos(phi) Q_q_aero
Q_phi_aero omega = Q_q_aero q_dot
```

The phase acceleration is

```text
J_phi alpha
  + 0.5 (dJ_phi/dphi) omega^2
  = Q_drive + Q_phi_aero - B_phi omega
```

The pure reference step uses semi-implicit Euler and retains unwrapped phase:

```text
omega_next = omega + alpha dt
phi_next = phi + omega_next dt
```

No per-step wing joint-state write, moving zero-width joint limit or native
PhysX linear mimic can implement the nonlinear sine relation by itself.
PhysX-side enforcement remains a later implementation stage.

## Alternatives considered

- Continue finite-stiffness position tracking: rejected as the primary
  mechanism because load changes the sinusoidal shape as well as the frequency.
- Continue moving zero-width limits: rejected for promotion because the tested
  realization does not provide position-velocity-consistent joint state.
- Use `q=Gamma sin(2 pi f(t) t)`: rejected because changing `f(t)` makes phase
  discontinuous and adds the incorrect derivative term `t f_dot`.
- Model the DC motor and gearbox in detail: deferred because the required
  parameters are unavailable and are outside the multibody-plant objective.
- Build the complete linkage geometry in the URDF: physically attractive, but
  deferred until linkage dimensions and joint topology are frozen.

## Consequences

- The sinusoidal trajectory is a mechanism relation, while frequency is a
  dynamic state rather than a directly prescribed joint command.
- Left/right synchronization is exact in the abstract common coordinate.
- Aerodynamic loading can cause transient frequency droop without distorting
  the sine relation.
- Known phase-inertia compensation prevents the speed-loop bandwidth from
  changing by orders of magnitude at the stroke endpoints.
- The virtual drive torque is not motor-shaft torque and cannot support motor
  sizing or electrical-power claims.
- The phase-inertia floor is a numerical closure whose influence must be
  reported.
- Existing kinematic, ideal-torque and prescribed variants remain available.
- The default plant is unchanged.

## Assumptions

- The physical linkage is rigid enough that a single phase defines both wing
  angles over the 0--5 Hz operating range.
- The stroke amplitude remains `30 degree`.
- Reverse rotation is not explicitly prohibited by the pure phase equation;
  the speed regulator is expected to hold nonnegative operating points.
- Unmeasured transmission losses are initially represented only by configurable
  phase damping, whose default is zero.
- The one-percent phase-inertia floor is numerical stabilization, not
  identified hardware inertia.

## Validation requirements

- Pure float32 and float64 tests for throttle mapping, variable-frequency
  derivatives, opposed left/right kinematics and finite outputs.
- Exact virtual-work equality between common joint load and phase load.
- An energy-rate check for the configuration-dependent inertia term.
- PI gain, saturation and conditional anti-windup tests.
- Continuous kinematics across integer multiples of `2 pi` while the stored
  phase remains unwrapped.
- Before PhysX promotion, a 2/3/4/5 Hz matrix at `1/240` and `1/480 s` must
  check sine-constraint error, frequency error, left/right synchronization,
  drive saturation, whole-articulation momentum closure and aerodynamic
  force/moment convergence.
- The phase-inertia floor must be swept to show whether reported body response
  is sensitive to this numerical closure.

## Reconsideration triggers

Revisit this decision when motor or transmission inertia is measured, when
linkage geometry supports an explicit physical constraint model, when the real
mechanism exhibits meaningful compliance or backlash, or when the required
PhysX constraint cannot satisfy the joint-state, power and momentum checks.
