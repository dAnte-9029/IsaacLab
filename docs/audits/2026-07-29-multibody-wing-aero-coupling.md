# Measured multibody per-wing aerodynamic coupling

Date: 2026-07-29
Branch: `feat/physx-wing-multibody`
Input commit: `f3dae3c8`

## Objective

Connect DeLaurier aerodynamics to the measured-wing PhysX articulation without
changing the established commanded-kinematics/base-wrench baseline. The new
path must use actual joint motion, apply one wrench to each wing link, and
preserve the equivalent whole-aircraft force and moment.

This implements the remaining scope approved by
`ADR-2026-07-29-ideal-coupled-wing-drive.md`. It does not identify a physical
motor model.

## Explicit modes

The base configuration retains:

```text
wing_aero_coupling_mode = commanded_base_equivalent
```

The new
`FlappingBotStraightFlightMeasuredWingMultibodyIdealCoupledDeLaurierEnvCfg`
selects:

```text
plant_variant = measured_wing_multibody
wing_drive_variant = ideal_coupled_drive
use_delaurier_wings = true
wing_moment_mode = strip_integrated
wing_aero_coupling_mode = actual_per_wing_link
```

Invalid combinations fail during environment construction. Per-wing mode
currently excludes aggregate induced drag and the historical qd-scaled twist
proxy because neither has a defined per-wing physical distribution.

## Actual joint-motion mapping

The two URDF joint coordinates have opposite signs. After removing the
link-specific neutral offsets, the DeLaurier physical inputs are:

```text
q_physical = [q_left - q_left_mid, -(q_right - q_right_mid)]
qd_physical = [qd_left, -qd_right]
qdd_physical = [qdd_left, -qdd_right]
```

Position, velocity and acceleration come from the current PhysX articulation
state. The commanded phase remains the prescribed dynamic-twist phase; the
default dynamic-twist mode is disabled.

Runtime diagnostics expose actual aerodynamic inputs and their differences
from the command. No acceleration filter is hidden in this implementation.

## Wrench application

DeLaurier returns each wing's force and moment about the wing-root origin in
the wing link frame. Before application, the moment is translated to the
measured wing COM:

```text
M_about_wing_COM =
    M_about_wing_origin
    + (p_wing_origin - p_wing_COM) x F_wing
```

One Isaac Lab buffer call writes three local-frame wrenches:

```text
base_link  <- tail + fuselage drag
left_wing  <- left DeLaurier force and COM moment
right_wing <- right DeLaurier force and COM moment
```

The wing wrench is not also applied to `base_link`. PhysX joint constraints
therefore transmit aerodynamic load and inertial reaction to the body.

For diagnostics, each per-wing wrench is independently transformed and
translated back to the base COM. Its sum remains the reported net wing wrench.

## Validation

Pure and contract tests verify:

- opposed joint-coordinate mapping;
- wing-root to wing-COM moment translation;
- explicit baseline and multibody mode selection;
- existing DeLaurier force, moment, phase and dynamic-twist regressions.

The headless integration test disables tail aerodynamics, fuselage drag and
gravity. It verifies that:

- the base external-wrench buffer is zero;
- both wing buffers contain the DeLaurier wrenches;
- recomposing the two wing-COM wrenches about the base COM matches the
  equivalent net diagnostic within `2e-5`;
- 120 environment steps complete with finite kinematics and wrenches;
- no termination occurs;
- the hard coupling remains below the existing `0.1 deg` gate.

Observed 4 Hz smoke values:

| Metric | Result |
|---|---:|
| Maximum synchronization error | 0.029567 deg |
| Maximum single-wing force, including startup | 50.2475 N |
| Maximum single-wing COM moment, including startup | 7.05139 N m |
| Steady maximum single-wing force after two cycles | 15.2834 N |
| Maximum acceleration input error at first step | 4810.44 rad/s^2 |
| Steady maximum mean acceleration input error after two cycles | 41.9887 rad/s^2 |
| Terminations | 0 |

The first-step acceleration spike occurs because the ideal drive starts at
zero joint velocity while phase zero commands maximum upstroke velocity. It
is an initialization transient, not a steady wingbeat result. The default
environment freezes the root during its startup interval, but a future
periodic-state initialization may represent startup more physically.

## Boundaries

- The smoke establishes wiring, wrench equivalence and short-horizon numerical
  stability, not closed-loop flight quality.
- Raw PhysX joint acceleration has a startup spike and smaller steady
  finite-difference error. Filtering is intentionally deferred until a
  frequency and time-step sensitivity test establishes whether it is needed.
- `robot.data.applied_torque` still does not isolate motor-shaft, gearbox and
  mimic-constraint torque.
- Tail and rudder loads remain applied through the existing base-equivalent
  path.

## Commands

```bash
conda activate env_isaaclab
export PYTHONPATH="$PWD/source/flapping_bot${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q \
  tests/test_multibody_wing_coupling.py \
  tests/test_delaurier_strip_wrench.py \
  tests/test_delaurier_dynamic_twist.py \
  tests/test_ideal_coupled_drive_contract.py
TERM=xterm python -m pytest -q -s \
  tests/test_multibody_wing_aero_coupling_isaac.py
```
