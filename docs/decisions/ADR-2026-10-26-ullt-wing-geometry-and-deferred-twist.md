# ADR: ULLT wing geometry and deferred twist

- Status: Geometry partially superseded on 2026-10-27; deferred twist and baseline preservation remain accepted
- Date: 2026-10-26
- Branch: `feat/ullt-state-space-wing-aero`, base `a7b3e08e0c1e`
- Scope: Opt-in Izraelevitz 2017 wing-aerodynamics backend, independent of formal RL training

> **Geometry superseded:** Read
> [ADR-2026-10-27-ullt-urdf-root-local-csv-geometry.md](ADR-2026-10-27-ullt-urdf-root-local-csv-geometry.md)
> before using the geometry statements below. The user confirms URDF roots at
> +/-56 mm and the complete CSV measured outward from each root. The earlier
> 0.650 m inference and claim of an overlong DeLaurier wing are withdrawn.
> The following geometry discussion is retained as historical reasoning, not
> a requirement to shorten the CSV or modify the existing plant.

## Context and evidence

The user describes a carbon-tube frame covered with kite fabric. The thickest
tube is the leading edge and connects to the flapping mechanism. Each flapping
axis is 50 mm from the vehicle centerline. The tip-to-tip span is 1.4 m. The
wing may also pitch about its leading-edge tube. No airfoil section definition
or measured aerodynamic force data are available.

The user subsequently confirms that the central 100 mm between the flapping
axes is an empty gap and must not contribute aerodynamic wing area or loads.

The user explicitly defers twist for the initial backend and requires that the
existing aerodynamic model and its default configuration remain available and
unchanged.

Repository evidence at the base commit:

- `source/flapping_bot/scripts/data/right_wing_te_polyline_mm.json` contains
  trailing-edge points from 0 to 650 mm; the fitting script's `te_polyline`
  schema assumes a straight leading edge.
- `outputs_DeLaurier/right_wing_te_fit_poly5_gap50.csv` extends to an inferred
  single-wing length of 0.700 m and contains `dhat=0.25`.
- `FlappingBotStraightFlightEnv.__init__` in `straight_flight_env.py` loads
  that CSV with `dhat=0.0`;
  `physics/wing_geom_csv.py::build_wing_geometry_from_csv` retains its inferred
  0.700 m length. The CSV loader's area estimate and the interpolated 80-strip
  area differ slightly; neither is a new measurement of the physical wing.
- `source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf`
  places the wing joints at body `y=+/-0.056 m`, with opposite small neutral
  roll offsets. These existing asset values differ from the user's 50 mm
  physical description and are not changed by this decision.
- `physics/delaurier_twist.py` supplies prescribed twist kinematics;
  `physics/virtual_twist.py::VirtualTwistCfg` labels stiffness, damping and
  effective torsional inertia as needing identification.

## Decision

1. Preserve existing DeLaurier defaults, robot assets, controller parameters,
   and RL task contracts. Any new backend must be explicitly selected.
2. Record the user-provided neutral/reference geometry as a 1.4 m tip-to-tip
   span and flapping-axis offsets of +/-0.050 m. Under the existing symmetric
   reference-planform interpretation, the derived axis-to-tip span is
   `1.4 / 2 - 0.050 = 0.650 m`. The 0.700 m centerline-to-tip distance must not
   also be used as the outboard span measured from the hinge.
   The central gap has no lifting elements or applied wing loads. This does not
   remove aerodynamic interaction between the two outboard wings.
3. Distinguish the flapping hinge axis from the spanwise leading-edge pitching
   axis. A future leading-edge pitch degree of freedom has a leading-edge
   offset of zero; it is not a quarter-chord mechanical hinge.
4. Defer both a separate leading-edge pitch degree of freedom and distributed
   dynamic twist in the first backend. Continue to model flapping and vehicle
   rigid-body motion. Do not infer torsional rigidity or an absence of membrane
   deformation from this temporary modeling choice.
5. Record the absence of section/polar and aerodynamic force measurements.
   Literature reproduction and numerical consistency can be validated, but
   real-wing force accuracy cannot yet be claimed.

No code or physical configuration is changed by this ADR. Mapping the new
aerodynamic reference geometry onto the existing articulation, including its
56 mm origins, remains an explicit integration design item. Do not silently
compensate by shifting loads or editing upstream assets. The lifting treatment
of the central 100 mm region is now fixed as an empty, non-lifting gap.

### Existing DeLaurier geometry behavior

The current environment creates 80 uniform strips over local `[0, 0.700] m`
and interpolates the CSV chord at those strip centers. The first approximately
50 mm of the CSV has a nonzero chord near 0.300 m; no root-gap mask is applied.
The local span coordinate is used directly in `h=-q*y`, `hdot=-qd*y` and
`hddot=-qdd*y`, without subtracting 50 mm.

`physics/qsm_delaurier1993.py::integrate_delaurier_strip_wrench` uses that same
span coordinate for force moment arms about the wing origin. Circulatory
normal force acts at quarter chord and apparent-mass normal force at half
chord, plus the separately represented free pitching couples. With `d_hat=0`,
these points are behind the assumed leading-edge pitch axis.

`FlappingBotStraightFlightEnv._compute_wing_delaurier_wrench` subsequently
transforms each wing wrench using its actual link pose and translates it to
the wing COM or base COM. Thus the URDF hinge offset is additional to the
0.700 m local wing extent; it is not deducted from the CSV coordinates.
Ignoring the small neutral roll, the current nominal aerodynamic tip span
is approximately `2 * (0.056 + 0.700) = 1.512 m`. This is a code/asset geometry
calculation, not a new runtime measurement. It does not mean that the existing
code puts strips directly in the central gap: it places an overlong local
wing outboard of each already-offset hinge.

These discrepancies affect local motion, area, aspect-ratio input and moment
arms, not just graphical placement. They are recorded for separate geometry
validation; this ADR does not alter the old backend or quantify resulting
force/trajectory error.

## Paper interpretation and pending closure choices

Reference: Izraelevitz, Zhu and Triantafyllou (2017), DOI
`10.2514/1.J055144`, local accepted manuscript under `docs/papers/`.

- Section III uses an elliptic planform with aspect ratio 3, sheared to a
  straight quarter-chord line. Section IV uses an elliptic flapping wing with
  total aspect ratio 6. Section V compares against rectangular-wing experiments
  with aspect ratio 3. The method is not restricted to rectangular planforms.
- Section II.A places the lifting-line control points and horseshoes at
  quarter chord and evaluates sectional motion at three-quarter chord.
  These aerodynamic locations do not define the mechanical pitching axis.
  Measure both positions from each local leading edge using local chord and
  chord direction. In particular, quarter chord does not mean the instantaneous
  resultant center of pressure of all unsteady loads.
- A straight leading edge with variable chord generally has a non-straight
  quarter-chord locus. The general geometric influence calculation supports
  sweep, but the additional downwash correction of Section II.D/Appendix B is
  derived only for unswept wings. Its applicability must be checked separately;
  do not silently transfer the straight-quarter-chord benchmark geometry.
- A thin, flat, attached-flow sectional prior is a possible initial closure,
  not an identified property of the carbon-tube/fabric wing. Lift slope,
  zero-lift angle, profile drag and tube/membrane effects need explicit
  provenance and sensitivity treatment when selecting that closure.
- Added-mass force/moment requires consistent sectional acceleration and
  reference-point handling; future free-body coupling must address the
  acceleration/load feedback rather than treating a delayed acceleration as
  an instantaneous solved state.

## Existing operating conditions

Reuse existing configuration and validation cases as the starting context;
the user need not resupply them:

| Existing source | Conditions | Meaning |
| --- | --- | --- |
| `FlappingBotStraightFlightDeLaurierEnvCfg` | 2--5 Hz, reset 4 Hz | Canonical task frequency limits |
| `PURE_RL_SHARED_CONTRACT` | 0--5 Hz, 2 Hz/s frequency slew, 480 Hz physics | Existing PureRL contract; no new training authorized |
| `straight_flight_env.py` wing-amplitude construction | Approximately +/-30 degrees, bounded by joint limits | Prescribed mechanical stroke |
| `validate_native_holonomic_mechanism.py` | Default 8 m/s, 2/3/4/5 Hz | Existing mechanism/aero-load numerical cases |
| `FlappingBotStraightFlightEnvCfg` | Reset forward speed 8 m/s; command 7 m/s; optional command range 2--8 m/s | Initial conditions and commands, not a validated airspeed envelope |

PureRL preview-speed bounds of 1--12 m/s are algorithmic bounds, not a physical
validation range. Likewise the DeLaurier 0.5 m/s speed clamp is a numerical
guard. Local incidence, sectional speed and reduced frequency must be reported
when assessing ULLT validity. The existing real-flight domain profile is
marked `candidate_not_promoted`, not an approved aerodynamic envelope.

## Alternatives considered

- Change the existing wing span or robot joint origins immediately: rejected
  because that would change the preserved baseline and mix geometry correction
  with backend implementation.
- Copy the CSV quarter-chord offset as the mechanical pitch axis: rejected
  because it conflicts with the user's leading-edge description.
- Enable the historical virtual-twist defaults: deferred because the user
  excludes twist initially and those structural parameters are unidentified.
- Treat all benchmarks as rectangular wings: rejected by Sections III--V.

## Consequences and assumptions

The initial backend can proceed without torsion experiments or force data for
paper reproduction. Real-wing predictions will remain analytical estimates.
The 0.650 m outboard-span derivation assumes the stated 1.4 m is the symmetric
reference/extended span, not the instantaneous projected span during flapping.
The central gap has been explicitly confirmed as empty. Aerodynamic-to-link
registration remains separate from that scalar span calculation.

## Validation requirements

- Check centerline, hinge, tip and leading-edge positions in one declared frame.
- Verify that the central gap contributes no lifting elements, area or loads,
  while retaining mutual induced-flow coupling between the wings.
- Test local quarter-/three-quarter-chord construction and left/right symmetry.
- Verify deferred twist contributes no extra pitch motion or hidden proxy.
- Reproduce paper geometries separately from the actual-wing geometry.
- Test force/moment reference translation, wake-state reset and time/space
  convergence; retain old-backend regression checks.
- Keep numerical validation distinct from physical force validation.

## Reconsideration triggers

Revisit twist when measured pitch/twist histories or structural identification
become available, or when the research scope explicitly includes flexibility.
Revisit sectional closures with membrane/camber observations or force data.
Revisit geometry registration before applying new-backend loads to the existing
articulation. Do not expand the geometry correction to default assets implicitly.
