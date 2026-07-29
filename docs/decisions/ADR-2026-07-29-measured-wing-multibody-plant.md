# ADR: Explicit measured-wing PhysX multibody plant

- Status: Accepted
- Date: 2026-07-29
- Scope: Runtime rigid-body mass partition and plant selection

## Context

The existing straight-flight plant suppresses appendage inertia, redistributes
removed appendage mass to `base_link`, and applies whole-aircraft COM and
inertia to the base. This baseline cannot reproduce inertial reactions from
the measured moving wings.

The accepted measured-property record defines `0.78261 kg` for the complete
non-wing body assembly and `0.06077 kg` for each wing. The body measurement
already includes tail, rudder, motor and gearbox hardware.

## Decision

Add an explicit `measured_wing_multibody` plant variant while retaining
`near_single_rigid_body` as the default.

The runtime mass partition is:

```text
base_link = 0.78231 kg
left_wing = 0.06077 kg
right_wing = 0.06077 kg
left_tail = 0.00010 kg
right_tail = 0.00010 kg
rudder = 0.00010 kg
total = 0.90415 kg
```

The three placeholder masses are subtracted from the measured non-wing body
mass so total mass is not duplicated. Their inertias are scaled from the
existing project asset by the same mass ratio. Their existing COMs are
retained.

The base and wing link-local COMs and inertias come from the accepted measured
property record. The body diagonal is projected to the nearest
triangle-consistent diagonal for runtime PhysX use:

```text
I_body_runtime = diag(
    0.003083333333333333,
    0.023056666666666667,
    0.019973333333333333,
) kg m^2
```

This runtime projection does not alter the frozen raw measurement record.

The variant bypasses the legacy appendage scaling, mass redistribution,
whole-aircraft base COM, whole-aircraft base inertia and total-mass scaling
hooks. The Stage-2 variant deliberately retains the existing kinematic wing
override; actuator replacement is a separate change.

## Alternatives considered

- Replace the default plant immediately: rejected because the established
  near-single-rigid-body baseline must remain reproducible.
- Assign zero mass to tail and rudder links: rejected because PhysX zero mass
  has special semantics and does not represent a normal massless moving link.
- Keep `0.78261 kg` on the base while adding placeholder masses: rejected
  because it would increase total mass above the measured `0.90415 kg`.
- Repartition measured tail and motor hardware independently: deferred because
  those component measurements are unavailable.

## Consequences

- Measured wing inertia becomes available to PhysX without changing the
  default plant.
- Tail and rudder structural inertia remains a numerical approximation.
- The measured variant is selectable independently of aerodynamic backend.
- The runtime base inertia is physical but differs slightly from the rounded
  metrology record.
- Kinematic wing override remains unsuitable for final multibody dynamics and
  must be replaced by the separately approved ideal coupled drive.

## Assumptions

- The body measurement includes all non-wing hardware.
- Tail and rudder placeholder masses are small enough not to alter the intended
  plant but positive enough for the current solver.
- Link names remain `base_link`, `left_wing`, `right_wing`, `left_tail`,
  `right_tail` and `rudder`.
- Existing tail/rudder COMs and inertia shapes are adequate numerical
  placeholders.

## Validation requirements

- Pure tests for mass partition, projection, dtype and input-shape handling.
- Exact `0.90415 kg` total mass.
- Explicit test that the default variant remains `near_single_rigid_body`.
- Headless PhysX write/readback for mass, COM and inertia on the real
  articulation.
- No mimic relation, actuator change, aerodynamic application-point change or
  controller tuning in this commit.

## Reconsideration triggers

Revisit this decision when tail/rudder component properties are measured, when
the non-wing body assembly is physically repartitioned, or when a stricter
inertia metrology release supersedes the current rounded diagonal.
