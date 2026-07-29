# ADR: Measured Link Mass Properties for the Multibody Plant

- Status: Accepted
- Date: 2026-07-29
- Scope: Body and wing mass, center of mass, inertia and frame conversion

## Context

The current flight environment suppresses appendage inertia and moves the
removed mass to `base_link`. That approximation cannot reproduce the wing
inertial coupling required by the PhysX multibody plant.

The measurement workbook `质量 重心位置 惯量.xlsx` contains the approved
body and right-wing values. Its SHA-256 is
`4673fdee7be187278bf6ae9957b78645c781330007a43e20a4f24f5dfd24831b`.
The workbook's `计算结果` sheet records:

- body mass and COM in `U_FRD`: cells `B41:B42`;
- right-wing mass and COM in the right-wing neutral-pose FRD frame:
  cells `B43:B44`;
- right-wing recommended inertia diagonal: cell `B46`;
- body recommended inertia diagonal: cell `B190`.

The body assembly measurement includes the non-wing vehicle hardware. The
right wing was measured separately and the left wing is treated as its
geometric mirror.

## Decision

Freeze the measured values without retuning:

```text
body:
  mass = 0.78261 kg
  COM_U_FRD = [-0.13103, 0.00625, -0.01500] m
  inertia_COM_U_FRD = diag(2.98e-3, 2.316e-2, 1.987e-2) kg m^2

right wing:
  mass = 0.06077 kg
  COM_W_R0_FRD = [-0.06040, 0.29394, 0.00000] m
  inertia_COM_W_R0_FRD = diag(2.70e-3, 1.01e-3, 3.71e-3) kg m^2
```

The right-wing `Izz` is the recommended bifilar-pendulum measurement.
`Ixx/Iyy` use the measured planform second-moment ratio normalized by that
`Izz`, because direct in-plane-axis oscillations were dominated by air drag.

Use the polar-vector frame transformation

```text
R_link_FLU_from_measurement_FRD = diag(1, -1, -1)
r_link = R r_measurement
I_link = R I_measurement R^T
```

Mirror right-wing link properties into the left-wing link with

```text
S_left_from_right = diag(1, -1, 1)
r_left = S r_right
I_left = S I_right S^T
```

Consequently, for a complete tensor, left/right mirroring changes the signs of
`Ixy` and `Iyz` and preserves `Ixz`. The measured release has no identified
off-diagonal products, so they are explicitly zero rather than inferred.

The link properties are about each link's own COM. No parallel-axis shift is
applied when writing link-local PhysX mass properties. The neutral joint
origins and rotations are only used later when assembling whole-aircraft
properties or running the articulation.

## Alternatives considered

- Put the whole-aircraft mass and inertia on `base_link`: retained as the
  existing near-single-rigid-body baseline, but unsuitable for wing inertia.
- Copy the right-wing COM into the left wing without reflection: rejected
  because the two link meshes extend along opposite local `y` directions.
- Invent off-diagonal inertia from CAD: rejected because it would mix a new
  unvalidated source into the measured parameter set.
- Adjust the body diagonal to enforce exact inertia triangle inequalities:
  rejected for this stage because it would silently alter the measured record.

## Consequences

- The three-link measured mass sum is `0.90415 kg`.
- The body and both wing tensors are symmetric and positive definite.
- The rounded body diagonal has
  `Ixx + Izz - Iyy = -3.10e-4 kg m^2`, a small physical-realizability
  discrepancy. Validation retains a named `5.0e-4 kg m^2` tolerance for this
  measured release; the stored values are not changed.
- Because the body measurement includes non-wing hardware, later plant
  integration must not also assign full measured masses to tail, motor or gear
  links unless the body measurement is repartitioned.
- This decision does not yet change any environment default or PhysX property.

## Assumptions

- `U_FRD` shares the `base_link` origin and differs from base FLU only by the
  FRD-to-FLU axis signs.
- `W_R0_FRD` shares the right-wing link/joint origin at neutral pose and differs
  from right-wing link FLU only by the FRD-to-FLU axis signs.
- The small opposite wing-joint roll angles in the URDF describe link placement
  in the base frame; they are not folded into link-local measured properties.
- The left wing has the same mass and principal magnitudes as the measured
  right wing.

## Validation requirements

- Exact provenance values and workbook hash.
- FRD-to-FLU vector and full inertia-tensor sign tests.
- Right-to-left COM and non-diagonal inertia mirror tests.
- Symmetry, finiteness and positive-definiteness checks.
- Explicit inertia triangle-margin check with the named body tolerance.
- Exact `0.90415 kg` body-plus-two-wings mass sum.
- No environment, actuator, aerodynamic or upstream asset modification in this
  stage.

## Reconsideration triggers

Revisit this decision when a complete six-component inertia measurement is
available, when body hardware is repartitioned into separate moving links, or
when metrology establishes a different link-frame origin or orientation.
