# PhysX ideal wing-coupling feasibility

Date: 2026-07-29
Branch: `feat/physx-wing-multibody`
Stage-1 input commit: `0353e2f1`

## Objective

Determine whether the installed Isaac Sim 5.1 / PhysX articulation solver can
maintain the mechanical relation

```text
q_left + q_right = 0
```

for measured-mass wings on a floating base. This is a coupling feasibility
test, not a complete plant or motor validation.

## Test construction

`tests/data/ideal_coupled_wings.urdf` contains:

- a floating base with the measured body mass and the nearest
  triangle-consistent diagonal inertia;
- two `0.06077 kg` wings with the measured wing inertias;
- a driven left revolute joint;
- a passive right revolute joint with
  `<mimic joint="left_wing" multiplier="-1" offset="0"/>`.

`tests/test_ideal_coupling_isaac.py` verifies the imported USD schema and runs
four-wingbeat headless CPU simulations. The drive is a numerical feasibility
drive (`Kp=2000 N m/rad`, `Kd=20 N m s/rad`, `100 N m` effort limit), not an
identified motor model.

The asymmetric-load case applies a zero-net hinge torque pair: a sinusoidal
torque on the right wing and an equal opposite torque on the base. This loads
the right joint without continuously injecting net angular momentum into the
whole articulation.

## Importer finding

The URDF importer creates `PhysxMimicJointAPI`, not an Isaac Lab fixed tendon.
The imported right joint has:

```text
reference joint = left_wing
gearing = +1
offset = 0
natural frequency = 25 rad/s
damping ratio = 0.005
```

The imported compliance is not sufficiently ideal for this plant. An initial
2 Hz run with those defaults produced a maximum synchronization error of
`0.101281 rad` (`5.8038 deg`). The accepted feasibility configuration
explicitly sets mimic natural frequency and damping ratio to zero before PhysX
initialization, selecting the hard-constraint path.

## Dynamic results

Acceptance threshold:

```text
max(abs(q_left + q_right)) < 0.1 deg
```

| Frequency | Step | Right hinge-load amplitude | Max sync error | RMS sync error | Max driver tracking error | Max driver effort |
|---:|---:|---:|---:|---:|---:|---:|
| 2 Hz | 1.0 ms | 0 N m | 0.000307 deg | 0.000206 deg | 0.011913 deg | 13.3493 N m |
| 5 Hz | 1.0 ms | 0 N m | 0.001916 deg | 0.001290 deg | 0.071482 deg | 35.7760 N m |
| 5 Hz | 1.0 ms | 0.25 N m | 0.012727 deg | 0.005825 deg | 1.5090 deg | 41.6759 N m |
| 5 Hz | 0.5 ms | 0.25 N m | 0.006028 deg | 0.004282 deg | 2.1730 deg | 55.1620 N m |

The loaded maximum synchronization error falls by approximately 53 percent
when the time step is halved. All hard-mimic cases pass the `0.1 deg` gate.

The unloaded floating-base response is also nonzero and increases from 2 Hz to
5 Hz:

| Frequency | Max base vertical speed | Max base pitch rate |
|---:|---:|---:|
| 2 Hz | 0.32870 m/s | 0.56236 rad/s |
| 5 Hz | 0.82287 m/s | 1.40564 rad/s |

This confirms that wing inertia is transmitted into the floating-base
dynamics in the minimal articulation. It is not yet a comparison with the
Amini reduced model.

## Decision

The hard `PhysxMimicJointAPI` path is technically suitable for implementing
the ideal left/right gear relation. The formal plant should:

1. preserve the URDF mimic relation during conversion;
2. explicitly set mimic natural frequency and damping ratio to zero before
   simulation initialization;
3. drive only the left/common coordinate;
4. leave the right mimic joint passive;
5. monitor `q_left + q_right` as a runtime diagnostic.

The default importer compliance must not be used unchanged.

## Limitations

- The numerical drive gains and `100 N m` limit are feasibility values, not
  measured motor or gearbox parameters.
- The loaded free-base pose response and drive tracking do not show sufficient
  time-step convergence to validate the complete plant.
- No aerodynamic wrench, gravity, tail link, controller or gearbox efficiency
  is included.
- Reported joint effort does not separately identify the internal mimic
  constraint impulse.
- The test establishes synchronization and inertial reaction only; it does not
  validate force amplitude against Amini et al.

## Reproduction

```bash
conda activate env_isaaclab
TERM=xterm ./isaaclab.sh -p -m pytest -q -s tests/test_ideal_coupling_isaac.py
```

Expected result: `2 passed`.
