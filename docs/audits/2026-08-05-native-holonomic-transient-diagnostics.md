# Native holonomic load and continuous-frequency audit

## Scope

This audit covers opt-in diagnostics for the accepted CPU native measured-wing
plant. It does not change the default applied wrench, add a controller, claim
real-flight validation, or identify motor-shaft torque.

Branch at execution was `feat/physx-wing-multibody`, HEAD
`eac56ee70a32709d923bfd03605d613c3dfc32f5`, with the diagnostic implementation
uncommitted. Authoritative run artifacts are the six `/tmp/native_transient_*_v3`
JSON/NPZ pairs. Earlier `v1` and `v2` artifacts are superseded because their
time-series buffers were not copied and their profiles did not all start from
the initialized 2 Hz state.

## Implemented quantities

`estimate_common_constraint_load` in
`source/flapping_bot/flapping_bot/physics/ideal_inverse_dynamics_phase_drive.py`
performs the fixed/floating-base reduction. The environment exposes its
inertia, bias, external-load, total-torque and mechanical-power terms only when
`native_holonomic_load_diagnostics=True`.

`straight_flight_env.py::_compute_wing_delaurier_wrench` accepts a diagnostic
acceleration-source override. With
`delaurier_shadow_actual_acceleration=True`, the normal prescribed-acceleration
wrench remains applied and the actual-acceleration result is cached separately.

The load is explicitly a multibody inverse-dynamics estimate at the ideal
common wing coordinate. It is not the exact PhysX constraint multiplier or a
motor-shaft quantity.

## Conditions

- CPU PhysX, 16 position and 4 velocity solver iterations.
- Three environments: slow 2--5 Hz ramp, fast 2--5 Hz ramp and smooth 2--5 Hz
  periodic target.
- Fixed root: two seconds, 8 m/s wind, no gravity, no tail.
- Free root: one second, 8 m/s initial speed, still air, gravity and neutral
  tail aerodynamics.
- Full per-wing DeLaurier wrench, prescribed acceleration applied, actual
  acceleration shadow only, no controller.

## Constraint tracking

Maximum over all three profiles:

| Root | Step | Max tracking error | Max synchronization error | Gate |
|---|---:|---:|---:|---|
| fixed | 1/240 s | 0.24972 deg | below 0.000004 deg | fail |
| fixed | 1/480 s | 0.06251 deg | below 0.000002 deg | pass |
| fixed | 1/1000 s | 0.01442 deg | below 0.000002 deg | pass |
| free | 1/240 s | 0.20773 deg | below 0.000004 deg | fail |
| free | 1/480 s | 0.05204 deg | below 0.000002 deg | pass |
| free | 1/1000 s | 0.01200 deg | below 0.000002 deg | pass |

The accepted 0.1 degree mechanism gate therefore continues to require 1/480 s
or smaller under continuous frequency changes.

## Applied versus actual-acceleration shadow wrench

The RMS norm difference as a percentage of the RMS applied wing-wrench norm
was:

| Root | Step | Force range | Moment range |
|---|---:|---:|---:|
| fixed | 1/240 s | 3.88--5.34% | 2.91--3.62% |
| fixed | 1/480 s | 1.92--2.64% | 1.44--1.81% |
| fixed | 1/1000 s | 0.92--1.26% | 0.69--0.87% |
| free | 1/240 s | 3.66--5.49% | 2.95--3.83% |
| free | 1/480 s | 1.83--2.74% | 1.49--1.93% |
| free | 1/1000 s | 0.88--1.31% | 0.72--0.93% |

The corresponding prescribed-versus-PhysX joint-acceleration RMS differences
also approximately halved with each halving of the time step. This is evidence
of a first-order sampling/alignment difference in the reported PhysX
acceleration, not evidence that the analytical prescribed acceleration should
be replaced in the applied DeLaurier path. Feeding the shadow result back would
reintroduce the step-delayed acceleration-force loop.

## Inverse-dynamics estimate and free-body boundedness

At 1/480 s, fixed-root RMS estimated torque was 8.15--11.38 N m and RMS
mechanical power was 81.53--129.06 W across the profiles. Relative to 1/1000 s,
the fixed-root RMS differences were at most 1.66% for torque and 2.45% for
power.

At 1/480 s, free-root RMS estimated torque was 6.07--9.36 N m and RMS mechanical
power was 55.15--102.88 W. Relative to 1/1000 s, differences were at most 1.01%
for torque and 1.06% for power. These are mechanism-output estimates, not motor
ratings.

All free-root states remained finite for one second. At 1/480 s, maximum root
linear speed was 11.23--12.19 m/s, maximum angular speed was 3.43--3.89 rad/s,
and displacement from the measurement start was 8.93--9.47 m. Relative to
1/1000 s, these root metrics differed by at most 1.32%. The motion is an
uncontrolled ballistic/aerodynamic response and is not a closed-loop stability
claim.

## Commands and tests

Pure tests:

```bash
conda activate env_isaaclab
PYTHONPATH=$(pwd)/source/flapping_bot ./isaaclab.sh -p -m pytest -q \
  tests/test_native_holonomic_transient_validation.py \
  tests/test_ideal_inverse_dynamics_phase_drive.py \
  tests/test_native_holonomic_drive_contract.py
```

Result: 16 passed.

The expanded targeted contract/regression set, including default-plant and
reset contracts, passed 37 tests. Ruff was not installed in `env_isaaclab`, so
the files were checked with `py_compile` and `git diff --check` instead.

The modified native aerodynamic integration gate ran in Isaac Sim and passed:

```text
max_tracking=0.062489401 deg
max_sync=0.000003415 deg
max_wing_force=15.457283 N
max_constraint_torque_estimate=16.055759 N m
max_shadow_force_delta=11.520251 N
```

The existing default-disabled one-second uncontrolled-flight smoke also
passed, with 0.03337 degree maximum tracking error and finite bounded root
state. This confirms that leaving both new flags false preserves the validated
native execution path.

Each matrix point used a fresh process, for example:

```bash
conda activate env_isaaclab
PYTHONPATH=$(pwd)/source/flapping_bot ./isaaclab.sh -p \
  scripts/flapping_px4/validate_native_holonomic_transients.py \
  --summary /tmp/native_transient_fixed_480_v3.json \
  --traces /tmp/native_transient_fixed_480_v3.npz \
  --dt-denominator 480 --device cpu --headless
```

Add `--free-root` for the released-base job. The matrix used denominators 240,
480 and 1000.

## Conclusion

The opt-in load estimate is numerically useful and explicitly bounded in
meaning. Continuous-frequency evidence supports 1/480 s as the current default:
it passes the trajectory gate and its main load/body metrics are close to the
1/1000 s reference. The 1/240 s step remains unsuitable for the authoritative
native plant. Actual PhysX joint acceleration remains a diagnostic rather than
the applied apparent-mass input.
