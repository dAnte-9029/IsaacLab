# Sinusoidal phase-speed per-wing aerodynamic gate

Date: 2026-08-01
Branch: `feat/physx-wing-multibody`
Input commit: `1798a1abbbb3bf3297ef0a37e0790b25d0ef7093`

## Superseded aerodynamic evidence

The aerodynamic runs in this document used
`rigid_props.retain_accelerations=True`. In PhysX this retains applied forces
between physics steps, while the environment also submits a newly computed
wing wrench every step. A constant-force diagnostic later demonstrated the
resulting sequence `F, 2F, 3F, ...`. The reported aerodynamic load growth,
failure times and load-component ablations below are therefore invalid as
plant evidence. They are retained only as a historical record of the
diagnostic sequence. The no-aerodynamics mechanism observations are not
affected. Corrected native-holonomic results are recorded in
`2026-08-03-native-holonomic-validation.md`.

## Objective and claim boundary

Connect the accepted nonlinear phase mechanism to the existing measured-wing
per-link DeLaurier path, then run the narrowest fixed-body gate before launching
the full 2/3/4/5 Hz time-step matrix.

This is a subsystem integration and failure diagnosis. It is not free-flight
stability, controller evaluation, motor identification or aerodynamic-model
validation. No plant parameter, phase inertia, constraint bandwidth, speed-loop
gain, aerodynamic coefficient or controller gain was tuned.

## Implemented path

The new explicit coupling mode is:

```text
sinusoidal_phase_per_wing_link
```

It requires:

- the measured-wing multibody plant;
- the sinusoidal phase-speed drive;
- actual PhysX wing position, velocity and selected acceleration;
- strip-integrated DeLaurier wing loads;
- one local-frame wrench on each physical wing COM;
- no tail aerodynamics, induced drag or kinematic joint override.

The established default and all previous coupling modes remain unchanged.

## Periodic initialization

The first 2 Hz, 1/480 s run started the mechanism at zero frequency and then
commanded 2 Hz. It crossed the 200 N diagnostic force bound on step 2:

- joint acceleration: `339.33 rad/s^2`;
- maximum computed wing-link force component: `256.42 N`;
- failure time: `0.00625 s`.

This is a start-up experiment, not a periodic time-step comparison. Subsequent
fixed-frequency runs initialized the accepted phase state consistently at
`phi=0`, `omega=2*pi*f`, `q=0` and
`q_dot=Gamma*2*pi*f`. Each sinusoidal phase benchmark is restricted to one
frequency per Isaac process so this reset state is unambiguous.

## Two-hertz gate results

All cases used a true fixed root, 8 m/s airspeed, zero angle of attack, no
gravity, no tail aerodynamics and no fuselage drag. Four cycles were requested.

| Case | Step | Result |
| --- | ---: | --- |
| Full wrench, actual acceleration | 1/480 s | Failed at 1.6458 s, about 3.29 cycles |
| Full wrench, actual acceleration | 1/240 s | Failed at 0.0375 s |
| Full wrench, zero acceleration | 1/480 s | Bounded by guards, physically unacceptable frequency reversal |
| Full wrench, prescribed acceleration | 1/480 s | Bounded by guards, excessive frequency and trajectory error |
| Force only, actual acceleration | 1/480 s | Failed at 1.3167 s, about 2.63 cycles |
| Moment only, actual acceleration | 1/480 s | Completed four cycles with degraded tracking |

At the 1/480 s full-wrench failure:

- actual frequency was `0.3026 Hz`;
- common wing position magnitude was `0.4018 rad`;
- maximum computed wing-link force component was `6951.20 N`;
- maximum computed wing-COM moment component was `1072.01 N m`;
- actual wing acceleration input magnitude was `622.37 rad/s^2`.

The zero-acceleration case completed the requested duration but its actual
frequency ranged from `-4.758` to `11.084 Hz`, its mean was `1.481 Hz`, and the
maximum sine-constraint error was `3.978 deg`.

The prescribed-acceleration case ranged from `0.209` to `6.944 Hz`, its mean
was `1.921 Hz`, and its maximum constraint error was `3.480 deg`.

The moment-only case had mean frequency `2.002 Hz`, range
`1.143--3.721 Hz`, maximum constraint error `0.950 deg`, zero reported
left/right synchronization error and maximum left-joint effort `13.68 N m`.

## Interpretation

The hard left/right mimic is not the failure source: synchronization remained
zero or at numerical-noise scale in every completed case.

Actual acceleration makes the full loop unbounded, but it is not the sole
cause. Both zero and prescribed acceleration retain excessive phase-speed and
sine-constraint excursions. The force-only failure and bounded moment-only
case identify the DeLaurier resultant force transmitted through the wing
COM-to-hinge lever as the dominant structural load path.

The immediate loop is:

```text
actual wing motion
  -> per-wing DeLaurier force
  -> hinge reaction through the wing COM lever
  -> finite-bandwidth nonlinear phase constraint
  -> numerical phase inertia and speed PI response
  -> changed phase speed and physical wing motion
```

The `1/240 s` actual-acceleration case failed much earlier than `1/480 s`.
Therefore the current explicit load/constraint/phase cycle is not time-step
converged. Running the remaining frequencies would not rescue the failed
lowest-frequency gate and was intentionally stopped.

## Decision boundary

The per-wing load plumbing is implemented and its one-step Isaac wiring test
passes. The coupled aerodynamic plant is not eligible for default promotion.

The next change requires a new human-approved modeling choice:

1. increase or identify phase-side reflected inertia and speed-loop bandwidth;
2. use an ideal frequency source and report required mechanism torque without
   allowing wing load to perturb frequency;
3. reformulate the acceleration-dependent aerodynamic coupling implicitly; or
4. add measured motor/transmission parameters.

Changing the numerical phase inertia or regulator gains only to pass this gate
would turn an unmeasured closure into an apparent physical result and is not
authorized by the current ADR.

## Evidence and commands

Representative command:

```bash
conda activate env_isaaclab
PYTHONPATH="$PWD/source/flapping_bot${PYTHONPATH:+:$PYTHONPATH}" \
./isaaclab.sh -p scripts/flapping_px4/validate_multibody_aerodynamics.py \
  --headless --job fixed --frequency-hz 2 --dt-denominator 480 \
  --coupling-mode sinusoidal_phase_per_wing_link \
  --acceleration-source actual_joint_acceleration \
  --wing-link-load-mode full_wing_link_wrench \
  --single-point --smoke --output-dir <new-output-directory>
```

Generated outputs were not committed:

- `/tmp/sinusoidal_phase_aero_periodic_smoke_xbCc79/result/summary.json`;
- `/tmp/sinusoidal_phase_aero_zeroaccel_vfq1jU/result/summary.json`;
- `/tmp/sinusoidal_phase_aero_prescribedaccel_4D2okt/result/summary.json`;
- `/tmp/sinusoidal_phase_aero_forceonly_QtXH3X/result/summary.json`;
- `/tmp/sinusoidal_phase_aero_momentonly_v1Ell5/result/summary.json`; and
- `/tmp/sinusoidal_phase_aero_actual_dt240_r6PhoI/result/summary.json`.
