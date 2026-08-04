# Measured-Wing Multibody Aerodynamic Validation

Date: 2026-07-30
Branch: `feat/physx-wing-multibody`
Plant implementation commit under test: `1798a1abbbb3bf3297ef0a37e0790b25d0ef7093`

## Superseded aerodynamic evidence

All per-wing aerodynamic runs in this document were configured with
`rigid_props.retain_accelerations=True`. The environment recalculated and
submitted a new wrench each physics step, so the PhysX retained-force state
grew as the cumulative sum of all previous samples. A later 1 N diagnostic
measured per-step impulses of approximately `F*dt`, `2F*dt`, `3F*dt`, and so
on; setting `retain_accelerations=False` restored impulse-momentum closure.
Consequently, the failure times, force/moment ablations, power estimates and
time-step conclusions below are not valid evidence about the physical plant.
They are retained as historical implementation context. The corrected native
holonomic plant evidence is in
`docs/audits/2026-08-03-native-holonomic-validation.md`.

## Objective and claim boundary

This audit diagnoses the measured-wing `actual_per_wing_link` aerodynamic
loop before default promotion. It tests the wing-drive/aerodynamic subsystem,
not free-flight stability or flight-controller performance.

No acceleration filtering, drive tuning, motor calibration, flight-controller
tuning, tail aerodynamics, gravity, fuselage drag, or default promotion was
performed.

The final fixed-body matrix uses a true PhysX fixed articulation root. An
earlier diagnostic restored the root state every physics step; its apparent
constraint-driven load growth is superseded because repeated root writes can
inject articulation constraint impulses.

## Implemented diagnostics

The following explicit ablations preserve actual PhysX wing position and
velocity:

- `zero_acceleration`: use zero only for the acceleration-dependent DeLaurier
  heave input;
- `prescribed_acceleration`: use analytical commanded acceleration for that
  input; and
- `actual_joint_acceleration`: preserve the implemented PhysX `joint_acc`
  input.

Two additional isolation paths were used:

- `actual_motion_base_equivalent`: compute aerodynamics from actual wing
  motion but apply the equivalent wrench to the fixed base so the load cannot
  feed back through the wing joints; and
- per-wing `force_only` and `moment_only` load ablations.

The default configuration remains `actual_joint_acceleration` with the full
per-wing wrench. All diagnostic variants are explicit and non-default.

## Execution contract

One Isaac process is used per environment because recreating multiple
`DirectRLEnv` instances in one `SimulationApp` was not reliable. A combined
11-test Isaac process similarly produced two order-dependent synchronization
failures; both tests passed in fresh processes.

The standard condition is 8 m/s, zero angle of attack, 2/3/4/5 Hz, and physics
steps of 1/240 and 1/480 s. The final matrix used eight cycles and analyzed the
last four when the case remained bounded. Diagnostic guards were 0.8 rad wing
position, 50 rad/s wing velocity, 200 N per-wing force component, 50 N m
per-wing COM moment component, and finite state checks.

Generated outputs were not committed:

- final three-source matrix:
  `/tmp/physx_aero_true_fixed_long_matrix_20260730`;
- aggregate:
  `/tmp/physx_aero_true_fixed_long_matrix_aggregate_20260730`;
- true-fixed commanded baseline:
  `/tmp/physx_aero_true_fixed_baseline_20260730`;
- force/moment and drive comparisons:
  `/tmp/physx_aero_driver_compare_20260730`.

Representative command:

```bash
./isaaclab.sh -p scripts/flapping_px4/validate_multibody_aerodynamics.py \
  --headless --job fixed --frequency-hz 4 --dt-denominator 480 \
  --coupling-mode actual_per_wing_link \
  --acceleration-source actual_joint_acceleration \
  --wing-link-load-mode full_wing_link_wrench \
  --single-point --total-cycles 8 --measurement-cycles 4 \
  --output-dir /tmp/actual_f4_dt480
```

## Results

### True-fixed eight-cycle acceleration-source matrix

Only four of 24 actual-per-wing cases completed eight cycles. Those four were
the zero-acceleration, 1/240 s cases, and they still had 8.28--10.70 degrees
maximum tracking error.

Failure time expressed as flapping cycles was nearly frequency invariant:

| Acceleration source | 1/240 s | 1/480 s |
| --- | ---: | ---: |
| zero | completed 8 cycles, degraded tracking | about 5.0 cycles |
| prescribed | about 4.5 cycles | about 2.5 cycles |
| actual `joint_acc` | about 6.0 cycles | about 3.0 cycles |

This strong dependence on integration step and weak dependence on physical
frequency identifies a discrete wing-load/drive cycle-map instability. The
1/240 s cases are artificially stabilized by numerical damping and cannot be
treated as converged.

The four-cycle zero-acceleration matrix had both steps bounded, but it also
failed convergence:

- wing-position phase differed by 1.72--4.37 degrees;
- chordwise-force fundamental amplitude differed by 84.7--90.7 percent;
- vertical-force fundamental amplitude differed by 8.08--29.1 percent; and
- pitch-moment phase differed by 48.4--69.0 degrees.

### Actual-motion shadow and load-component ablations

At 4 Hz and 1/480 s, all three acceleration sources completed four cycles when
the same computed wrench was applied to the fixed base rather than the wing
links. Therefore actual kinematics, the DeLaurier calculation, and the ideal
drive remain bounded when the aerodynamic load cannot act on the wing joints.

With actual acceleration and wing-link loading:

- force only completed four cycles but reached 9.90 degrees tracking error;
- COM moment only completed with 2.05 degrees tracking error; and
- the full wrench crossed the force guard at 0.754 s.

The force transmitted through the wing COM-to-hinge lever is the dominant
source of tracking degradation. The intrinsic COM moment alone is not the
dominant instability source. The full wrench grows earlier because
force-induced trajectory error changes the subsequent computed force and
moment.

### Work and drive diagnostics

During the first bounded cycles the computed aerodynamic hinge power was
negative, so the initial load removed mechanical energy rather than acting as
a simple sign-reversed negative damper. As the full-wrench case departed from
the prescribed trajectory, instantaneous computed hinge power became positive
and reached approximately 748--874 W near the prescribed- and
actual-acceleration failure points. The disturbed trajectory therefore enters
a flutter-like energy-injection region.

The current `ideal_coupled_drive` is not a hard prescribed mechanism. It is a
high-stiffness implicit PD drive:

- even with actual-motion loads applied to the fixed base, the left driver
  torque fundamental was about 55.8 N m and mean computed driver power about
  348 W at 4 Hz;
- force-only wing loading raised mean driver power to about 668 W; and
- all compared cases reached approximately 311 N m peak driver torque during
  startup.

These values show strong dynamic interaction between the ideal PD drive and
wing load. They must not be interpreted as a validated real motor or gearbox.

### True-fixed commanded baseline

All eight `commanded_base_equivalent` cases completed eight cycles. Prescribed
kinematics were step-converged, and vertical-force fundamental amplitude
differed by at most about 0.16 percent with at most 0.21 degrees phase error.
Pitch-moment phase still differed by approximately 3.1--13.9 degrees.
Chordwise-force fundamental metrics remained poorly conditioned and strongly
step-sensitive. This bounded baseline confirms that instability requires
wing-joint load feedback, while also confirming that 1/240 s is inadequate
for higher-harmonic moment claims.

### Free-body closure diagnostic

The earlier 0.1 s free-body cases remained finite but had order-one external
wrench/momentum residuals. That result remains a failed conservation
diagnostic. It does not assess flight stability and does not become valid
because the fixed-body fixture was corrected.

## Root-cause conclusion

The evidence rejects two simple explanations:

- missing flight-controller tuning cannot explain the result because the final
  experiment has a true fixed root; and
- a globally reversed aerodynamic torque sign is unlikely because initial
  hinge work is dissipative and the moment-only path remains bounded.

The immediate cause is the dynamic loop

```text
actual wing motion
  -> DeLaurier wing force
  -> force lever about the wing hinge
  -> high-stiffness implicit PD drive and mimic response
  -> changed wing phase/velocity
  -> changed DeLaurier wing force
```

Acceleration-dependent aerodynamic terms materially change this loop's phase
and growth rate, but they are not the sole cause: zero acceleration also fails
at the fine step after about five cycles. The near-constant failure cycle count
and severe step dependence show that the present ideal-drive/per-link-load
integration is not numerically converged.

## Decision and next task

The measured mass distribution, hard opposed mimic relation, and per-wing
wrench mapping remain useful selectable components. The complete aerodynamic
plant must not become the default yet.

The next design decision is no longer acceleration filtering. It is to replace
or reformulate the wing-motion constraint/load coupling. Candidate directions
requiring approval are:

1. a genuinely prescribed ideal mechanism for plant validation, with reaction
   loads measured but no finite-compliance PD tracking dynamics;
2. a physically parameterized motor/gearbox drive with realistic reflected
   inertia, torque/speed limits, damping, and efficiency; or
3. a coupled implicit treatment of aerodynamic added mass together with a
   separately validated drive.

Whichever path is selected must first pass the same true-fixed eight-cycle
matrix and a whole-articulation conservation check before free-flight
controller work.

## Prescribed-mechanism follow-up

The accepted prescribed-mechanism direction was implemented after the
diagnostics above. Both measured wings are passive articulation links whose
opposed positions are imposed by time-varying zero-width PhysX joint limits
generated from one analytical sine coordinate. The DeLaurier calculation uses
that analytical position, velocity and acceleration, while its two physical
wing wrenches are still applied to the corresponding wing links.

The true-fixed 8 m/s, zero-angle-of-attack matrix completed at least eight
cycles at 2/3/4/5 Hz for both 1/240 and 1/480 s physics steps. All eight cases
remained finite and no diagnostic guard fired. Across the complete matrix:

- maximum opposed synchronization error was `0.000161 deg`;
- vertical-force fundamental amplitude differed by at most `0.302 percent`;
  and
- vertical-force fundamental phase differed by at most `0.203 deg`.

The first version of this matrix labeled the analytical aerodynamic position
and velocity inputs as actual joint motion. A corrected rerun instead reduced
the opposed PhysX `joint_pos` and `joint_vel` states to their common physical
coordinate. It found:

- physical position amplitude agreed between steps within `0.000181 percent`;
- physical position phase differed by `1.496--3.745 deg`;
- maximum instantaneous position error was `1.538--3.835 deg` at 1/240 s and
  `0.769--1.922 deg` at 1/480 s;
- the physical-position lag relative to the analytical reference was exactly
  one physics step, `360 f dt`; and
- reported PhysX common `joint_vel` amplitude was only
  `0.000558--0.001329 rad/s`, versus `6.444--16.111 rad/s` implied by the
  position trajectory.

The one-step position offset follows the benchmark ordering: the analytical
phase and moving limit advance before the physics step, while the benchmark
samples the previous PhysX state. More importantly, the near-zero
`joint_vel` shows that moving zero-width limit projection does not populate a
joint velocity consistent with the imposed position sequence.

The matrix therefore shows bounded loads and excellent opposed
synchronization, but it does not establish a kinematically consistent
prescribed multibody state. The nonzero floating-base reaction can include
position-projection impulses and cannot by itself validate physical wing
inertia. The unbounded finite-PD cycle map is absent, but default promotion is
still blocked.

The pitch-moment fundamental is less converged. Relative to 1/480 s, the
1/240 s result differs by 1.75--5.56 percent in amplitude and by
3.19--14.45 degrees in phase over 2--5 Hz. The chordwise-force fundamental is
small relative to its mean and residual content, so its 45--93 percent
relative step differences are poorly conditioned and are not a useful
stability metric.

A separate no-aerodynamics, no-gravity, floating-base 2 Hz gate confirmed that
the moving constraints transmit wing inertia into the articulation:
maximum base vertical speed was `0.073174 m/s`, maximum base pitch rate was
`0.114704 rad/s`, and maximum reported wing acceleration was
`42.5094 rad/s^2`. This establishes nonzero inertial reaction transmission;
it is not yet a whole-articulation momentum-conservation closure result.

The prescribed mechanism therefore passes only the boundedness and opposed-
synchronization gates. It does not pass the joint-state kinematic-consistency
gate or a strict all-load time-step convergence gate. A 1/960 s moment sweep
would not resolve the more fundamental near-zero `joint_vel` state and should
not be the next experiment.

Generated follow-up outputs were not committed:

- `/tmp/prescribed_plant_stability_matrix_dt240_20260730/summary.json`;
- `/tmp/prescribed_plant_stability_matrix_dt480_20260730/summary.json`; and
- `/tmp/prescribed_plant_stability_f5_dt480_20260730/summary.json`.

Corrected physical-joint-state outputs:

- `/tmp/prescribed_plant_physx_state_matrix_dt240_20260730/summary.json`; and
- `/tmp/prescribed_plant_physx_state_matrix_dt480_20260730/summary.json`.

## Automated checks and operational limitation

The final code validation must be run in isolated Isaac processes. Ruff is not
installed in `env_isaaclab`; the completed checks were:

- 55 targeted pure tests passed;
- the existing per-wing aerodynamic coupling Isaac test passed in a fresh
  process;
- the new aerodynamic worker Isaac test passed in a second fresh process;
- `py_compile`, 120-column line-length checks, and `git diff --check` passed.

Isaac's editable upstream asset converter resolves the robot through the main
checkout and regenerated that checkout's already-dirty upstream `.asset_hash`
and `config.yaml` metadata. Those files were neither staged nor reverted.
