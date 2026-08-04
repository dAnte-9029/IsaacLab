# Ideal inverse-dynamics phase-drive gate

## Superseded aerodynamic evidence

The aerodynamic runs in this document inherited
`rigid_props.retain_accelerations=True`. Because a new per-wing wrench was
also applied every physics step, PhysX accumulated rather than replaced the
load. The reported aerodynamic tracking error, effort and power values are
therefore invalid as quantitative plant or drive evidence. The
no-aerodynamics tracking results remain usable, and the mechanism remains an
experimental comparison, but its aerodynamic failure conclusion has not been
rerun with corrected force-buffer semantics.

## Scope

This audit records the first implementation and narrow validation gate for the
experimental `ideal_inverse_dynamics_phase_drive` defined by
`ADR-2026-08-01-ideal-inverse-dynamics-phase-drive.md`. It does not promote the
variant, change the default plant, retune the flight controller or claim a
motor model.

## Implemented boundary

The variant keeps phase and frequency as ideal mechanism states, reads actual
PhysX joint position and velocity, and sends a single generalized effort to the
left wing while the passive right wing is coupled by the hard linear mimic.
The effort is computed from the PhysX generalized mass, Coriolis/centrifugal
and gravity tensors, with a common-coordinate projection and aerodynamic hinge
load feedforward. DeLaurier uses actual joint position and velocity and the
analytical mechanism acceleration.

No per-step joint-state write and no moving joint limit are used. All earlier
drive variants remain selectable and the repository default is unchanged.

## Validation sequence

The initial numerical tracking natural frequency was 50 Hz. Pure tensor and
contract tests passed before simulation. The fixed-root, no-aerodynamics gate
then passed at both requested frequencies:

| Frequency | Maximum position error | Synchronization error | Velocity RMS disagreement | Maximum drive effort |
|---:|---:|---:|---:|---:|
| 2 Hz | 0.052125 deg | 0 deg | 1.313% | 1.286869 N m |
| 5 Hz | 0.217724 deg | 0 deg | 3.283% | 7.994657 N m |

The narrow aerodynamic gate used a fixed root, 2 Hz, `1/480 s`, 8 m/s, zero
angle of attack, eight cycles, full per-wing link wrenches and prescribed
mechanism acceleration as the DeLaurier acceleration input. It remained finite
for all eight cycles, unlike the earlier phase-penalty implementation, but did
not satisfy the tracking requirement:

| Quantity | Result |
|---|---:|
| Maximum position error | 10.227747 deg |
| Maximum synchronization error | 0.000003415 deg |
| Maximum left drive effort | 163.343079 N m |
| Actual frequency range | 2.0--2.0 Hz |
| Actual position first-harmonic amplitude | 0.577026 rad |
| Reference position amplitude | 0.523599 rad |
| Actual position first-harmonic phase | -0.112521 rad |
| Aerodynamic hinge torque first-harmonic amplitude per wing | 0.552699 N m |
| Drive torque first-harmonic amplitude | 70.151559 N m |
| Mean drive power diagnostic | 115.708966 W |

The finite eight-cycle result is a meaningful improvement in boundedness, but
the 10.23 degree trajectory error and 163.34 N m numerical drive effort are not
acceptable for an ideal 30 degree sinusoidal mechanism.

## Single bandwidth ablation

A single 20 Hz tracking-bandwidth ablation was run to distinguish excessive
explicit feedback bandwidth from insufficient load rejection. It made the
result worse. The no-aerodynamics 5 Hz gate reached 0.545985 degree position
error and failed the 0.5 degree limit. At the same 2 Hz aerodynamic point the
maximum position error rose to 31.636432 degrees while the maximum effort fell
only to 117.579559 N m. The configuration was therefore restored to 50 Hz and
no further tuning sweep was performed.

## Interpretation

The ideal frequency state and the hard left/right relation are not the failed
parts: frequency remains exactly 2 Hz and synchronization remains within
microdegrees. The failure lies in realizing the nonlinear sinusoidal mechanism
with an explicit joint effort calculated from unconstrained articulation
inverse-dynamics tensors while PhysX separately resolves the hard mimic and
external wing-link loads through its constraint solver.

The aerodynamic hinge torque harmonic is less than 1 N m in the common
coordinate, yet the numerical drive reaches more than 160 N m. That disparity
shows that the large effort is dominated by correction of solver/model
mismatch and delayed tracking error rather than the physical aerodynamic hinge
load itself. Lowering the correction bandwidth reduces effort only modestly
while greatly increasing trajectory distortion, so bandwidth tuning cannot
turn this realization into a trustworthy ideal mechanism.

This experiment does not show that the measured multibody plant or per-wing
aerodynamic wrench is intrinsically unstable. It shows that the present
effort-level realization of the ideal nonlinear linkage is not sufficiently
constraint-consistent under aerodynamic loading.

## Gate decision

The implementation remains available as an explicit experimental comparison,
but it fails promotion. Do not run the 2/3/4/5 Hz by time-step matrix, use the
reported effort for motor sizing, retune the flight controller around this
plant or make it the default.

The next technically distinct option is a mechanism-level constraint rather
than another effort-gain sweep: explicit linkage geometry represented by
PhysX joints, or a custom nonlinear holonomic constraint whose position and
velocity relation is solved in the same constraint system as wing contact and
aerodynamic loads. That option requires a separate design decision before
implementation.

## Reproduction artifacts

- 50 Hz narrow-gate summary:
  `/tmp/ideal_inverse_aero_okPyie/result/summary.json`
- 20 Hz ablation summary:
  `/tmp/ideal_inverse_aero_20hz_w96Ykq/result/summary.json`

The paths are temporary local evidence and are intentionally not committed as
generated simulation output. The command and dirty-worktree provenance are
embedded in each summary.
