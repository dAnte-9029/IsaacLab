# ADR: Native holonomic wing mechanism constraint

- Status: Accepted
- Date: 2026-08-03
- Scope: Experimental measured-wing PhysX mechanism
- Supersedes for promotion: `ADR-2026-08-01-ideal-inverse-dynamics-phase-drive.md`

## Context

The measured-wing plant needs the inertia and constraint reactions of two
massive wing links, while the real gear and linkage mechanism prescribes an
opposed sinusoidal output. The effort-level phase drives leave a finite
tracking error because their torque is computed outside the PhysX solve and
only affects the next integration step. Under per-wing aerodynamic loading the
remaining error required large numerical effort and was step-size sensitive.

The matching PhysX 107.3 source provides `IPhysxCustomJoint v1.0`. It can
register a project-defined joint type whose solver-preparation callback emits
native `Px1DConstraint` rows. The development environment is version-locked to
Isaac Sim 5.1 and PhysX tag `107.3-omni-and-physx-5.6.1`.

## Decision

Add a project-local native Kit extension named
`omni.flapping_bot.holonomic_constraint`. It registers the concrete USD joint
type `FlappingWingTrajectoryJoint` and one angular equality row between the
base and left-wing link.

The constraint is

\[
C=q_L-q_{L,\mathrm{mid}}-\Gamma\sin\phi=0,
\]

with velocity target

\[
\dot q_{\mathrm{ref}}=\Gamma\cos\phi\,\dot\phi.
\]

The native solver row uses

\[
J=[-\hat a,\hat a],
\qquad
e=C,
\qquad
v_{\mathrm{target}}=\dot q_{\mathrm{ref}},
\]

where \(\hat a\) is the left hinge axis in world coordinates. The existing
PhysX mimic relation remains responsible for

\[
q_L-q_{L,\mathrm{mid}}=-(q_R-q_{R,\mathrm{mid}}).
\]

One trajectory constraint is authored per environment. Adding a second
trajectory row to the right wing would be redundant with the mimic relation.

The phase remains an externally integrated ideal mechanism coordinate:

\[
f_{\mathrm{target}}=f_{\max}\operatorname{clamp}(u,0,1),
\]

\[
\dot f=(f_{\mathrm{target}}-f)/\tau_f,
\qquad
\dot\phi=2\pi f.
\]

Per-environment position and velocity targets are transferred through one
batched native call before each physics step. Repeated USD edits are not used
for target updates. Wing actuators are passive; no per-step joint-state write,
moving joint limit, position drive or inverse-dynamics effort is applied.

## Alternatives considered

- Explicit linkage geometry: deferred because the complete linkage dimensions
  and joint topology are not available in the current asset.
- Effort-level inverse dynamics: retained as an experimental comparison, but
  not promoted because its aerodynamic gate has material tracking error and
  numerical effort.
- Moving zero-width limits: retained only as a comparison because imposed
  position and reported joint velocity were inconsistent.
- Detailed motor and gearbox dynamics: deferred because the required physical
  parameters have not been measured and are outside the current plant goal.
- Two independent wing trajectory constraints: rejected as redundant with the
  hard mimic and likely to make the constraint set ill-conditioned.

## Consequences

- PhysX solves the prescribed wing relation and the external aerodynamic load
  in the same solver step, so aerodynamic load changes reaction impulse rather
  than trajectory ownership.
- The mechanism is ideal and rheonomic. Constraint reaction is required ideal
  mechanism load, not identified motor-shaft torque or electrical power.
- The extension is binary and version-locked. Its build and ABI smoke tests are
  required before the plant variant can be selected.
- The Kit extension must be enabled at process startup with
  `--ext-folder <project extension parent> --enable
  omni.flapping_bot.holonomic_constraint`. Enabling it after Kit startup is
  rejected because the USD schema registry is already cached.
- The custom loop joints live below `/World/flapping_bot_constraints`, outside
  the live-cloned environment trees. Their body relationships use absolute USD
  paths into the corresponding environment.
- This variant sets `scene.replicate_physics=False`. PhysX fast replication
  does not expose replicated rigid bodies to an external custom joint during
  normal USD parsing. This makes the implementation correct for independent
  environments but reduces large-batch creation and training performance.
- The batched target path copies a small target vector from Torch to host
  memory. Its large-environment overhead remains unmeasured.
- The current custom row is CPU-only. PhysX rejects it in a direct-GPU scene as
  a non-GPU-compatible `PxConstraint`; the environment fails closed for CUDA
  simulation devices.
- The native variant uses 16 position and 4 velocity solver iterations. A
  previous 32/4 setting compensated for an invalid retained-force validation
  configuration; corrected per-step force semantics pass with 16/4. This does
  not change the settings of other plant variants.
- Aerodynamic validation must use \`retain_accelerations=False\` because a new
  wrench is computed and applied every physics step. Retaining forces while
  resubmitting them creates an unphysical cumulative load.
- Existing plant and drive variants remain available. The established default
  is unchanged during validation.

## Assumptions

- The left revolute-joint frame has local `+x` aligned with the positive
  engineering flap coordinate.
- The original revolute joint continues to constrain the other five relative
  degrees of freedom.
- The custom loop joint is excluded from the reduced-coordinate articulation
  tree and acts as a loop-closure constraint.
- The native extension is loaded before the USD stage and source environment
  are authored.
- Physics fast replication is disabled for this explicit plant variant.
- Phase target updates occur before the corresponding PhysX simulation step.

## Validation requirements

- Build the extension against the exact PhysX 107.3 development tree and load
  it in the installed Isaac Sim 5.1 runtime.
- Confirm that `FlappingWingTrajectoryJoint` is recognized as a
  `UsdPhysicsJoint` and that every environment path creates exactly one native
  constraint. Completed on CPU for one and two environments.
- For a fixed-root CPU environment, verify a sinusoidal target with maximum
  position error below 0.1 degree. Completed at 2 Hz and 5 Hz with
  (1/480\,\mathrm{s}): the two-environment maximum was (0.06291^\circ).
- Verify at least two environments receiving independent phase targets.
  Completed at 2 Hz and 5 Hz; mean recovered frequencies differed from the
  targets by less than (5\times10^{-7}\,\mathrm{Hz}).
- Repeat the trajectory gate with per-wing DeLaurier loads. Completed on CPU
  at 2/3/4/5 Hz, 8 m/s and (1/480\,\mathrm{s}) with 16/4 solver iterations.
  Maximum position errors were 0.01002/0.02252/0.04001/0.06249 degree and all
  synchronization errors were below (2\times10^{-6}\,^\circ). Earlier retained-
  force and wind-configuration results are superseded by the corrected audit.
- Repeat the mechanism gates with GPU dynamics. Completed as a negative
  capability gate: direct-GPU PhysX rejects the custom constraint. The variant
  now fails closed on CUDA and remains CPU-only.
- Verify left/right synchronization below 0.1 degree, bounded state, finite
  outputs, reset behavior and stage reload/shutdown. CPU synchronization,
  finite outputs, nonzero-frequency reset, a 0/2/5/3/0 Hz transition and
  fresh-process shutdown are complete. Same-process stage reload remains
  unresolved after a second environment construction hung during scene setup.
- Release the base with no external force and check whole-system momentum.
  Completed at 2 and 5 Hz for 1/480 and 1/1000 s. The maximum angular-momentum
  residual decreased from (2.35\times10^{-4}) to
  (1.68\times10^{-4}\,\mathrm{kg\,m^2/s}).
- Release the base under per-wing aerodynamics and check impulse-momentum
  closure. Completed for 2 and 5 Hz at 1/480 s. Relative linear errors were
  below (4\times10^{-6}) and relative angular errors were below 2.9 percent.
- Run a bounded uncontrolled-flight smoke with gravity and wing/tail
  aerodynamics. Completed for 1 s at 4 Hz and 8 m/s initial forward speed with
  neutral surfaces; all states remained finite and bounded. Fuselage drag
  remained at its zero default.
- Before default promotion, explicitly accept the CPU-only runtime, disabled
  physics replication and unresolved same-process stage reload.

## Reconsideration triggers

Reconsider this design if CPU-only execution is incompatible with the intended
workload, if same-process stage reload is required, if cloned targets
cross-couple, or if explicit linkage geometry becomes available.
