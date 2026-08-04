# PhysX custom holonomic constraint feasibility audit

## Scope

This read-only audit evaluates whether the failed effort-level sinusoidal wing
drive can be replaced by a nonlinear holonomic constraint solved inside PhysX.
It covers the installed Isaac Sim 5.1 / PhysX runtime, the current measured-wing
plant, GPU and cloned-environment implications, and the smallest technically
sound proof-of-concept boundary. It does not implement a native extension,
change the plant, tune a controller, or approve a new architecture.

The intended mechanism relation is

\[
C(q_L,\phi)=q_L-q_{L,\mathrm{mid}}-\Gamma\sin\phi=0,
\]

with

\[
q_R-q_{R,\mathrm{mid}}=-(q_L-q_{L,\mathrm{mid}}),
\qquad
\dot\phi=2\pi f.
\]

The key distinction from the rejected effort-level realization is that the
constraint solver would compute the reaction multiplier \(\lambda\) together
with the articulation state. The wing trajectory would no longer be approached
through an explicit penalty torque whose error is corrected one step later.

## Observed repository facts

The current source environment applies `PhysxMimicJointAPI` to the right-wing
joint before cloning. Its configured relation is PhysX's
`gearing * q_reference + q_mimic + offset = 0`, with gearing 1 and offset 0;
see `apply_hard_opposed_wing_mimic()` in
`source/flapping_bot/flapping_bot/assets/ideal_coupled_drive.py`.

`StraightFlightEnv._setup_scene()` applies this schema to `env_0` and then calls
`clone_environments(copy_from_source=False)`. `InteractiveSceneCfg` defaults to
`replicate_physics=True`, and the flight configuration defaults to 256
environments. Therefore a new constraint must participate correctly in USD
cloning and PhysX replication; creating an untracked native object for only the
source environment is insufficient.

The measured-wing variants require `plant_variant='measured_wing_multibody'`.
For per-wing aerodynamic coupling, `StraightFlightEnv` submits one local-frame
wrench for the base and one for each wing link in a single
`set_external_force_and_torque()` call. This is compatible in principle with a
solver-level wing constraint: external wing loads remain external, while the
constraint reaction is transmitted through the articulation.

Only one new trajectory constraint should be added. With two wing hinge
coordinates, the existing mimic removes one relative degree of freedom and the
new constraint removes the remaining common degree of freedom. Adding a
trajectory constraint independently to both wings would duplicate the mimic
relation and create a redundant, potentially ill-conditioned constraint set.

## Observed installed-runtime facts

The active environment contains Isaac Sim 5.1 and
`omni.physx-107.3.26+107.3.3`. A headless startup log reports that
`libomni.physx.plugin.so` registers `omni::physx::IPhysxCustomJoint v1.0` and
`omni::physx::IPhysxJoint v1.0`. String inspection of that same binary finds
`registerCustomJoint`, `PhysXCustomJoint.cpp`, `PxConstraintConnector`, and
validation messages requiring a joint-preparation function and complete joint
callbacks. This is direct evidence that the shipped native plugin contains a
custom-joint registration path.

The corresponding functionality is not exposed by the installed project-level
interfaces:

- `omni/physx/bindings/_physx.pyi` exposes no custom-joint or custom-constraint
  registration class or method.
- Live introspection after `AppLauncher` startup finds no such method on the
  Python `PhysX` object.
- `pxr.PhysxSchema` and `usdrt.PhysxSchema` expose built-in mimic, gear,
  rack-and-pinion and other schemas, but no arbitrary custom-joint schema.
- The installed `omni.physx` extension contains the runtime binary, Python
  stubs and limited documentation, but not `IPhysxCustomJoint.h`,
  `PxConstraint.h`, `PxPhysicsAPI.h`, or a custom-joint sample.

Consequently the feature exists below Python, but the exact ABI contract needed
to compile and register a project-local native extension is absent from the
current repository and Conda installation.

## PhysX capability and GPU implications

The public [PhysX 5.4 custom-constraint
documentation](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/Joints.html#custom-constraints)
describes a C++ extension mechanism built from a stateless solver-preparation
callback, a `PxConstraintShaderTable`, a `PxConstraintConnector`, and
`PxPhysics::createConstraint()`. The solver-preparation callback emits one or
more `Px1DConstraint` rows. Dynamic constraint data must be marked dirty when
changed.

[PhysX articulation loop-closure
documentation](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/Articulations.html#closing-loops)
states that articulation links can participate in ordinary rigid-body joints
used to close loops. This supports the topology needed here: a
base-to-left-wing loop constraint can coexist with the wing's
reduced-coordinate revolute joint, while the existing mimic couples the right
wing.

The [PhysX GPU rigid-body
documentation](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/GPURigidBodies.html)
distinguishes solver execution from joint-shader preparation. D6 joints have a
fully GPU-compatible pipeline; other joint shaders can be prepared on the CPU
and solved on the GPU. A custom constraint can mark itself GPU compatible only
when its rows satisfy the required restrictions. Therefore GPU simulation is
not categorically excluded, but a custom joint does not automatically preserve
the low-overhead, many-environment path used by Isaac Lab.

## Feasible mathematical topology

The most direct constraint is rheonomic: phase is an externally integrated
mechanism state rather than an additional PhysX rigid-body coordinate. At each
physics substep the native constraint data would receive

\[
q_{\mathrm{ref}}=q_{L,\mathrm{mid}}+\Gamma\sin\phi,
\]

\[
\dot q_{\mathrm{ref}}=\Gamma\cos\phi\,\dot\phi.
\]

The solver row would enforce the angular position and velocity relation between
the base and left-wing link. The existing mimic would impose the right-wing
relation. Aerodynamic and inertial loads would change the solved reaction
multiplier rather than the prescribed kinematics:

\[
M(q)\ddot q+h(q,\dot q)=Q_{\mathrm{external}}+J_C(q)^T\lambda,
\]

\[
J_C(q)\dot q=\dot q_{\mathrm{ref}}.
\]

This realizes an ideal prescribed mechanism, not a physical motor model. Its
reaction force and power are useful as required-actuation diagnostics but must
not be interpreted as achievable motor torque without adding actuator limits
and drivetrain dynamics.

A phase rotor modeled as another rigid body would be more literal, but a single
PhysX custom constraint is pairwise between actors. Relating body-relative wing
angle to a separate rotor coordinate introduces a multi-body coordinate
dependency and substantially more connector and Jacobian work. It is not the
smallest proof of concept.

## Feasibility classification

| Question | Finding |
|---|---|
| Can PhysX solve a custom nonlinear equality constraint? | Yes, through the native C++ custom-constraint mechanism. |
| Can current project Python or USD APIs define this constraint? | No exposed entry point was found. |
| Is a one-environment native proof of concept mathematically sound? | Yes: one base-to-left-wing rheonomic loop constraint plus the existing hard mimic. |
| Can it coexist with per-wing aerodynamic wrenches? | Yes in principle; the solver reaction should transmit those loads. |
| Is GPU execution impossible? | No, but custom row compatibility must be proven and preparation may remain on CPU. |
| Is 256-environment Isaac Lab replication currently supported? | Unresolved; no registration, schema, cloning, target-buffer or tensor API is available in the installed package. |
| Can this be implemented safely from the current checkout alone? | No. Exact-version native SDK headers and a supported integration example are missing. |

## Required implementation boundary

A production implementation would be a version-locked native Kit extension,
not another Python drive class. At minimum it needs:

1. the exact `IPhysxCustomJoint v1.0` header and matching PhysX/Carbonite build
   dependencies for `omni.physx 107.3.26`;
2. a registered custom joint type and a supported USD parser/schema path, or a
   documented native lifecycle hook that creates and destroys the constraint
   for every cloned environment;
3. a batched update interface for per-environment phase, reference position and
   reference velocity before every physics substep;
4. a solver-preparation callback with one nonredundant angular row and explicit
   GPU-compatibility handling;
5. reset, cloning, stage-reload and shutdown handling;
6. diagnostics for constraint position error, velocity error, impulse/reaction,
   power and solver warnings.

Updating USD attributes from Python every substep is not an acceptable
substitute for item 3: it would rely on repeated stage edits and physics parsing,
and would not provide a demonstrated batched GPU path.

## Recommendation and gate

Do not implement this option in the current plant branch yet. It is not blocked
by the mathematics or the PhysX solver, but by the missing supported native
development interface in the installed Isaac Sim distribution.

The next evidence-producing gate should be deliberately smaller than the bird
model:

1. obtain the exact Isaac Sim 5.1 / `omni.physx 107.3.26` native SDK header or an
   NVIDIA custom-joint extension sample;
2. build and load a minimal custom angular constraint between two rigid bodies;
3. verify a time-varying sinusoidal target under an external load for one CPU
   environment and one GPU environment;
4. verify cloning and independent target updates for at least two replicated
   environments;
5. only then integrate one base-to-left-wing constraint with the existing mimic
   and measured-wing plant.

If the exact headers and supported registration lifecycle cannot be obtained,
this route should be treated as unavailable for this Isaac Sim installation.
Reconstructing the ABI from binary strings or linking against a different public
PhysX SDK version would be unsafe.

## Commands executed

Repository inspection used `git status`, `git log`, `rg`, `find`, `nl`, and
`strings`. Python introspection was run in `env_isaaclab` through
`./isaaclab.sh -p` after headless `AppLauncher` startup. No simulation matrix,
source test, build, or native compilation was run because this audit did not
change executable code and the required native headers are absent.

## Unresolved questions

- Whether NVIDIA distributes the `IPhysxCustomJoint v1.0` header and a matching
  extension template separately for Isaac Sim 5.1.
- Whether the interface supports a custom joint on articulation links through
  the USD replication path used by Isaac Lab, rather than only direct native
  PhysX creation.
- Whether a GPU-compatible custom row remains efficient enough for the intended
  256-environment workload.
- Whether reaction impulse and power are exposed through a stable API suitable
  for the project's diagnostics.
