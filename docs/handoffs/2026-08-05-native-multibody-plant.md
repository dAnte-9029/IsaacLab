# Native measured-wing multibody plant handoff

## 1. Stage identity

- Integration branch: `flapping_rl`
- Completed implementation commit: `8a50c5d1c5dbf4e1988d7db5146c07f17929e336`
- Stage range: `cbdec2106b2e82bbcfbb355fab8cfd51fdda56ff..8a50c5d1c5dbf4e1988d7db5146c07f17929e336`
- Date: 2026-08-05

The former `feat/physx-wing-multibody` worktree was clean and removed after
the completed history was fast-forwarded and pushed to `origin/flapping_rl`.

## 2. Objective and outcome

The stage replaced the canonical near-single-rigid-body commanded-wing plant
with an explicit measured-wing PhysX multibody plant. The accepted mechanism
uses a project-local native rheonomic constraint to prescribe the common
sinusoidal stroke while PhysX solves the body and wing reaction dynamics.

`FlappingBotStraightFlightDeLaurierEnvCfg` now selects the native multibody
plant by default. The former behavior remains explicitly selectable as
`FlappingBotStraightFlightCommandedKinematicsDeLaurierEnvCfg`.

## 3. Approved plant contract

The authoritative decisions are recorded in:

- `docs/decisions/ADR-2026-07-29-engineering-flap-phase-sine.md`
- `docs/decisions/ADR-2026-07-29-measured-multibody-mass-properties.md`
- `docs/decisions/ADR-2026-07-29-measured-wing-multibody-plant.md`
- `docs/decisions/ADR-2026-08-03-native-holonomic-wing-mechanism.md`
- `docs/decisions/ADR-2026-08-04-promote-native-multibody-default.md`
- `docs/decisions/ADR-2026-08-05-native-holonomic-load-and-transient-diagnostics.md`

The resulting canonical configuration is CPU PhysX with measured body and wing
properties, opposed left/right wing motion, per-wing DeLaurier loads,
`dt=1/480 s`, decimation 4, 16 position and 4 velocity solver iterations,
`retain_accelerations=False`, and `scene.replicate_physics=False`. The physical
stroke is `q=Gamma*sin(phase)`, with phase zero at the neutral pose and the wing
starting its upstroke.

The project-local `omni.flapping_bot.holonomic_constraint` extension must be
built for the installed Isaac Sim/PhysX ABI and enabled at Kit startup. The
environment fails closed when the extension is absent or an incompatible GPU
or replication configuration is requested.

## 4. Main implementation surfaces

The exact file list is available from the stage commit range. The primary
surfaces are:

- `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`
- `source/flapping_bot/flapping_bot/physics/measured_mass_properties.py`
- `source/flapping_bot/flapping_bot/physics/multibody_mass_distribution.py`
- `source/flapping_bot/flapping_bot/physics/multibody_wing_coupling.py`
- `source/flapping_bot/flapping_bot/physics/ideal_inverse_dynamics_phase_drive.py`
- `source/flapping_bot/flapping_bot/assets/native_holonomic_drive.py`
- `source/flapping_bot/native_extensions/omni.flapping_bot.holonomic_constraint/`
- `source/flapping_bot/flapping_bot/analysis/native_holonomic_validation.py`
- `source/flapping_bot/flapping_bot/analysis/native_holonomic_transient_validation.py`
- `scripts/flapping_px4/validate_native_holonomic_mechanism.py`
- `scripts/flapping_px4/validate_native_holonomic_transients.py`
- the corresponding project tests and audit/ADR documents.

## 5. Validation completed

Use the committed reports for full commands, configurations, tables and claim
boundaries:

- `docs/audits/2026-07-29-physx-sfwm-inertial-validation.md`
- `docs/audits/2026-07-30-multibody-aerodynamic-validation.md`
- `docs/audits/2026-08-03-native-holonomic-validation.md`
- `docs/audits/2026-08-04-multibody-training-runtime-feasibility.md`
- `docs/audits/2026-08-05-native-holonomic-transient-diagnostics.md`

The native mechanism, per-wing aerodynamic coupling, free-body momentum,
reset/lifecycle, short uncontrolled flight and continuous-frequency numerical
gates were executed. The continuous-frequency matrix supports `1/480 s` and
`1/1000 s` under the 0.1 degree tracking gate; `1/240 s` fails. Left/right
synchronization remained below approximately `4e-6 deg`.

The final integration step reran 32 targeted pure/contract tests and passed
`py_compile` and `git diff --check`. The latest audit records the larger Isaac
Sim gates and the 18-case fixed/free-root transient matrix.

The optional mechanism load and power outputs are multibody inverse-dynamics
estimates at the ideal common wing coordinate. They are not the exact PhysX
constraint multiplier, motor-shaft torque or electrical power. The
actual-joint-acceleration DeLaurier calculation remains a non-applied shadow;
the canonical applied aerodynamic input uses prescribed analytical
acceleration.

## 6. Known limitations

- The native custom constraint is CPU-only; direct-GPU PhysX rejects it.
- A fresh Isaac Sim process is the supported formal-run lifecycle because full
  same-process environment destruction and reconstruction remains unresolved.
- Existing controller gains and learned policies have not been validated
  against the promoted plant.
- The current 1024-environment CPU figure is a narrow fixed-root runtime probe,
  not an end-to-end RL training-throughput result.
- Uncontrolled free-body tests show finite numerical behavior and integration
  evidence, not attitude stability, task success, model identification,
  sim-to-real evidence or real-flight validation.
- Exact native constraint force and motor/gearbox feasibility are not claimed.
- The direct-GPU implicit-drive candidate is an explicit approximation and is
  not the authoritative multibody plant.
- Additional limitations and superseded experimental paths are indexed in
  `docs/PROJECT_STATE.md`.

## 7. Unresolved human decisions

No new plant decision is required before the canonical-task smoke. Before
controller retuning, define and approve the tuning scenarios, gain bounds,
success/failure criteria and held-out evaluation scenarios. A detailed
motor/gearbox model remains deferred unless measured drivetrain parameters and
a motor-shaft-level research objective are introduced.

## 8. Exact next task

1. Run a fresh-process smoke of the canonical task through the normal training
   launcher with the required native extension startup arguments.
2. Rebuild the straight-flight controller baseline against the promoted plant
   in a separate controller branch.
3. Separately implement a reproducible 1024-environment CPU-native end-to-end
   RL throughput smoke with free root, the complete wing/tail plant,
   observations, rewards, terminations, repeated batched resets, policy
   inference and separated physics/transfer/optimizer timings.

## 9. Scope for the next stage

Allowed:

- canonical task startup and fresh-process smoke diagnostics;
- controller-only profiles, diagnostics and tests after the tuning contract is
  approved;
- a separate CPU-native end-to-end throughput benchmark.

Prohibited without a separate decision:

- changing mass, inertia, aerodynamic coefficients or wing-mechanism settings
  to make the previous controller gains work;
- substituting the direct-GPU implicit-drive approximation for the canonical
  native plant;
- combining controller tuning, plant modification and held-out evaluation in
  one change;
- presenting numerical integration gates as aerodynamic-model or real-flight
  validation.

## 10. Clean continuation and reproduction

The repository root may be occupied by unrelated work. Start from a new
worktree rather than switching or cleaning that checkout:

```bash
cd /home/zn/IsaacLab
git fetch origin flapping_rl
git worktree add .worktrees/<next-worktree> -b <next-branch> origin/flapping_rl
cd .worktrees/<next-worktree>
conda activate env_isaaclab
rg --files -g AGENTS.md
```

Build the native extension when the installed ABI artifact is unavailable:

```bash
scripts/flapping_px4/build_holonomic_constraint_extension.sh
```

Canonical Isaac Sim processes must include the extension startup arguments
documented in
`docs/decisions/ADR-2026-08-04-promote-native-multibody-default.md`.

## 11. Suggested skills

- `get-available-resources`: use before the 1024-environment end-to-end CPU
  throughput experiment.
- `grill-with-docs`: use if the controller tuning and held-out evaluation
  contract needs design review.
- `domain-modeling`: use only if plant/controller terminology or interface
  ownership changes.
