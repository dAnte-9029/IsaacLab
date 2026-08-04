# ADR: Promote native measured-wing multibody plant

- Status: Accepted
- Date: 2026-08-04
- Scope: Canonical project DeLaurier environment configuration
- Depends on: `ADR-2026-08-03-native-holonomic-wing-mechanism.md`

## Context

The project-local native trajectory constraint has passed the accepted CPU
mechanism, aerodynamic-load, momentum, reset, lifecycle and short uncontrolled
flight gates. The remaining limitations are known runtime constraints rather
than unresolved plant equations: the custom constraint is CPU-only, requires
physics replication to be disabled, must be loaded at Kit startup, and has not
passed same-process full-stage reconstruction.

The canonical DeLaurier task still selected the historical commanded wing
kinematics, near-single-rigid-body mass redistribution and equivalent base
wrench. That default bypassed wing inertia and joint reaction dynamics, so it
no longer represented the accepted plant.

## Decision

`FlappingBotStraightFlightDeLaurierEnvCfg` now inherits from
`FlappingBotStraightFlightMeasuredWingMultibodyNativeHolonomicDeLaurierEnvCfg`.
This changes the canonical DeLaurier task and all teacher, pure-RL and path
tracking configurations derived from it to the following plant contract:

```text
plant_variant = measured_wing_multibody
wing_drive_variant = native_holonomic_drive
wing_aero_coupling_mode = native_holonomic_per_wing_link
sim.device = cpu
sim.dt = 1/480 s
decimation = 4
control_dt = 1/120 s
scene.replicate_physics = false
solver iterations = 16 position / 4 velocity
retain_accelerations = false
```

Flight defaults remain 2 to 5 Hz with a 4 Hz reset and both wing and tail
aerodynamics enabled. The legacy behavior remains available through the
explicit project-local
`FlappingBotStraightFlightCommandedKinematicsDeLaurierEnvCfg`; it is no longer
the canonical DeLaurier task.

## Launch contract

The native extension is ABI-bound to the installed Isaac Sim 5.1 and matching
PhysX 107.3 development tree. Build it from an activated project environment:

```bash
conda activate env_isaaclab
scripts/flapping_px4/build_holonomic_constraint_extension.sh
```

Every process constructing the canonical DeLaurier task must load the extension
when Kit starts and must keep PhysX on CPU:

```bash
./isaaclab.sh -p <script.py> --device cpu \
  --kit_args "--ext-folder $(pwd)/source/flapping_bot/native_extensions --enable omni.flapping_bot.holonomic_constraint"
```

The environment deliberately fails closed if the extension is absent, the
simulation device is CUDA, physics replication is enabled, or the aerodynamic
coupling does not match the native mechanism.

## Consequences

- Canonical DeLaurier simulations now include measured wing mass and inertia,
  the hard opposed-wing gear relation, native trajectory reactions and
  per-wing aerodynamic load transfer.
- Existing controller gains and learned policies are not claimed compatible.
  Closed-loop baselines must be rebuilt against this plant.
- GPU PhysX and large-scale direct-GPU RL are unavailable for the authoritative
  plant. The measured 1024-environment fixed-root probe reached about 12,006
  environment steps per second on CPU; this is not an end-to-end training-time
  guarantee.
- A fresh Isaac Sim process is the supported lifecycle for formal runs until
  same-process full-stage reconstruction is fixed.
- The old commanded-kinematics plant remains selectable for A/B comparisons,
  regression reproduction and controller migration diagnostics.

## Validation required for this promotion

- Pure configuration contracts must prove that the canonical class selects the
  native measured-wing class and that the legacy class remains explicit.
- The native extension build and ABI load smoke must pass in `env_isaaclab`.
- A canonical configuration construction or short fresh-process environment
  smoke must confirm CPU, `1/480 s`, decimation 4, disabled replication and the
  native extension startup path.
- Existing native mechanism and aerodynamic regression tests must remain green.

## Deferred work

- Rebuild the straight-flight and path-tracking controller baselines.
- Add a reproducible end-to-end CPU RL throughput smoke including policy,
  observations, rewards, resets and optimizer work.
- Resolve same-process full-stage reconstruction if interactive workflows need
  repeated environment teardown and creation.
- Treat the direct-GPU implicit-drive plant only as an explicit approximation;
  it does not replace the native reference.
