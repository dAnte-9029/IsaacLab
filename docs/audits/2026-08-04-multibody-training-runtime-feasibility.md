# Multibody training runtime and GPU feasibility audit

Date: 2026-08-04
Branch: `feat/physx-wing-multibody`
Working-tree HEAD: `1798a1abbbb3bf3297ef0a37e0790b25d0ef7093`

## Scope and claim boundary

This audit records a narrow runtime and capability investigation after the
CPU-only limitation of the native holonomic wing constraint was confirmed. It
answers three questions:

1. which multibody plant is currently the latest and highest-fidelity
   implementation in this working tree;
2. whether the measured-wing plant can run at large batch sizes on direct-GPU
   PhysX without the custom constraint; and
3. how the raw environment-step throughput of the CPU native plant compares
   with that GPU-capable candidate.

The probes used a fixed root, deterministic 8 m/s relative flow, 4 or 5 Hz
flapping, a 1/480 s physics step, a 1/120 s environment step and one second of
simulation. They do not constitute free-flight, controller, RL convergence or
sim-to-real validation. The helper scripts were temporary files under `/tmp`
and were not added to the repository; the reported configuration and results
therefore require a formal project-local runner before they become a
reproducible regression gate.

## Current implementation status

The latest and highest-fidelity multibody implementation in the working tree
is:

```text
plant_variant = measured_wing_multibody
wing_drive_variant = native_holonomic_drive
wing_aero_coupling_mode = native_holonomic_per_wing_link
environment config =
  FlappingBotStraightFlightMeasuredWingMultibodyNativeHolonomicDeLaurierEnvCfg
```

Concrete code evidence:

- `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py:659`
  defines the native measured-wing environment;
- the DeLaurier per-wing specialization is defined at the same file's line 687;
- `source/flapping_bot/native_extensions/omni.flapping_bot.holonomic_constraint/plugins/FlappingWingTrajectoryConstraint.cpp:160`
  prepares the native solver row;
- the custom joint is registered at line 354 of that C++ file;
- `source/flapping_bot/flapping_bot/assets/ideal_coupled_drive.py:135`
  applies the hard opposed-wing mimic relation.

This plant keeps measured body and wing mass properties, solves one native
rheonomic left-wing trajectory row together with the hard opposed-wing mimic,
and applies the two DeLaurier wrenches to the corresponding wing links. It is
the accepted numerical reference described by
`docs/decisions/ADR-2026-08-03-native-holonomic-wing-mechanism.md` and validated
by `docs/audits/2026-08-03-native-holonomic-validation.md`.

At the time of this probe, three status distinctions were important:

- latest working-tree implementation: the CPU native holonomic DeLaurier
  plant above;
- latest committed branch implementation: commit `1798a1ab` adds measured
  per-wing aerodynamic loads, while the native extension and its validation
  remain uncommitted working-tree work;
- repository default at probe time: the established commanded-kinematics,
  near-single-rigid-body plant was still the default. It was subsequently
  promoted by
  `docs/decisions/ADR-2026-08-04-promote-native-multibody-default.md`.

## Why the native constraint remains CPU-only

The environment fails closed when `sim.device` starts with `cuda`, at
`source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py:838`,
and requires `scene.replicate_physics=False` at line 843.

This is an API capability boundary rather than a tuned parameter:

- PhysX marks `PxConstraintFlag::eGPU_COMPATIBLE` as internal-only in
  `/home/zn/PhysX-107.3/physx/include/PxConstraint.h:63`;
- `/home/zn/PhysX-107.3/physx/source/physx/src/NpConstraint.cpp:295` rejects
  attempts to set that flag through the API;
- the Omni custom-joint registration surface accepts a CPU
  `PxConstraintSolverPrep` callback but exposes no GPU joint-preparation
  callback;
- the direct-GPU joint manager interprets compatible constraint data as
  `PxgD6JointData` at
  `/home/zn/PhysX-107.3/physx/source/gpusimulationcontroller/src/PxgJointManager.cpp:358`.

Consequently, manually raising the internal flag on the project custom
constraint would not make it GPU compatible and could cause an invalid data
reinterpretation.

## Direct-GPU measured-wing capability probe

The GPU candidate reused the existing explicit variant:

```text
FlappingBotStraightFlightMeasuredWingMultibodyIdealCoupledDeLaurierEnvCfg
```

It uses the same measured link properties and per-wing aerodynamic wrenches,
but replaces the native trajectory row with the existing PhysX implicit
position/velocity drive on the left wing. The hard mimic remains responsible
for the opposed right wing. The relevant dispatch is at
`source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py:2092`.

Every aerodynamic probe used:

```text
sim.device = cuda:0
scene.replicate_physics = true
retain_accelerations = false
physics_dt = 1/480 s
environment_dt = 1/120 s
fixed_root = true
airspeed = 8 m/s
duration = 1 s
```

Results:

| Frequency | Environments | Finite | Maximum tracking error | Steady maximum tracking error | Maximum opposed synchronization error | Maximum single-wing force | Raw throughput |
|---:|---:|---|---:|---:|---:|---:|---:|
| 4 Hz | 64 | yes | 0.531259 deg | 0.357037 deg | 0.001086 deg | 16.9212 N | not recorded |
| 4 Hz | 1024 | yes | 0.531259 deg | 0.357037 deg | 0.001086 deg | 16.9212 N | 47,653.5 env-steps/s |
| 5 Hz | 1024 | yes | 0.701540 deg | 0.433466 deg | 0.001455 deg | 20.7042 N | 47,497.9 env-steps/s |

A separate 64-environment, 4 Hz, no-aerodynamics probe remained finite with a
0.290635 degree maximum tracking error and a 0.000444 degree synchronization
error.

These results establish that direct-GPU PhysX, fast physics replication, the
measured-wing articulation, hard mimic and per-wing aerodynamic load path can
operate together at 1024 environments. They do not qualify the implicit drive
as equivalent to the native mechanism. Its trajectory error is load-dependent
and about one order of magnitude larger than the accepted CPU native
trajectory error.

The historical aerodynamic rejection of the implicit-drive path cannot be
reused as contrary evidence because those runs retained and resubmitted forces
every step. The corrected probes used `retain_accelerations=False`; a complete
2/3/4/5 Hz requalification is still pending.

## CPU native throughput probe

The CPU probe used the accepted native holonomic DeLaurier plant with the same
fixed-root, aerodynamic, time-step and duration conditions. Physics
replication remained disabled as required by the custom joint.

| Environments | Scene creation | One-second loop time | Raw throughput |
|---:|---:|---:|---:|
| 64 | 2.578 s | 3.372 s | 2,277.6 env-steps/s |
| 256 | 4.913 s | 5.269 s | 5,830.9 env-steps/s |
| 1024 | 32.906 s | 10.235 s | 12,005.8 env-steps/s |

At the same 1024-environment scale, the measured raw step-throughput ratio was

\[
\frac{47{,}653.5}{12{,}005.8}=3.97.
\]

GPU scene creation was about 3.35 s at 1024 environments, versus 32.91 s for
the CPU native plant. Scene creation is normally amortized over a long
training process, so the approximately fourfold raw step-throughput ratio is
the more relevant preliminary result.

For scale only, and excluding policy inference, optimization, reset, logging
and transfer overhead, 100 million environment steps would take approximately

\[
T_{\mathrm{CPU}}=\frac{10^8}{12{,}005.8}=2.31\ \mathrm{h},
\]

\[
T_{\mathrm{GPU}}=\frac{10^8}{47{,}653.5}=0.58\ \mathrm{h}.
\]

These are not training-time promises. A formal RL throughput smoke must include
the free root, complete observations, rewards, terminations, batched resets,
CPU-to-GPU transfers and policy/optimizer work.

## Strict-trajectory GPU alternatives

The current installed PhysX version offers no supported public route for an
arbitrary rheonomic custom constraint that is both solver-level hard and
direct-GPU compatible.

The available architectural alternatives are:

1. train directly with the CPU native plant;
2. train with the GPU implicit-drive surrogate and accept policies only after
   CPU-native evaluation;
3. implement a project-local prescribed-coordinate multibody solver in
   batched Torch/CUDA; or
4. provide explicit linkage geometry using only GPU-compatible built-in
   constraints, or undertake a deep PhysX GPU solver fork.

The third option can impose

\[
q_r(t)=\Gamma\sin\phi(t)
\]

exactly at the model level while solving the full body response from a
partitioned multibody equation,

\[
M_{bb}\ddot x_b
=W_b-h_b-M_{bq}\ddot q_r.
\]

It can retain measured wing inertia, time-varying system mass properties,
Coriolis/centrifugal terms and per-wing aerodynamic loads, but it would be a
specialized GPU dynamics implementation rather than a PhysX articulation
constraint. It therefore requires its own ADR and validation against the CPU
native reference before implementation.

## Interpretation and next gate

The CPU native holonomic DeLaurier plant remains the current authoritative
multibody reference. The GPU implicit-drive result is a promising training
capability finding, not a promotion decision and not a replacement for the
native plant.

Before selecting a training architecture, run a formal 1024-environment CPU
native RL throughput smoke containing the complete free-flight environment,
policy inference, reward/termination computation and repeated batched resets.
If its end-to-end throughput is acceptable, direct CPU-native training avoids
the load-dependent soft-trajectory gap. If it is not acceptable, create a new
ADR comparing the validated GPU surrogate against a prescribed-coordinate GPU
multibody plant.
