# Wing-Tail-Controller Interfaces

Date: 2026-07-13. The first four sections describe current interfaces. The `WingWrench` section is **Proposed, not implemented**.

## Current wing-model inputs and outputs

| Interface | Inputs | Outputs | Current caller |
|---|---|---|---|
| `compute_delaurier_strip_loads` + `integrate_delaurier_strip_wrench` | `(B,N_strip)` prescribed kinematics, `WingGeometry`, rho/U/theta parameters, `DeLaurierParams` | raw attached-flow components `(B,N_strip)`, then `F_c/M_c (B,3)` about wing-root origin | `_compute_wing_delaurier_wrench` default `strip_integrated` mode |
| `compute_aero_wrench_delaurier1993` | same | legacy aggregate `F_c (B,3)`, zero `tau_c (B,3)`, power, separation ratio | legacy fixed-quarter-chord mode and compatibility callers |
| `QuasiSteadyWingModel.compute_forces` | joint state, root linear/angular velocity | per-wing `forces (N,W,3)`, `torques (N,W,3)`, hinge torque | simple baseline path |
| `_compute_wing_delaurier_wrench` | `v_air_b`, cached command kinematics, robot link/world data | net `F_b (N,3)`, `tau_b (N,3)` | `_apply_action` |

The new strip result exposes raw named scalar components, geometry and named wing-origin moment diagnostics in Wang frame. It does not implement corrected-force distribution.

## Current tail-model inputs and outputs

| Interface | Inputs | Outputs | Notes |
|---|---|---|---|
| `TailAeroModel.compute_wrench` | `root_lin_vel_b`, `root_ang_vel_b`, split elevons, rudder, optional `base_com_pos_b` | net force and moment `(N,3)` in body frame | preferred split-elevon interface; symmetric `elevator_rad` remains fallback |
| `_surface_wrench` | one `TailSurfaceCfg`, local flow inputs, surface deflection, reference COM | `force_b`, `torque_b` | torque is `r x F`; no intrinsic section moment |
| env action allocation | normalized action `[freq,rudder,pitch,roll]` | flap Hz, rudder rad, left/right elevon rad | joint and command limits applied in environment |

## Current controller inputs and outputs

| Controller | Inputs | Output | Current use |
|---|---|---|---|
| `PX4LikeStraightLineController.compute_actions` | local position, ground velocity, optional wind, Euler attitude, body rate | `(N,4)` normalized action + diagnostics | external straight script or environment teacher |
| `PX4LikeLoiterController.compute_actions` | same plus circle config held in controller | `(N,4)` + diagnostics | external loiter script |
| `PX4LikePathTrackingController.compute_actions_from_query` | same plus `closest_point_xyz`, tangent, curvature, height query | `(N,4)` + diagnostics | path environment teacher |

Controllers do not receive wing/tail coefficients, raw wrenches, or corrected-model output. They receive limits and mission/state data. They are stateful: TECS filters/integrators and inner pitch action integral reset through `reset()`.

## Environment integration interface

The active integration boundary is internal to `FlappingBotStraightFlightEnv._apply_action`:

```text
(F_wing_b, M_wing_b) + (F_tail_b, M_tail_b) + F_drag_b
    -> set_external_force_and_torque(forces=(N,1,3), torques=(N,1,3), body_ids=base_link, is_global=False)
```

`Articulation.set_external_force_and_torque` buffers force/torque only; Isaac applies it during `write_data_to_sim` before the physics step. Existing debug caches only retain aggregate wing/tail/net forces and net torque.

## Current coupling points

- Environment owns DeLaurier coordinate mapping, equivalent AC closure, tail construction, wind conversion and final wrench sum.
- `FlappingBotPathTrackingEnv` inherits this plant and replaces mission/teacher/reward/termination behavior.
- Tail compatibility effectiveness is an environment config override passed into `TailAeroCfg`.
- Controller tuning scripts choose controller profiles based on state source, not aerodynamic model selection.

## Undesirable coupling

- Aerodynamic frame mapping and moment closure live in the direct environment instead of a named physics output contract.
- Stabilization/plant controls (mass/COM/inertia override, induced/parasite drag, virtual moments and reset hold) coexist with task orchestration; comparative experiment manifests must record them.
- A complete-vehicle corrected `Fx/Fz` cannot be assigned to the wing component without an approved ownership/allocation rule.

## Proposed, not implemented: `WingWrench` contract

This is an interface proposal only; it does not select a correction or moment-closure algorithm.

```text
WingStripLoads
  force_c_or_b: (N_env, 2, N_strip, 3) N
  point_b_from_reference: (N_env, 2, N_strip, 3) m
  reference: explicit enum (base_com / base_origin)
  frame: explicit enum (body_flu / wing_link / world)
  intrinsic_moment_b: optional (N_env, 2, N_strip, 3) N m
  prior_force_b: (N_env, 2, 3) N
  prior_moment_b_about_reference: (N_env, 2, 3) N m

WingWrench
  force_b: (N_env, 3) N
  moment_b_about_base_com: (N_env, 3) N m
  provenance: disabled/prior/corrected and closure identifier
  diagnostics: named, optional tensors
```

Compatibility requirements:

1. `disabled` must reproduce the existing DeLaurier equivalent-AC force and moment within a declared tolerance.
2. Existing task IDs, tail model and controller APIs remain unchanged.
3. The environment should receive one complete, explicitly referenced `WingWrench`, then add unchanged tail/drag contributions.
4. No implicit FRD/FLU conversion; conversion must occur at named adapter boundaries.

## Proposed diagnostics

- per-wing/per-strip prior force and application point;
- aggregate prior, corrected and applied wing force/moment;
- raw tail and drag wrench separately;
- force/moment conservation residuals;
- frame/reference identifiers and phase/frequency inputs;
- correction envelope/OOD/fallback flags if a model is approved;
- action, saturation and controller setpoint diagnostics already partly cached.

## Minimum future integration placement

- Load a corrected model in a dedicated physics-side adapter initialized by explicit environment config, not in a controller or script.
- Compute closure in a pure physics function that consumes a declared strip-load object and correction result.
- Keep `_apply_action` as orchestration: request a `WingWrench`, add tail/drag, then send one final base-link wrench.
- Future likely files: `physics/qsm_delaurier1993.py` (strip contract), a new physics data-type/module, and `direct/flapping_bot/straight_flight_env.py` (adapter wiring/diagnostics), plus focused tests/docs.
- Files that should not be modified for this integration: `px4_like` gains/controllers, `path_tracking` mission/reward logic, upstream `source/isaaclab`, robot assets, unless a separate approved task proves a change is necessary.
