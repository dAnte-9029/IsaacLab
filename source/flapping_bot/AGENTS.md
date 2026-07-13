
# Flapping-Bot Extension Guidelines

These instructions apply to all files under `source/flapping_bot/`.

## 1. Architecture boundaries

Keep the following responsibilities separate:

* `flapping_bot/physics/`: aerodynamic, actuator, force, moment, and plant-model calculations.
* `flapping_bot/px4_like/`: controller logic, estimators, control allocation, and controller profiles.
* `flapping_bot/direct/`: Isaac Lab environment orchestration, state collection, model calls, force application, reset, observation, reward, and termination.
* `flapping_bot/path_tracking/`: path representation, task references, guidance, mission progression, and tracking utilities.
* `flapping_bot/config/`: explicit configuration values and model-selection settings.
* `flapping_bot/assets/` and `flapping_bot/scenes/`: robot and scene definitions.

Do not move responsibilities across these boundaries merely for implementation convenience.

## 2. Dependency direction

Preferred dependency direction:

```text
config / shared data types
        ↓
physics and control modules
        ↓
environment integration
        ↓
scripts and evaluation workflows
```

Rules:

* Physics modules must not import controllers, rewards, missions, or environments.
* Controllers must not modify aerodynamic coefficients, inertia, mass, tail geometry, or other plant parameters.
* Path-tracking modules must not implement aerodynamic equations.
* Environment files should orchestrate modules rather than contain new model equations.
* Scripts should call package APIs rather than duplicate package logic.

Avoid circular imports.

## 3. Physical and controller separation

Treat the following as separate categories:

### Plant parameters

* aerodynamic coefficients;
* force and moment models;
* mass and inertia;
* geometry and application points;
* actuator effectiveness;
* actuator limits and delays when represented as plant properties.

### Controller parameters

* rate-loop gains;
* attitude-loop gains;
* position, speed, altitude, and path-loop gains;
* filters and anti-windup settings;
* controller-specific output limits.

Do not tune plant parameters and controller gains together inside the same implementation task unless the experiment explicitly studies joint calibration.

## 4. Configuration discipline

* Put user-selectable behavior in typed configuration objects or explicit configuration files.
* Do not introduce unexplained magic coefficients inside environment or controller code.
* Preserve existing task IDs, public entry points, and configuration names unless an approved migration is provided.
* New model variants must be selectable explicitly.
* Configuration defaults must be documented and deterministic.

## 5. Tensor and numerical requirements

For tensor-based code:

* preserve batch dimensions;
* preserve input device and dtype;
* avoid creating CPU tensors inside GPU execution paths;
* use explicit broadcasting;
* guard divisions and inverse operations near singular values;
* detect or prevent NaN and Inf propagation;
* avoid unnecessary per-environment Python loops.

Do not use `squeeze()` without specifying the intended dimension when batch size may be one.

## 6. Logging and diagnostics

New plant or controller components should expose diagnostics needed to distinguish contributions, such as:

* prior force;
* corrected force;
* wing moment;
* tail force and moment;
* controller setpoints;
* actuator commands;
* saturation state;
* termination reason.

Do not add large unconditional logs to every simulation step. Logging must be configurable.

## 7. Compatibility and tests

For each new feature:

* retain a baseline mode;
* verify disabled-feature equivalence when applicable;
* add deterministic unit tests for pure logic;
* add a minimal integration or smoke test when the feature touches an environment;
* state which Isaac Sim or GPU-dependent checks were not run.

## 8. Scope control

Before modifying a specialized subdirectory, read its nearest `AGENTS.md`.

When a requested change spans physics, control, environment, and scripts, split it into staged changes unless there is a strong technical reason not to do so.
