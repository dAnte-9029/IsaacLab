# Generic Path-Tracking RL Design

## Goal

Build a single reinforcement learning policy for the flapping vehicle that can track a broad class of missions instead of task-specific controllers for straight flight and loiter only. The policy should consume flight state plus local path context and output the existing four control channels:

- `throttle`
- `rudder`
- `elevon_pitch`
- `elevon_roll`

The final policy should support arbitrary smooth waypoint missions through a unified path-tracking interface, while using the existing PX4-like controller stack only as a training aid and comparison baseline.

## Scope

This design covers:

- mission representation
- internal path representation
- baseline/controller reuse
- RL observation and reward structure
- training curriculum
- validation strategy

This design does not cover:

- sensor-estimation robustness upgrades beyond current baseline support
- full 3D Frenet-frame geometric control
- implementation details for distributed training infrastructure

## Current State

The repository already contains:

- a PX4-like straight-line controller
- a PX4-like loiter controller
- directional guidance with local path tangent and curvature support
- TECS-style longitudinal control
- a teacher-guided straight-flight RL environment

The current RL environment is still task-specific:

- observations and rewards are tied to straight-flight quantities
- the teacher interface is built around a straight-line mission
- evaluation focuses on straight-flight robustness rather than general mission tracking

The current baseline has evidence for:

- straight flight
- loiter

It does not yet have evidence for:

- generic multi-segment missions
- 3D waypoint missions with coupled turn-and-climb behavior

## Design Principles

1. Use a single mission abstraction for baseline and RL.
2. Keep the action interface unchanged.
3. Separate mission definition from control-law implementation.
4. Prefer physically flyable reference paths over mathematically sharp waypoint polylines.
5. Use the PX4-like baseline to accelerate training, not to permanently constrain the learned controller.
6. Expand task distribution over time instead of replacing old tasks.

## Mission Representation

### External Mission Format

The external mission remains a waypoint list, aligned with PX4/QGC-style user workflows.

Each waypoint may carry:

- position
- altitude
- optional loiter semantics

The RL policy should not directly consume the raw waypoint list.

### Internal Path Format

The environment converts the waypoint mission into a smooth, flyable reference path parameterized by path progress `s`.

The first implementation uses a PX4-style `2.5D` representation:

- horizontal path geometry in `xy`
- altitude profile `z(s)`
- implicit free speed selection by the policy, constrained by reward rather than a hard speed profile

This is intentionally not full 3D Frenet geometry in the first version.

## Path Smoothing and Mission Execution

### Horizontal Path

Horizontal mission execution follows PX4-style path management rather than global spline fitting.

The path manager should provide:

- acceptance radius / switch distance logic
- early segment switching
- corner cutting
- dynamically computed turn radius

The turn radius should be computed automatically from vehicle capability, not fixed globally.

A suitable first formulation is:

- define a lateral acceleration limit from roll capability and tuning
- compute the minimum feasible turn radius from horizontal speed
- clip by neighboring segment geometry and safety bounds

### Vertical Path

Altitude should not be piecewise-linear between waypoints. Instead, altitude transitions should be smoothed with a climb/descent feasibility constraint.

The first version should use:

- smooth `z(s)` transitions
- maximum flight-path-angle constraint `gamma_max`

To reduce early complexity, altitude variation is limited to straight segments in the first training distribution:

- straight segments may climb or descend
- turn segments hold altitude
- loiter segments hold altitude

## Unified Path Interface

Both baseline and RL should consume a common local path description at every control step.

The interface should expose:

- closest point on path
- local tangent
- local curvature
- current path progress `s`
- current target altitude `z(s)`

This extends the current baseline structure rather than replacing it. The existing guidance stack already supports local tangent and curvature, so the main missing step is to generalize from special-case `line` and `circle` missions to a path manager that emits the same local geometry for arbitrary missions.

## Baseline Role

The PX4-like baseline remains part of the system, but its role changes.

### What the Baseline Is Good For

- defining a structured local path-tracking reference
- generating demonstration trajectories
- warm-starting policy learning
- recovering from large tracking errors
- acting as an interpretable comparison baseline

### What the Baseline Should Not Do

- permanently constrain policy actions
- define the final runtime controller
- gate the learned policy into only baseline-like behavior

### Training Role

The baseline should be used in three stages:

1. rollout generation for lightweight behavior cloning warm start
2. early PPO training with a masked imitation loss
3. complete removal of teacher influence late in training

The baseline is a recovery teacher, not a lifelong teacher.

## RL Observation Design

### Current-State Features

The policy should observe current flight state, including:

- attitude-related features
- body-frame linear velocity
- body-frame angular velocity
- current height error
- optional short history stacking

### Explicit Current Path Features

The policy should also observe current local path geometry explicitly:

- lateral path error
- height error relative to `z(s)`
- tangent-alignment error
- current curvature
- current path progress

### Future Preview

The policy should receive explicit future path preview points in addition to the baseline’s implicit guidance preview.

The first version uses:

- 5 preview points
- time-based preview rather than distance-based preview

Recommended preview times:

- `0.2 s`
- `0.4 s`
- `0.7 s`
- `1.2 s`
- `2.0 s`

Each preview point initially provides only:

- body-frame relative position `(dx, dy, dz)`

This keeps the observation compact while still encoding future bends and climbs.

## Action Interface

The policy action space remains unchanged:

- `throttle`
- `rudder`
- `elevon_pitch`
- `elevon_roll`

No additional abstraction layer is inserted between the policy and actuator command in the main design.

## Reward Design

The reward should balance tracking quality and mission progress.

### Reward Components

1. path tracking
   - lateral error
   - altitude error
   - tangent/heading alignment error
2. progress
   - reward for path-progress increment `delta_s`
3. flight quality
   - low-speed penalty
   - excessive attitude penalty
   - excessive angular-rate penalty
4. control quality
   - action magnitude penalty
   - action rate penalty
5. termination
   - ground impact
   - severe instability
   - extreme path deviation

### Speed Treatment

The first version does not enforce a hard target speed profile. The policy is allowed to vary speed, but it must still:

- make forward mission progress
- avoid excessively low speed
- maintain stable, usable flight

This avoids forcing the controller onto a single trim condition while preventing degenerate “slow but safe” behavior.

## Training Strategy

### Warm Start

Use baseline-generated rollouts to perform a lightweight behavior-cloning initialization. The goal is not full imitation, only to avoid a cold-start random policy.

### PPO Phase

Train with PPO using the generic path-tracking environment.

Add an imitation term only on a masked subset of states. The imitation loss should not apply globally.

### Recovery-State Mask

Enable imitation only when at least one of the following is true:

- lateral path error exceeds threshold
- altitude error exceeds threshold
- airspeed is too low
- tilt exceeds threshold
- angular rate exceeds threshold

Outside these states, the policy should optimize only the RL objective.

### Teacher Fadeout

The imitation weight should decrease to zero over training. The final policy must run without baseline support.

## Curriculum

The curriculum should expand task diversity rather than replacing earlier tasks.

### Phase 1

- no wind
- random combinations of:
  - level straight
  - climb straight
  - descent straight
  - level large-radius turn
  - level loiter

### Phase 2

- still no wind
- increase mission complexity:
  - more segments
  - smaller turn radii within feasibility limits
  - larger climb/descent demand on straight segments

### Phase 3

- introduce light wind
- keep Phase 1 and Phase 2 missions in distribution

### Phase 4

- stronger wind
- more diverse mission combinations
- later addition of coupled turn-and-climb tasks

The guiding rule is:

- old tasks never disappear entirely from the sampling distribution

## Validation Plan

### Baseline Validation

Before relying on the baseline as a teacher for the new environment, verify it on the unified path interface for:

- level straight
- climbing straight
- descending straight
- large-radius level turn
- level loiter

The baseline does not need to solve all future missions to remain useful, but it must be trustworthy on the primitives used for warm start and recovery guidance.

### RL Validation

Evaluate RL in three layers:

1. simple no-wind missions
2. more complex no-wind missions
3. wind-perturbed missions

### Metrics

Use unified evaluation metrics:

- mean and p95 lateral path error
- mean and p95 altitude error
- path-progress efficiency
- termination rate
- control smoothness

## Alternatives Considered

### Task-Specific Policies

Rejected because separate straight/loiter/multi-task policies would not naturally generalize and would create policy switching and forgetting problems.

### High-Level Teacher Only

Rejected as the primary path because it risks making RL a thin actuator allocator underneath a classical controller, which does not match the goal of eventually letting RL take over mission execution.

### Full 3D Frenet Formulation From Day One

Rejected for first implementation because it adds substantial geometric and debugging complexity before the generic path-tracking interface is proven.

## Main Risks

1. Baseline may not remain reliable on generalized path primitives.
2. Preview representation may initially be too weak or too strong.
3. Reward may encourage pathological speed choices.
4. Mission generation may create edge cases outside the feasible flight envelope.
5. Imitation mask thresholds may be too permissive or too restrictive.

## Risk Mitigations

1. Validate baseline on primitives before using it for teaching.
2. Start with 5 preview points and expand only if needed.
3. Include progress reward and low-speed penalty together.
4. Constrain mission generator with roll, curvature, and path-angle feasibility limits.
5. Log recovery-mask activation rates and inspect teacher dependence during training.

## Recommended Next Step

Create a new generic path-tracking environment rather than further specializing the current straight-flight RL environment. Treat straight and loiter as mission instances of the new environment, then connect the existing PX4-like baseline to the same path interface for warm start, recovery guidance, and comparison.
