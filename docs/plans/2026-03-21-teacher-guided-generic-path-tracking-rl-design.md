# Teacher-Guided Generic Path-Tracking RL Design

## Goal

Build a single path-tracking RL policy for the DeLaurier flapping-wing vehicle that starts from the current PX4-like teacher baseline, gradually reduces teacher dependence, and ends with a pure RL policy that can track mixed missions under varying conditions. The policy keeps the real-vehicle control interface:

- `throttle` / flap frequency
- `rudder`
- `elevon_pitch`
- `elevon_roll`

## Current Starting Point

The repository now has a usable PX4-like teacher baseline for:

- straight flight
- loiter
- random multi-segment 2.5D path missions
- no-wind, steady crosswind, and simple time-varying wind checks

The baseline uses truth state and is already integrated into the path-tracking environment. This baseline is good enough to serve two purposes:

1. as an interpretable comparison controller
2. as an early-stage training prior for RL

## Chosen RL Strategy

The selected training strategy is **teacher-guided PPO with eventual teacher removal**.

This means:

1. start with PPO in the real action space
2. use the teacher only to constrain early exploration
3. optionally use short behavior cloning warm start if it reduces instability
4. anneal teacher guidance until the policy can fly without it

This is intentionally **not** residual RL. The final deployed controller should not require the PX4-like baseline online.

## Task Definition

The RL task is **generic path tracking**, not separate “straight” and “loiter” controllers.

The environment should generate missions from a unified path manager and expose the same local path interface for every mission type. Initial training will still bias the mission distribution toward easier cases, but the policy interface remains the same from day one.

The first mission family should include:

- level straight segments
- level turns
- level loiter
- multi-segment combinations of straight / turn / loiter
- altitude changes on straight segments only

Turn and loiter altitude remain fixed in the first version to keep the vertical problem simpler.

## Observation Design

The first RL version should use **truth state** rather than estimated state. This isolates control-learning from estimator quality.

The policy observation should contain four groups of information.

### 1. Vehicle state

- body-frame linear velocity
- body-frame angular velocity
- projected gravity / attitude representation
- current height relative to the local path reference

### 2. Current path geometry

- current closest-point relative position
- local tangent direction
- local curvature
- lateral tracking error
- height tracking error
- path-alignment error

### 3. Future preview

The policy should receive **5 future preview points** expressed in the body frame. The agreed initial design is:

- explicit current path geometry
- plus 5 future point positions

This gives the policy enough look-ahead to handle bends and upcoming altitude changes without forcing it to reproduce the baseline’s internal guidance math.

### 4. Control context

- previous action
- possibly current actuator state / flap frequency if not already implicit

Wind truth should **not** be included in the observation, even when wind exists in training. The policy must infer wind from motion and path error.

## Action Design

The action interface remains exactly:

- `action_freq`
- `action_rudder`
- `action_elevon_pitch`
- `action_elevon_roll`

No intermediate command abstraction will be introduced in the first version.

## Reward Design

The reward should prioritize mission completion and tracking quality, not teacher imitation.

### Core positive terms

- path-progress reward via `delta_s`
- small bonus for staying alive / continuing valid flight

### Core error penalties

- lateral path error
- height error
- tangent / course alignment error

### Flight-quality penalties

- excessive attitude / angular rate
- action magnitude and action rate
- too-low airspeed / near-stall behavior
- crash / out-of-bounds / timeout termination

### Important non-goal

The reward should **not** force tracking of a fixed cruise speed such as `7 m/s`. Speed can vary as long as the vehicle tracks the path and remains flyable. Only low-speed safety and pathological behavior should be penalized.

## Teacher Usage

The teacher’s role is limited and staged.

### Stage A: strong teacher envelope

Use the current teacher-action envelope to keep exploration local while PPO learns basic control.

### Stage B: weak teacher envelope

Relax the allowed deviation from teacher actions so the policy can discover better behaviors.

### Stage C: teacher-off continuation

Disable teacher guidance and continue PPO from the best weak-teacher checkpoint.

The teacher should also remain available for:

- behavior cloning data generation
- benchmark comparison
- debugging failed RL runs

## Behavior Cloning

Behavior cloning is optional, not mandatory.

If early PPO proves slow or unstable, use a short BC warm start:

1. collect teacher rollouts on the same generic path-tracking environment
2. pretrain the policy on those trajectories
3. hand the policy to PPO for fine-tuning

BC should only accelerate initialization. It should not replace PPO.

## Curriculum

The agreed curriculum is:

1. truth observations first
2. no-wind and mild missions first
3. harder mission mixtures next
4. teacher removal after the policy is already competent
5. wind curriculum after the no-wind pure-RL policy is stable
6. estimated-state / noisy-observation robustness after that

Wind should eventually be included directly in training, but not exposed explicitly in observations.

## Evaluation Strategy

Training should use the existing **train + watcher** workflow so checkpoint quality is monitored during long runs.

Evaluation should track:

- completion rate
- mean / p95 lateral error
- mean / p95 height error
- mean / p95 alignment error
- termination / crash rate
- cross-task aggregate score

Evaluation should be organized in this order:

1. no-wind truth-state missions
2. no-wind mixed mission suite
3. steady wind suite
4. OU / gust suite
5. later, estimated-state versions of the same suites

## Efficiency and Infrastructure

The implementation should keep using:

- RSL-RL PPO
- `train_and_watch.py`
- `watch_and_eval.py`
- dual-GPU split when available:
  - training on one GPU
  - checkpoint evaluation on the other GPU

Multi-environment scaling should be treated as part of the first implementation pass, not a later optimization.

## Near-Term Success Criteria

The first milestone is achieved when:

1. a teacher-guided PPO policy trains on the generic path-tracking env
2. the policy performs well on no-wind mixed missions
3. a teacher-off continuation run stays stable
4. the resulting pure-RL checkpoint remains competitive with the teacher baseline on the same suite

That milestone then becomes the launch point for wind robustness and later estimated-state robustness.
