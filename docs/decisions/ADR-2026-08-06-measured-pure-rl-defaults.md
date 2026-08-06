# ADR: Make the accepted curriculum-1 contract the MeasuredPureRL default

- Status: Accepted
- Date: 2026-08-06
- Scope: MeasuredPureRL environment and project-local training launcher
- Supersedes in part: the Stage 0 decision to keep reset freeze as a command-line override

## Context

The direct action, normalized observation, curriculum-1 reward and termination
contracts have passed pure tests, a 64-environment native runtime gate, fixed
actions and exactly reproduced bounded random actions. Keeping required launch
properties as optional command-line flags now creates avoidable invalid modes:
the native constraint rejects direct-GPU PhysX, while the inherited 240-step
reset freeze consumes more than the current 48-step PPO rollout.

## Decision

`FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg` explicitly owns the
accepted curriculum-1 defaults:

- CPU native measured-wing multibody plant;
- 480 Hz physics and 60 Hz policy actions;
- direct `[frequency, rudder, left elevon, right elevon]` actions;
- 0--5 Hz frequency range and tail aerodynamics from actual joint angles;
- no extra action LPF or rate limiter;
- normalized 555-value actor observation with randomized route heading and
  flap phase;
- preview speed bounds 1--12 m/s;
- no wind, command randomization or teacher guidance;
- curriculum-1 reward and per-term telemetry;
- ground, 75-degree tilt, 3 m cross-track and 3 m height-error termination;
- zero reset-freeze steps.

The project-local `train_and_watch.py` recognizes the exact MeasuredPureRL task
ID. Without extra flags it selects CPU for PhysX and RSL-RL, loads the native
extension, uses isolated generated assets and defaults to 64 environments.
Explicit positive `--num-envs` and reset-freeze overrides remain available.

Other tasks preserve the launcher defaults of 512 environments and their
existing device behavior. Historical environment configurations preserve their
reward, observation, action and 240-step reset freeze.

## Alternatives considered

1. Keep requiring `--native-cpu --freeze-steps-after-reset 0 --num-envs 64`.
   This leaves the canonical task easy to launch in a known-invalid mode.
2. Change global launcher and base-environment defaults. This would silently
   affect controller, teacher and legacy PureRL tasks.
3. Add a new runner configuration under `source/isaaclab_tasks`. Repository
   boundaries prohibit modifying upstream task infrastructure without separate
   exact-file approval, and PPO hyperparameter redesign is not required for a
   launch smoke.

## Consequences

- The canonical MeasuredPureRL launch no longer needs native CPU or reset-freeze
  flags.
- The first PPO rollout contains physical transitions instead of only frozen
  reset frames.
- Existing RSL-RL network and PPO algorithm parameters remain the baseline and
  are not claimed optimal for this observation or reward.
- A short PPO smoke remains launch-only evidence, not convergence evidence.

## Validation requirements

- Static contracts must prove the MeasuredPureRL defaults and preservation of
  base-environment behavior.
- Launcher contracts must prove automatic CPU/native-extension selection,
  64-environment defaulting and unchanged behavior for other tasks.
- A fresh-process 64-environment PPO smoke must complete finite rollout and
  optimizer work, write a checkpoint and exercise automatic resets.

## Reconsideration triggers

Reconsider these defaults if the authoritative plant gains supported GPU
constraints, measured CPU throughput supports a different environment count,
the PPO rollout length changes enough to justify a reset freeze, or a separately
approved learning experiment establishes different runner parameters.
