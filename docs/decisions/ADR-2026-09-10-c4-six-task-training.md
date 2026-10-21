# ADR: Six-task C4 random-curve training

Status: user-authorized implementation; formal training not launched.

## Context

The original full-joint `model_175.pt` passes fixed continuous-curve and independent
seed-104729 random-96 evaluations with 1.2 s preview (random mean lateral error
0.2574 m). The engineering C4 environment and sampled kinematic rejection sampler
are available. This stage implements training infrastructure, not evidence that
additional training is necessary or that a new checkpoint is promoted.

## Decision

Use `FlappingBotStraightFlightDeLaurierMeasuredPureRLC4JointEnvCfg`, instantiated by
`scripts/flapping_px4/train_continuous_curve.py`. The existing C4 replay task remains
separate. No upstream framework or existing generic launcher is modified.

- Six fresh on-policy task fractions: C1/C2c/C3a/C3b/C3c/C4 =
  0.09/0.12/0.09/0.15/0.15/0.40. Sampling and PPO target weights share the same config.
- `c4joint` reuses the existing `c3joint` sampler with conditional old weights,
  then independently replaces selected rows with screened random C4 curves.
  Old event generators and old-task relative proportions are preserved.
- Physical family IDs are 0..5. PPO task/phase IDs remain the old 0..6 plus
  **C4 = 7**; family 5 must not be confused with PPO group 5 (C3b).
- C4 rows receive 0.24/0.48/0.72/0.96/1.20 s preview; old rows retain their
  original preview, including C2's existing query override. Actor/critic inputs
  remain the 555 policy values; task labels are rollout metadata only.
- Extend the existing task-aware advantage normalization and stratified batches
  to six groups. Insufficient group samples still fail closed. Phase minima are
  disabled for this mixed run; critic loss semantics are preserved.
- Keep split actor, fixed observation normalization, governor-gap reward and
  bounded warm-start guard. Adaptive sampling, hard slices, distillation and yaw
  consistency remain disabled. No plant, reward or actor-capacity change.

## Initialization and resume

`--weights` loads the original full-joint model strictly, with an empty optimizer
and learning iteration zero. Source actor/heading metadata is checked. Runner
configuration is deep-copied before RSL-RL construction because that constructor
mutates its input dictionaries; the complete source architecture must survive in
`params/agent.yaml` for inference and resume.

`--resume` requires a C4 checkpoint with matching seed, environment count, rollout
length, minibatches, weights, sampler and preview contract. Restore model,
optimizer moments, optimizer learning rate, warm-start counter, task adapter
counter and Torch RNG state. Resume starts at the number of completed iterations,
not the saved checkpoint's zero-based loop index. Simulation starts fresh episodes;
this is optimizer/training-progress continuation, not bitwise PhysX/episode replay.

Default run budget is 256 CPU-native environments, 48 policy steps per rollout,
16 minibatches, 201 iterations, checkpoint every 25. These are explicit initial
experiment settings, not proven optimal hyperparameters. `--smoke` uses 96 envs,
16 steps, four minibatches and five iterations (one additional iteration on resume).

## Observability and evaluation

Per-task TensorBoard logs report counts, lateral/vertical error, tangent speed,
frequency request/application gap, tail limit and current completion. Episode CSV
records first terminal state task, duration, survival, ever-reached completion and
progress before reset. Task-aware CSV records sample counts, advantage statistics
and effective weights. Sampling counters record candidates, rejections, acceptances
and exhausted rows. Training summaries explicitly reject promotion claims.

`evaluate_c4_training.py freeze` creates independent development and holdout recipe
files (seeds 130363 and 196613), each 96 paths by default, with randomized initial
flap phases. Candidate distribution is the documented mild C4 distribution; these
sets do not establish extrapolation to a wider physical envelope. Train never reads
them. Repeated output-directory creation is rejected.

`evaluate_c4_training.py evaluate` uses explicit curves and checkpoints, evaluates
C4 with 1.2 s preview, then invokes the existing frozen C3c/C3b/C3a/C2c/C1 evaluator
in fresh CPU-native processes with original observation semantics. Run retention
even if C4 fails. Source JSON, not wrapper exit status, determines the result.
C4 checks full completion, survival/completion >=.95, success >=.90, means <=.50 m,
mean of per-case P95 <=1.50 m, reverse fraction <=.01, mean of per-case roll P95
<=25 degrees, no roll-limit termination and reported slice success >=.90.
These are explicit case-balanced summaries, not pooled quantiles. A single
checkpoint's combined pass is not promotion. Candidate selection uses development;
final holdout and adjacent-checkpoint evidence remain separate validation work.

## Validation and limits

- 117 pure/config tests passed after extending old AST expectations to permit C4.
- New mixed CPU-native runtime test passed: all six families, correct PPO IDs,
  short/long row preview and partial reset isolation.
- Existing C3 geometry/reward/reset native test passed (two unrelated tests deselected).
- Initial five-iteration smoke performed actual updates after burn-in, with actor
  delta norm 0.015755 and 19 optimizer state entries. Fresh resume advanced counters
  5 to 6, actor delta 0.009122, Adam steps 8 to 12, preserving the learning-rate ramp.
- First resume smoke identified the mutated-config class-name omission; fixed with
  a deep copy and verified using new smoke artifacts rather than editing old ones.
- Existing Gym/Isaac shutdown warnings remain. No formal 201-iteration run, full
  checkpoint retention sweep, holdout scoring or promotion was performed here.

Reconsider task fractions or complexity only using development evidence while
preserving old gates. Long-preview frequency-governor gaps remain a diagnostic to
track, not a reason to change actuator physics to make the task easier.
