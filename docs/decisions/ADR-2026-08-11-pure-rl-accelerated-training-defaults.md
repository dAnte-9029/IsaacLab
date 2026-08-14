# ADR-2026-08-11: PureRL accelerated training defaults

- Status: Accepted
- Date: 2026-08-11
- Scope: `train_and_watch.py` defaults for measured PureRL C1/C2/C3 tasks
- Extends: `ADR-2026-08-11-pure-rl-sample-equivalent-promotion-cadence.md`
- Supersedes: measured PureRL launcher defaults 64 environments, 4 minibatches, 2,000 iterations, and a
  100-iteration save interval
- Extended by: `ADR-2026-08-12-pure-rl-sequential-evaluation-default.md`

## Context

Equal-work CPU-native benchmarking measured 2.61 times the 64-environment throughput with 256 environments and
16 minibatches. A subsequent 80-iteration C2a validation processed 983,040 transitions and produced adjacent
`model_1550.pt` and `model_1575.pt` checkpoints that passed both the 80-case C2a grid and current C1 retention.

The project launcher still defaulted measured PureRL tasks to 64 environments and inherited four PPO
minibatches. Its 2,000-iteration and 100-iteration-save defaults were calibrated for that smaller rollout batch.
Changing only environment count and minibatches would quadruple the default sample budget and checkpoint sample
interval.

## Decision

For every measured PureRL C1, C2a, C2b, and C2c task launched through
`scripts/flapping_rl/train_and_watch.py`, omitted command-line options resolve to:

- 256 training environments;
- 16 PPO minibatches;
- 500 PPO iterations;
- a 25-iteration checkpoint interval.

Explicit command-line values retain precedence. Non-measured tasks retain 512 environments, 2,000 iterations,
a 100-iteration save interval, and their registered PPO minibatch configuration.

The project launcher is the source of these task-aware operational defaults. Upstream IsaacLab runner
configuration is not modified.

## Alternatives considered

- Keep the accelerated configuration opt-in. Rejected because the validated configuration would continue to be
  bypassed by ordinary measured PureRL launch commands.
- Change only environments and minibatches. Rejected because it would silently quadruple the default rollout
  sample budget and checkpoint spacing.
- Change the shared upstream RSL-RL runner configuration. Rejected because it would affect unrelated tasks and
  violate the project adapter boundary.

## Consequences

- A default measured PureRL run preserves the previous total transitions, PPO minibatch size, optimizer work,
  and checkpoint transition interval while reducing wall-clock training time.
- Historical 64-environment runs and explicit reproduction commands remain valid.
- Evaluation suite sizes remain 16/80/112/144 for C1/C2a/C2b/C2c and are independent of training environment
  count.
- Plant, action, observation, reward, termination, curriculum geometry, physics frequency, policy frequency, and
  PPO loss coefficients are unchanged.

## Assumptions

- Measured PureRL continues to use 48 rollout steps per environment and four PPO learning epochs.
- The validated host remains able to run 256 CPU-native environments without memory pressure.
- Training and checkpoint evaluation remain sequential rather than concurrent for authority runs.

## Validation requirements

- Launcher tests must cover measured defaults, explicit overrides, C2 inheritance, and unchanged non-measured
  defaults.
- The generated train command must contain 256 environments, 500 iterations, save interval 25, and 16
  minibatches when those options are omitted.
- Python compilation and `git diff --check` must pass.

## Reconsideration triggers

Reconsider these defaults if the rollout length or PPO epoch count changes, 256 environments no longer fit the
authority host, a larger CPU batch passes an equal-work convergence and retention gate, or a future authoritative
plant supports a qualified GPU training path.
