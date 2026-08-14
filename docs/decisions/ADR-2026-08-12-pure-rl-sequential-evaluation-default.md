# ADR-2026-08-12: PureRL sequential evaluation default

- Status: Accepted
- Date: 2026-08-12
- Scope: measured PureRL C1/C2/C3 launcher evaluation process policy
- Extends: `ADR-2026-08-11-pure-rl-accelerated-training-defaults.md`

## Context

The validated CPU-native throughput measurements were collected in train-only mode, and the acceleration audit
requires checkpoint evaluation to run sequentially after training. The launcher nevertheless still started a
checkpoint watcher by default. On the authority host, the watcher created a second CPU-native simulator process
that competed with PPO training for the same CPU resources and increased iteration time substantially.

## Decision

Measured PureRL C1, C2, and C3 tasks launched through `scripts/flapping_rl/train_and_watch.py` run without a
concurrent checkpoint watcher or final one-shot evaluation by default. Their checkpoints are evaluated later in
fresh, sequential processes.

`--concurrent-eval` explicitly restores the prior train-and-watch behavior for a deliberate experiment.
`--train-only` remains supported as an explicit compatibility and authority flag. Supplying both flags is an
error. Non-measured tasks retain their existing concurrent-evaluation default.

Authority reproduction commands include `--train-only` even though it matches the measured-task default. This
makes the process-isolation requirement visible and protects the command from future default changes.

## Alternatives considered

- Keep concurrent evaluation as the measured-task default. Rejected because it does not reproduce the validated
  training configuration and causes material CPU contention on the authority host.
- Disable concurrent evaluation for every launcher task. Rejected because no equivalent throughput evidence was
  collected for unrelated tasks and their established default should remain available.
- Remove the watcher. Rejected because explicit concurrent evaluation remains useful when independent compute
  resources are intentionally assigned.

## Consequences

- Ordinary measured PureRL launches use the validated 256-environment, 16-minibatch train-only configuration.
- Checkpoint files continue to be saved every 25 iterations, but evaluation and promotion evidence must be
  generated after training in fresh processes.
- Existing automation can keep using `--train-only`; deliberate overlap must add `--concurrent-eval`.
- This decision changes launcher orchestration only. Plant, task, PPO, reward, termination, and checkpoint formats
  are unchanged.

## Assumptions

- CPU-native training and evaluation contend for the same authority-host CPU resources when they overlap.
- Sequential evaluation remains acceptable because promotion already requires complete stage and retention
  evidence rather than watcher output alone.

## Validation requirements

- Tests must show measured PureRL tasks default to no watcher.
- Tests must show `--concurrent-eval` enables the watcher, `--train-only` disables it, and both flags fail closed.
- Tests must show non-measured task defaults are unchanged.
- Python compilation and `git diff --check` must pass.

## Reconsideration triggers

Reconsider the default if training and evaluation are assigned independently qualified compute resources, or if
a benchmark demonstrates that concurrent evaluation does not reduce authoritative training throughput.
