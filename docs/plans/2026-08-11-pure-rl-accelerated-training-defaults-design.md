# PureRL Accelerated Training Defaults Design

Date: 2026-08-11
Status: Approved

## Objective

Promote the validated CPU-native PureRL training configuration to the default used by
`scripts/flapping_rl/train_and_watch.py` for all measured PureRL C1, C2a, C2b, and C2c tasks.

## Decision

When the user does not provide explicit overrides, a measured PureRL task uses:

- 256 training environments;
- 16 PPO minibatches;
- 500 PPO iterations;
- a 25-iteration checkpoint interval.

These values preserve the rollout samples, minibatch size, optimizer work, and checkpoint sample interval of
the previous 64-environment, 4-minibatch, 2,000-iteration, 100-iteration-save configuration. Explicit command-line
values retain precedence. Non-measured tasks retain their existing defaults.

## Scope

The change belongs in the project launcher rather than upstream IsaacLab runner configuration. The measured
environment configuration already declares 256 environments, while the launcher currently overrides it to 64.
Changing the launcher also permits task-aware iteration and save defaults without changing unrelated tasks.

No plant, action, observation, reward, termination, curriculum geometry, physics frequency, policy frequency, or
PPO loss coefficient changes.

## Validation

Contract tests must demonstrate the four measured-task defaults, explicit overrides, and unchanged non-measured
defaults. Existing launcher tests and Python compilation must pass. Documentation must stop presenting
64/4/2000/100 as the current measured PureRL launch configuration.
