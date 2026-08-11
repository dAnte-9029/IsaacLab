# ADR-2026-08-11: PureRL sample-equivalent promotion cadence

- Status: Accepted
- Date: 2026-08-11
- Scope: PureRL longitudinal checkpoint eligibility and adjacency
- Extends: `ADR-2026-08-10-pure-rl-curriculum-domain-contract.md`
- Supersedes: fixed 200/100 PPO-iteration cadence when rollout environment count differs from 64

## Context

The original longitudinal promotion procedure made checkpoints eligible after PPO iteration 200 and required two
passing checkpoints 100 iterations apart. Those values were established with 64 environments and 48 rollout
steps per environment. CPU-native throughput testing selected 256 environments as a validation candidate. Since
one 256-environment iteration contains four times as many transitions, retaining the literal 200/100 values would
silently quadruple the evidence threshold and spacing.

Changing the global defaults to 50/25 would create the opposite error: historical 64-environment evidence would
become eligible with one quarter of the intended samples.

## Decision

Promotion cadence is defined by samples, using the existing 64-environment procedure as the reference:

- minimum evidence threshold: `64 * 48 * 200 = 614,400` transitions;
- adjacent evidence interval: `64 * 48 * 100 = 307,200` transitions.

For a run with `N` environments and `S` steps per environment per PPO iteration, tooling derives exact iteration
values by dividing both transition thresholds by `N * S`. The division must be exact; otherwise promotion fails
closed and the rollout configuration requires a separate approved cadence.

The accepted mappings for the current 48-step rollout are:

| Environments | Minimum iteration | Evaluation interval |
| ---: | ---: | ---: |
| 64 | 200 | 100 |
| 128 | 100 | 50 |
| 256 | 50 | 25 |

Existing `evaluate_longitudinal_promotion()` defaults remain 200/100 for backward compatibility. Non-64 runs
must explicitly use the schedule returned by `build_sample_equivalent_promotion_schedule()`.

## Consequences

- Existing C2a promotion evidence is unchanged.
- A 256-environment run saves promotion evidence every 25 iterations and becomes eligible at iteration 50.
- Total rollout samples, PPO minibatch size and optimizer work can be compared without conflating environment
  count with evidence maturity.
- C1/C2 retention metrics, current-stage grids, two-adjacent-pass requirement and fail-closed behavior are unchanged.
- This decision does not change the plant, action, observation, reward, termination, curriculum geometry,
  physics rate, policy rate or PPO loss.

## Validation

Unit tests cover the accepted mappings, invalid inputs, non-exact division and a 256-environment 50/25 promotion
pair. Runtime validation resumed C2a at 256 environments for 80 iterations and evaluated `model_1550.pt` and
`model_1575.pt` in fresh C2a and C1 processes. Both checkpoints passed the 80-case C2a grid and C1 retention;
their 25-iteration separation is 307,200 transitions. The promotion builder selected `model_1575.pt`.

Evidence: `/home/zn/temp/pure_rl_cpu_native_256_validation_20260811/promotion_evidence.json`.
