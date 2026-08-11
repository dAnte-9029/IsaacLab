# PureRL Sample-Equivalent Promotion Design

Date: 2026-08-11
Status: Approved

## Problem

The longitudinal promotion contract currently uses PPO iteration numbers: evidence becomes eligible at iteration
200 and two passing checkpoints must be 100 iterations apart. Increasing the rollout from 64 to 256 environments
quadruples the samples per iteration, so retaining 200/100 would silently quadruple the sample threshold and
checkpoint spacing.

## Considered approaches

1. Keep 200/100 for every environment count. This preserves literal iteration semantics but makes promotion four
   times more expensive at 256 environments.
2. Change the global defaults to 50/25. This is rejected because it would reinterpret existing 64-environment
   evidence and permit promotion too early.
3. Preserve the 64-environment reference in samples and derive exact iteration values for each rollout size. This
   is approved because it preserves old behavior and gives 256 environments the sample-equivalent 50/25 schedule.

## Design

The reference contract is 64 environments, 48 steps per environment per PPO iteration, minimum iteration 200 and
evaluation interval 100. Therefore the frozen sample thresholds are:

- minimum evidence: 614,400 transitions;
- adjacent evidence interval: 307,200 transitions.

A small promotion-schedule builder derives iteration values from `num_envs` and `num_steps_per_env`. It fails
closed when either sample threshold cannot be represented by an exact integer iteration count. Existing
`evaluate_longitudinal_promotion()` defaults remain 200/100 so historical 64-environment evidence is unchanged.
Callers using another rollout size must pass the derived schedule explicitly.

The accepted curriculum ADR is not rewritten. A new ADR records the sample-equivalent interpretation and
supersedes only its iteration-cadence clause.

## Validation

Unit tests cover the 64/128/256 mappings, invalid/non-divisible configurations and promotion with the 256 schedule.
Runtime validation uses 256 CPU-native environments and 16 minibatches, saves every 25 iterations, and evaluates
saved checkpoints sequentially rather than during training. C1 retention and the target longitudinal grid remain
fail-closed promotion gates.
