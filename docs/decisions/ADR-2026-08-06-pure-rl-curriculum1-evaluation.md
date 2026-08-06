# ADR: Version the PureRL curriculum-1 evaluation and success gate

- Status: Accepted
- Date: 2026-08-06
- Scope: `Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-Direct-v0`
- Evaluation contract: `pure_rl_curriculum1_v1`
- Evaluation suite: `pure_rl_curriculum1_nowind_v1`

## Context

The inherited straight-flight watcher scores body-frame forward-speed error
against a command and world-frame lateral displacement. Those quantities do
not match curriculum 1: the policy should discover a stable forward speed, and
the route heading is randomized. The old watcher also accepted the first
episodes to finish across parallel environments, which overrepresents early
failures and does not guarantee coverage of initial conditions.

A short PPO launch smoke had already produced two finite checkpoints, but its
single-episode legacy watcher result could only establish checkpoint loading.
It could not decide whether a policy had learned stable straight flight.

## Decision

The canonical MeasuredPureRL task uses the versioned evaluation contract
`pure_rl_curriculum1_v1`. Legacy rows remain readable but cannot participate in
selection under this contract.

The default no-wind suite evaluates the Cartesian product of four route
headings and four initial flap phases:

```text
heading = {0, pi/2, pi, -pi/2} rad
phase   = {0, pi/2, pi, 3pi/2} rad
```

Sixteen environments receive one fixed pair each and contribute exactly one
episode. Per-environment quotas prevent fast-failure sampling bias. The
watcher checks the actual reset heading and phase against the registered grid
and fails closed on a mismatch.

Evaluation uses pre-reset environment caches so the terminal state is not
replaced by the automatic-reset state. It reports:

- episode duration, timeout and termination rates, including four cause rates;
- route-relative cross-track and height errors;
- along-track progress, along-track velocity and reverse-motion fraction;
- tilt and body angular rate;
- actual flap frequency, frequency-limit occupancy, tail-limit occupancy and
  normalized applied-action change.

There is no target forward-speed error. The diagnostic score is:

```text
100 * (
    0.40 * timeout_score
  + 0.20 * tanh(max(mean_progress, 0) / 12 m)
  + 0.15 * exp(-(mean_cross_track / 0.5 m)^2)
  + 0.15 * exp(-(mean_height_error / 0.5 m)^2)
  + 0.05 * exp(-(mean_max_tilt / 45 deg)^2)
  + 0.05 * authority_score
)
```

The score is diagnostic. Curriculum-1 success requires every registered gate:

| Metric | Threshold |
|---|---:|
| Timeout rate | `>= 0.80` |
| Termination rate | `<= 0.20` |
| Mean absolute cross-track error | `<= 0.50 m` |
| Mean absolute height error | `<= 0.50 m` |
| Mean along-track progress | `> 0 m` |
| Frequency-limit fraction | `<= 0.25` |
| Tail-limit fraction | `<= 0.10` |

Checkpoint selection first prefers gate passage, then timeout rate, episode
duration, termination rate, diagnostic score, progress and cross-track error.
Checkpoint index breaks an exact tie. A selected checkpoint that fails the
gate is only the best available fallback and is not a successful controller.

## Consequences

- Route rotation no longer changes metric meaning.
- Selection no longer imposes a desired forward speed.
- Every default evaluation covers the same registered initial-condition grid.
- Legacy and versioned results may coexist in one CSV without contaminating
  checkpoint selection.
- The aggregate grid is a curriculum gate, not evidence of convergence,
  robustness to wind, seed robustness or sim-to-real performance.

## Validation requirements

- Pure tests must cover quota allocation, reset-grid checking, metric
  aggregation, threshold boundaries, contract filtering and selection order.
- The CPU native runtime test must verify finite pre-reset evaluation caches.
- A formal checkpoint evaluation must run in a fresh Isaac Sim process and
  inspect the versioned CSV and JSON artifacts, not only process status.

## Reconsideration triggers

Reconsider the fixed grid or thresholds only after multi-seed training evidence
shows that the gate is either trivially satisfied or excludes visibly stable
straight flight. Add wind as a separately versioned curriculum suite rather
than silently changing this no-wind contract.
