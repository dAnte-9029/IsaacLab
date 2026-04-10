# Tail Balance Grid Search Design

**Date:** 2026-04-08

**Goal:** Build a controller-only, non-RL search workflow that tunes the tail pitch-authority balance for DeLaurier path tracking, with the explicit objective of improving `level_turn` and `level_loiter` altitude hold without regressing `level_straight`.

**Scope:**
- Search only:
  - `tail_fixed_horizontal_effectiveness`
  - `tail_elevon_effectiveness`
- Keep:
  - `DeLaurier` wing backend
  - `fly_path_mission.py` controller-only rollout path
  - no RL actor, reward, or training changes

## Evidence That Motivates This Search

The first controller defect has already been repaired: TECS now includes bank-aware pitch-side compensation. That changed controller intent but only slightly improved mission-level altitude hold.

The next dominant bottleneck is downstream in tail aero balance:
- action mapping is not swallowing the pitch command
- elevon alpha clipping is not the first limiter
- the default fixed-horizontal tail torque and elevon torque nearly cancel during bank entry

Sensitivity probes already show both of these changes help:
- raising `tail_elevon_effectiveness`
- lowering `tail_fixed_horizontal_effectiveness`

The remaining task is no longer bug isolation. It is controlled parameter search under hard regression guards.

## Design Choice

### Option A: Two-stage coarse-to-fine grid search over tail balance only

This is the chosen design.

Why:
- only two parameters are in scope
- the response surface matters more than a single black-box optimum
- hard constraints on `straight` are easier to reason about with an explicit grid
- results stay reproducible and easy to compare later

### Option B: Random/LHS search over the same two parameters

Rejected for the first implementation.

Why:
- less interpretable than a grid for a 2-D problem
- still needs a second pass to understand tradeoffs
- no practical advantage at this search size

### Option C: Bayesian optimization

Rejected.

Why:
- over-designed for a 2-parameter search with explicit hard constraints
- harder to debug when the score function changes

## Search Objective

The search is `balanced`, not turn-only.

That means:
- `level_turn` and `level_loiter` should improve materially
- `level_straight` must not regress beyond a strict guardrail
- random seeded missions must not collapse mission completion

## Evaluation Set

Each candidate is evaluated on:

### Canonical missions
- `level_straight`, `2200` steps
- `level_turn`, `2400` steps
- `level_loiter`, `2800` steps

### Random seeded missions
- same `fly_path_mission.py` controller-only rollout path
- `mission_mode=random`
- `2600` steps
- fixed no-wind seeds:
  - `11`
  - `23`
  - `37`
  - `53`
  - `71`
  - `89`

All runs must stay on the non-RL, DeLaurier, teacher-controller path.

## Search Space

### Stage 0 baseline
- current repo defaults

### Stage 1 coarse grid
- `tail_fixed_horizontal_effectiveness ∈ {0.60, 0.70, 0.80, 0.90, 1.00}`
- `tail_elevon_effectiveness ∈ {1.00, 1.20, 1.40, 1.60, 1.80}`

### Stage 2 local refinement
- keep the top 4 feasible candidates from Stage 1
- refine around each with:
  - `tail_fixed_horizontal_effectiveness ± 0.05`
  - `tail_elevon_effectiveness ± 0.10`
- clamp back into a sane search interval
- de-duplicate overlap before running

## Hard Constraints

A candidate is infeasible if any of these fail:
- `level_straight.completed_path` is worse than baseline
- `level_straight.mean_abs_height_error_post_warmup_m > 1.05 * baseline`
- random-mission completion rate drops by more than `5%` relative to baseline

Hard constraints apply before ranking.

## Scoring

Feasible candidates are ranked by a weighted score:

- `0.35 * turn_height_gain`
- `0.35 * loiter_height_gain`
- `0.15 * random_height_gain_avg`
- `0.10 * random_progress_gain_avg`
- `0.05 * straight_margin`
- `-0.05 * saturation_penalty`

Where:
- `turn_height_gain` compares candidate `level_turn` post-warmup mean absolute height error against baseline
- `loiter_height_gain` compares candidate `level_loiter` the same way
- `random_height_gain_avg` averages seeded random-mission height improvement
- `random_progress_gain_avg` averages seeded random-mission final progress improvement
- `straight_margin` rewards staying comfortably inside the straight hard constraint
- `saturation_penalty` lightly penalizes pathological solutions that win only by pinning controls

## Saturation Metrics

The search should not require new rollout instrumentation if existing trajectory fields are sufficient.

Use trajectory-derived fractions:
- `freq_sat_frac = fraction(freq_hz >= 4.9)`
- `pitch_sat_frac = fraction(action_elevon_pitch <= -0.98)`

These do not hard-reject a candidate. They only provide a light penalty and post-hoc interpretability.

## Outputs

Each search run should produce:
- `manifest.json`
- `leaderboard.csv`
- `per_run_metrics.csv`
- `response_surface.csv`
- `summary.json`
- `summary.md`

Expected content:
- `per_run_metrics.csv`: one row per rollout
- `leaderboard.csv`: one row per parameter pair with feasibility and score
- `response_surface.csv`: parameter grid plus key canonical metrics for plotting or pivoting
- `summary.md`: brief human-readable report with baseline, best candidate, runners-up, and constraint decisions

## Execution Model

Default execution is sequential for reproducibility and GPU safety.

The runner may expose an optional bounded worker count for subprocess fan-out, but `1` should remain the default.

## Promotion Rule

The first search pass should not immediately rewrite repo defaults.

Promotion process:
- choose `candidate_best` from the main search
- run a confirmatory holdout on 6 extra random seeds
- only then change default tail balance values in repo config

## Non-Goals

- no RL changes
- no reward tuning
- no controller gain retuning beyond what is already fixed
- no wind robustness search in the first version
- no broad multi-parameter optimization

## Minimal Implementation Strategy

Implement a new search runner under `scripts/flapping_px4/` that:
- launches `fly_path_mission.py` directly for canonical and random cases
- collects `summary.json` and `trajectory_env0.csv`
- computes hard constraints and score
- writes aggregate search artifacts

This keeps rollout logic in one place and avoids modifying the validated controller-only mission path unless the search truly needs more fields.
