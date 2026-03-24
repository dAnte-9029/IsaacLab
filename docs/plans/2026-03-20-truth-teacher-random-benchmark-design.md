# Truth Teacher Random Benchmark Design

## Goal

Add a truth-state random-mission benchmark for the PX4-like path-tracking teacher so we can answer one concrete question with data: whether the current teacher is strong enough to act as both a baseline and a reliable guide for RL on the current mission family.

## Scope

This design covers:

- random mission sampling for the current primitive mission family
- truth-teacher rollout execution on sampled missions
- per-episode metrics and aggregate summary
- a lightweight batch runner for repeated seeds

This design does not cover:

- switching the teacher to estimated/noisy observations
- changing the teacher control law itself unless the benchmark reveals a clear bug
- changing the RL environment distribution beyond current mission primitives

## Current State

The repository already has:

- a generic path-tracking environment with random mission sampling
- a truth-state teacher controller path through `FlappingBotPathTrackingEnv`
- a deterministic mission rollout script for canonical cases
- a deterministic baseline suite for `straight`, `turn`, `loiter`, and one fixed multi-segment mission

What is still missing is a randomized benchmark over many sampled missions. Right now we have evidence on representative fixed cases, but not on mission-distribution performance.

## Recommended Approach

### Option A: Add random-mission mode to `fly_path_mission.py` and build a batch runner on top

This is the recommended approach.

`fly_path_mission.py` already contains the teacher rollout logic, logging, completion handling, and plotting-compatible output format. We extend it so that it can either inject a fixed canonical mission or inject a sampled random mission from a given seed. Then we add a small batch runner that repeatedly calls it across seeds, collects `summary.json` files, and writes an aggregate benchmark report.

Why this is the best option:

- reuses the existing validated rollout path
- keeps deterministic and random evaluation in the same artifact format
- minimizes duplicated control/evaluation logic
- makes later noisy/estimated baseline comparisons easier because they can reuse the same runner shape

### Option B: Write a completely separate random benchmark script

This would avoid touching the current deterministic script, but it would duplicate rollout bookkeeping, mission injection, completion handling, metrics, and artifact writing. That duplication would quickly drift.

### Option C: Evaluate random missions only inside the RL env / watcher stack

This is too indirect for the current goal. We want a teacher-only benchmark first, not a checkpoint-evaluation path tied to PPO.

## Benchmark Definition

Each rollout uses:

- truth position, velocity, attitude, body rates
- truth wind if enabled
- truth path query from the current random mission

Each episode logs:

- sampled mission segments
- completion flag
- failure kind if any
- final progress ratio
- mean / p95 lateral error
- mean / p95 height error
- mean / p95 alignment error
- mean speed

The batch summary reports:

- number of episodes
- completion rate
- mean and p95 of the per-episode metrics
- worst seeds / missions for diagnosis

## Mission Distribution

The first random benchmark uses the current environment family:

- 2 to 4 segments
- primitives from `straight`, `turn`, `loiter`
- climb/descent only on straight segments

This matches the current RL environment distribution and answers the exact question we need right now: whether the teacher is good enough on the distribution that RL will initially see.

## Success Criteria

The benchmark is successful if:

- it runs repeatably from a seed range
- it produces aggregate metrics and per-episode artifacts
- it lets us identify whether the teacher is already adequate or whether specific mission patterns (likely turn-heavy patterns) still need work

## Next Step

Implement the random benchmark with test-first helper coverage, then run a first benchmark pass and use the results to decide whether the next task is teacher tuning or RL warm-start work.
