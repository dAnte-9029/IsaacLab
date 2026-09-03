# ADR: Target the rare full-C3 joint failure slices

- Date: 2026-09-02
- Status: accepted diagnostic experiment
- Scope: measured CPU-native PureRL full-C3 joint training

## Context

The completed seed-0 full-C3 joint run produced one checkpoint, `model_175.pt`, that passed C3c,
C3b, C3a, C2c and C1. No adjacent checkpoint passed every suite, so the run is not promotable.
The neighboring checkpoints retained C1, C2c and C3a but failed repeatable C3b and C3c slices.
C3b failures concentrated in templates 5 and 7 with strong positive climb, especially right turns.
C3c failures concentrated in high-demand positive-climb coupled maneuvers. A corrected 24-second
diagnostic showed that earlier candidates could complete those C3c paths after the unchanged
20-second authority horizon, while `model_175.pt` completed them within that horizon. The gate is
therefore learnable and remains unchanged.

The fixed top-level mixture assigns 25 percent of resets to C3b and 25 percent to C3c, but the
specific failed C3b and C3c combinations each occupy less than one percent of all resets. This
makes the observed checkpoint instability consistent with rare-slice dilution. It does not justify
a reward, actor, horizon, learning-rate or global task-mixture change.

## Decision

Add default-disabled stratified sampling inside the existing C3joint C3b and C3c families. The
top-level `15/20/15/25/25` C1/C2c/C3a/C3b/C3c distribution and the five-group task-aware PPO
objective remain fixed.

Within C3b, retain the original sampler for 50 percent of C3b rows. Allocate the other 50 percent
equally to template 5 (turn then climb) and template 7 (climb then turn), always with a continuous
10-to-12-degree positive climb. Sample the two templates and both turn signs independently with
equal probability. This remains balanced in distribution when asynchronous partial resets contain
only one path.

Within C3c, retain the original continuous sampler for 60 percent of C3c rows. Allocate 40 percent
to positive-climb coupled maneuvers sampled continuously at normalized elliptical demand 0.8 to 1.0,
with 6-to-8-degree roll magnitude and balanced turn signs. This targets the failed neighborhood and
does not copy the discrete frozen evaluation grid.

Each sampled path carries a `hard_slice_id`. Runtime telemetry reports total and sign-specific C3b
and C3c hard-slice fractions. Frozen evaluation overrides clear this diagnostic label, so training
coverage metadata cannot be mistaken for evaluation evidence.

The first experiment is a bounded same-stage diagnostic: resume the full-joint seed-0
`model_175.pt` together with its optimizer state and train 25 further PPO updates, materializing
`model_200.pt`. It retains the split actor, heading-canonical observation, governor-gap reward,
learning-rate state, CPU-native plant, 256 environments, 16 minibatches and seed 0. This continuation
tests local checkpoint stability; it is not authority evidence for a fresh recipe.

## Alternatives considered

- Changing the global five-family mixture was rejected because C1, C2c and C3a retention already
  passed and the observed failures occupied narrow within-family slices.
- Changing the reward, actor, learning rate or C3c horizon was rejected for this diagnostic because
  it would confound whether additional exposure to the rare slices was sufficient.
- Deterministically cycling templates and signs inside each sampler call was rejected after runtime
  evidence showed that asynchronous single-path resets restart the local rank and collapse coverage
  to the first template and sign.

## Assumptions

- The narrow failure slices are underrepresented enough that added on-policy exposure can improve
  local checkpoint stability without changing the global objective.
- Continuous sampling around the failed severity range is training coverage, not leakage from the
  discrete frozen evaluation grid.
- The unchanged 20-second C3c gate remains achievable because `model_175.pt` already passed it.

## Consequences and validation

Existing C3a, C3b, C3c and baseline C3joint sampling remain unchanged unless the explicit
`--c3-full-joint-hard-slice-continuation` route is selected. Promotion gates and the 20-second C3c
horizon remain unchanged. The diagnostic is successful only if fresh-process frozen suites show
that the continued checkpoint retains all five abilities; one pass still cannot satisfy the adjacent
checkpoint promotion rule.

If the bounded continuation is useful, the next authority experiment must restart from promoted
C2c with the hard-slice sampler enabled and must obtain an adjacent all-suite passing pair. If it
damages broad C3b/C3c coverage, reduce the within-family targeted fractions before changing the
global mixture or optimization algorithm.

Reconsider this decision if balanced asynchronous sampling still fails to stabilize neighboring
checkpoints, if broad C3b/C3c slices regress, or if the next task changes from event-sequence
coverage to continuously curved three-dimensional paths. The completed bounded-run evidence and
the pre-fix sampling limitation are recorded in
`docs/audits/2026-09-03-pure-rl-c3joint-hard-slice-continuation.md`.
