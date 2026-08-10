# ADR: PureRL Frequency Governor and Curriculum Retention

- Status: Accepted
- Date: 2026-08-10
- Supersedes: the normalized frequency-action-delta term in `ADR-2026-08-06-pure-rl-curriculum1-reward.md`
- Extends: `ADR-2026-08-06-pure-rl-curriculum1-evaluation.md`

## Decision

The measured multibody PureRL task applies a frequency-only physical slew governor after mapping the policy command to hertz. The default rise and fall limits are both 2.0 Hz/s. The three tail-surface actions remain direct and continue to use actual joint angles for aerodynamic calculations.

The curriculum-1 reward penalizes the squared applied frequency slew divided by 2.0 Hz/s. It no longer penalizes normalized frequency-action delta. The evaluation contract is advanced to `pure_rl_curriculum1_v2` and the default suite to `pure_rl_curriculum1_nowind_v2`; the v1 suite name remains a deprecated case-grid alias, but new evaluations always emit the v2 contract. V1 and v2 results must not be mixed for checkpoint selection.

Every later primitive curriculum stage rehearses straight flight with a stage-specific probability of 0.50, 0.35, 0.25, and 0.20. A promoted checkpoint must be evaluated on every completed curriculum stage. The retention matrix fails closed on missing, duplicate, future-stage, or ambiguous-checkpoint records. Per-stage success gates determine retention; score drop relative to the diagonal checkpoint is recorded as a forgetting diagnostic and is not yet an additional gate.

## Rationale

The previous policy could move the frequency channel at policy rate and learned oscillations close to the wingbeat rate. Expressing both the actuator constraint and reward in physical units removes that unintended phase-modulation path while keeping the action interface interpretable. Stage-wise rehearsal reduces abrupt distribution replacement, and the retention matrix makes forgetting visible before a later-stage checkpoint is promoted.

## Consequences

Existing measured PureRL checkpoints are pre-governor artifacts and require retraining for v2 comparisons. The 2.0 Hz/s limit is a configurable engineering default, not an identified ES08MDII actuator constant. Real-flight data may later revise the value, but no actuator-identification claim is made here. Non-opt-in environments retain their previous action dynamics.
