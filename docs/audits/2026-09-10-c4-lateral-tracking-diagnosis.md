# C4 lateral tracking diagnosis

## Scope and authority

Read-only analysis of the original full-joint `model_175.pt` and one fresh
24-environment CPU-native diagnostic ablation. No policy training, plant,
reward, observation, actor or frozen-grid changes. Worktree:
`.worktrees/native-multibody-rl`, base `41ea8565`. This document is an audit,
not a promotion or a new evaluation contract.

Evidence directories (ignored generated artifacts):

- `logs/continuous_curve/20260909_zero_shot_175_retry/`: original 96-case
  summary, replay recipes and first-episode traces.
- `logs/continuous_curve/20260909_c4_matched_diagnostic/`: 24-case diagnostic
  summary, replay recipes, checkpoint provenance and traces.
- Original run's `diagnosis/`: matched input recipes, offline analysis scripts,
  console reports and `lateral_response.png`.

## Controlled result

Three groups each contain four horizontal/vertical mirror combinations and two
flap phases, with global heading fixed at zero. Wavelength, second harmonic,
entry, checkpoint, plant, observation and episode duration remain fixed.
The baseline quadrature slope amplitude is 5 degrees.

| Group | Single change from quadrature | Success | Mean absolute lateral error |
| --- | --- | --- | --- |
| qbase | none; heading-rate amplitude 0.011 rad/m | 0/8 | 0.548352 m |
| qsmall | heading-rate amplitude reduced to 0.009 rad/m | 8/8 | 0.453704 m |
| qphase0 | slope phase changed from pi/2 to zero | 4/8 | 0.538559 m |

All 24 episodes are finite, survive and complete 140 m. Baseline reproduction
against the matching original cases differs by at most 0.000041 m in mean
lateral error. Reducing amplitude by 18.18% reduces error by 17.26%; changing
phase reduces the group mean by only 1.79%, while redistributing errors across
mirrors (phase-zero errors span approximately 0.450--0.664 m).

The controlled evidence supports amplitude as the main driver of the overall
quadrature accuracy failure at this operating point. Coupling phase affects
individual cases; it is not irrelevant. These eight paired cases do not prove
this ordering for every curve or demand level. The diagnostic grid is separate
from, and does not relax, the original frozen gate.

## Observed closed-loop mechanism

Original 96 traces show recurrent outward displacement rather than a constant
world-direction bias. After 20 m entry, restricting to |dpsi/ds| > 0.003 rad/m,
the fraction with lateral_error * heading_rate < 0 is 96.06% for quadrature,
96.81% for coupled and 100% for biased curves (means of per-case fractions).
The lateral normal is defined in `pure_rl_spatial_path.py::query_spatial_path`;
these signs indicate the outside of either turn direction.

For the four yaw-zero/phase-zero quadrature mirrors, signed error and curvature
have correlations from -0.814 to -0.937 over t=3--18 s. Course-vs-reference
heading error RMS is approximately 2.72--3.35 degrees. Errors recur well after
entry and through the 140 m completion point. For example,
`quadrature_h+1_v+1_yaw0_phase0` has lateral error -0.712 m at 3.017 s,
+0.729 m at 9.017 s, and -0.805 m at 13.017 s as turning direction changes.
Biased curves also sustain outward offsets around 0.35--0.40 m despite passing.
Thus passing the mild curve does not mean zero-bias tracking.

A descriptive lagged least-squares fit of actual horizontal course rate to
reference `k * tangent_speed` on t=3--18 s gives best lags 0.10--0.45 s across
these four quadrature mirrors. Actual course rate uses a 61-sample, degree-3
Savitzky--Golay derivative; fits allow gain and offset. Fitted gains are about
1.43--1.46: this is lagged corrective over-response, not uniform turn-rate
attenuation. Periodicity, smoothing and closed-loop feedback prevent treating
this as an isolated neural-policy latency measurement.

## Actor input and actuator checks

Reconstructed previews from saved pose, replayed path and the existing 0.12--
0.60 s query agree with stored actor input to 2.87e-6 normalized units on the
sampled yaw-zero/phase-zero traces. Maximum per-step progress change over all
96 traces is 0.1874 m. No large projection jump or preview reconstruction bug
was observed; these checks do not prove every possible projection is safe.

Offline strict actor replay reproduces saved raw actions to 6e-8. On sampled
quadrature states (3--18 s, every sixth policy step), removing just the horizontal
curved deviation from the tangent-line preview changes normalized tail outputs
by RMS 0.023--0.028. Removing the current lateral offset from all five preview
points changes them by RMS 0.227--0.257. State/action history is held fixed.
The future curvature contributes only about 0.090--0.108 m RMS lateral deviation
at the furthest point; normalization divides preview coordinates by 5 m.

These are unequal-size, deliberately counterfactual input perturbations, not
closed-loop trials or per-unit sensitivity measurements. They establish that
the actor uses curvature information but that accumulated offset has a much
larger action contribution at the observed states. They support insufficient
anticipatory compensation as an interpretation, not proof that extending the
preview would fix the problem or that the actor ignores curvature.

Command-to-actual tail traces show a best alignment lag of roughly two policy
steps (0.033 s) on a 60 Hz grid for all three surfaces across all 12 sampled
family/mirror cases. Considerable instantaneous tracking differences exist,
but similar differences also occur on passing biased paths. The saved actual
joint debug values precede the final physics substep (approximately 2.08 ms);
this limits timestamp precision. These data do not identify servo dynamics as
the principal C4-specific cause, nor certify unlimited actuator authority.

## Interpretation and next action

The established failure is finite-accuracy continuous-turn tracking: this
checkpoint follows a recurring outside-offset trajectory, with phase-shifted
course corrections. Increasing continuous curvature raises that offset beyond
the 0.50 m mean-error gate. The policy has never been trained on C4. A plausible
controller-level explanation is inadequate learned anticipation/compensation
for continuously varying curvature; the present evidence cannot uniquely
separate neural representation, learned feedback gains and vehicle dynamics.

The existing training reward uses `cross_track_scale_m=1.5` in
`direct/flapping_bot/pure_rl_reward.py`, whereas evaluation requires mean error
<=0.50 m. That scale makes offsets relatively inexpensive, but does not prove
reward optimality or causation. No reward change during zero-shot could itself
change deterministic inference actions.

The supported next step is controlled C4 learning from the passing 0.009 demand
region toward 0.011, preserving the original frozen targets and old-task
retention. Do not attribute all quadrature failures to slope phase, change the
plant, or declare the 0.6 s preview insufficient based on these results alone.
A preview/feedback intervention would need a separate matched closed-loop test.

## Authorized preview-only A/B follow-up

The user authorized a zero-shot horizon comparison without monitoring. The
checkpoint evaluator now accepts `--preview-time-scale` (default 1). Scale 2
replaces only the five future sample times with 0.24/0.48/0.72/0.96/1.20 s;
scale 1 keeps the original query unchanged. Projection state, route frame,
rewards, termination, normalization, actor shape and weights are preserved.
The long-preview adapter returns a replacement query without mutating the base
cache, so repeated calls cannot compound the time scale.

Launcher: `logs/continuous_curve/20260910_preview_ab/run.sh`.
It runs long-preview six-env/eight-step smoke, then fresh short and long
96-case evaluations sequentially, checking durable completion markers because
the repository wrapper does not propagate child Python failures. Three focused
scoring/interpolation tests pass; runtime smoke is a prerequisite inside the
background pipeline. Results have not yet been inspected. Each arm contains
its own provenance, identical curve recipes, summary and traces. Five-point
sampling remains the same shape but changes input distribution; a negative
zero-shot result does not prove that longer preview cannot help after training.
