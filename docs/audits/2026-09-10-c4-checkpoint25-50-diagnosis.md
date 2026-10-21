# C4 checkpoint 25 to 50 paired trace diagnosis

## Scope and evidence

Read-only analysis of existing deterministic CPU-native development evaluations;
no training, simulation rerun, holdout scoring, or controller/plant changes.
Worktree base: `41ea8565`, existing implementation changes uncommitted.
The training handoff's “formal training not run” statement is historical and stale.

Inputs under `logs/continuous_curve/`:
- `20260910_c4_early_diagnosis/25/`
- `20260910_c4_development_checkpoints/50/`

All 96 recipe records match exactly; trace values used are finite. Initial policy
inputs match exactly for every case. Both runs use 1.2 s preview. Trace semantics
come from `analysis/continuous_curve_evaluation.py::evaluate_checkpoint`: policy
input/raw action precede the recorded state by one 1/60 s policy step. Executed
surface diagnostics are the final physics substep, not direct torque measurements.
`straight_flight_env.py` records tail-limit activity from normalized commanded
surface magnitude versus the configured reward threshold; it is not a measured
mechanical stop or evidence of insufficient aerodynamic authority.

Reproduction: activate `env_isaaclab`, set `TERM=xterm`, then run
`./isaaclab.sh -p logs/continuous_curve/20260910_c4_25_50_diagnosis/analyze.py`.
The saved `comparison.json` contains time and progress bins and onset definitions;
`signed_bias.json` contains additional signed-error/initial-observation checks.
Analysis completed locally without a long-running job. Runtime assertions checked
96 matched cases, recipe equality and finite analyzed arrays. No software behavior
changed, so no new unit or native simulation tests were run.

## Observed facts

- Success drops from 96/96 to 5/96; both survive and complete all 96 paths.
- At 1--3 s, mean absolute lateral error increases from 0.076 to 0.358 m,
  while tail-limit fractions are 0.330% and 0.017%, respectively.
- After 3 s, case-averaged signed lateral error is -0.035 m for checkpoint 25
  versus +0.660 m for checkpoint 50. All 96 checkpoint-50 cases have positive
  time-mean signed error, versus 23 checkpoint-25 cases.
- Executed rudder changes in the negative direction in all 96 cases; mean
  difference after 3 s is -8.375 degrees. This is a comparison of evolved closed
  loops and does not isolate direct policy bias from subsequent compensation.
- On identical initial inputs, mean raw-action changes (channels 0..3) are
  [-0.10875, -0.15010, -0.22493, -0.13749]. Policy output changes precede feedback
  state divergence; this rules out different initial observations as its source.
- Median first sustained differences (12 consecutive samples, 0.2 s) are:
  requested frequency >0.1 Hz at 0.192 s; rudder >1 degree at 0.417 s;
  applied frequency >0.1 Hz at 0.433 s; signed lateral error difference >0.1 m
  at 1.167 s. Thresholds differ in units and do not rank causal importance.
- First-second mean requested frequency changes 3.101 to 2.874 Hz; applied
  frequency changes 3.412 to 3.240 Hz. At 10--20 s checkpoint 50 instead has
  higher applied frequency (2.977 versus 2.889 Hz), while tangent speed is lower
  (7.822 versus 8.930 m/s). Persistent drift cannot simply be called persistent
  low frequency.
- Progress alignment preserves the degradation: mean absolute lateral error
  at 20--40 m is 0.136 versus 0.604 m, and at 80--120 m is 0.260 versus 0.643 m.

## Interpretation and remaining uncertainty

The immediate behavioral failure is a systematic signed tracking offset with a
changed surface/frequency operating point. It occurs before substantial tail-limit
activity and across varied path directions, weakening explanations based solely
on curve anticipation, speed-induced time misalignment, or early saturation.
Identical-input action differences establish policy change, but paired closed-loop
traces do not identify whether the rudder change, elevon change, frequency change,
or their interaction causes the drift. Nor do they explain which PPO gradient or
reward component caused that change during iterations 26--50.

Next discriminating experiment: controlled channel replacement between checkpoints
25 and 50 on the same observations, separately testing frequency and tail channels,
with closed-loop comparisons on this development set. Keep holdout sealed and
plant/reward/preview fixed. This experiment was not launched in this task.
