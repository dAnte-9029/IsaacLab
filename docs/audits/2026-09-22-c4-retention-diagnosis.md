# C4 retention: D-50 to D-75 diagnosis

## Scope and provenance

Read-only analysis of the completed seed-0 A--D pilot, followed by two diagnostic
channel-combination evaluations on the same 96-case development set. No training,
holdout access, physical changes or threshold relaxation. Worktree:
`.worktrees/native-multibody-rl`, branch `feat/native-multibody-rl`, base `41ea8565`.
Preexisting dirty work remains intact. No commit was requested.

Reproducible analysis: `source/flapping_bot/flapping_bot/analysis/c4_retention_diagnosis.py`.
Evidence: `logs/continual_c4/20260922_diagnosis_verified/summary.json` records hashes
of the script, checkpoints, memory, event files, diagnostic logs and development traces.
Pilot inputs/results remain unchanged under `logs/continual_c4/20260922_pilot/`.

## Completed pilot

All four arms completed 101 updates and evaluations at 50/75/100, using 4,964,352
training transitions in 7,126.79 seconds. No checkpoint passes all six suites.

| Arm | C4 successes at 50 / 75 / 100 | Legacy suites passed at 50 / 75 / 100 |
| --- | --- | --- |
| A fixed | 5 / 0 / 0 of 96 | 3 / 4 / 2 of 5 |
| B adaptive | 67 / 0 / 0 of 96 | 4 / 4 / 3 of 5 |
| C memory | 96 / 0 / 0 of 96 | 3 / 4 / 4 of 5 |
| D adaptive + memory | 96 / 0 / 0 of 96 | 4 / 5 / 3 of 5 |

D-50 misses only C3b's mean horizontal error gate: 0.50599 m versus 0.500 m.
D-75 passes all old suites but fails C4. These are research checkpoints, not promoted
controllers. Memory delays regression in this seed; long-term retention is unresolved.

## Observations

### The actor changes on identical states

Both D actors were evaluated on every stored policy input from all 96 D-50 and
96 D-75 development traces, 115,104 observations per checkpoint. Reconstructed
actions match the corresponding recorded actions within 1.20e-7, checking strict
weight loading, actor construction and observation/action alignment.

On D-50 observations, the mean D-75 minus D-50 normalized action differences are
`[+0.0294, -0.1291, -0.1993, -0.1507]` in frequency/rudder/left/right order.
RMS differences are `[0.0704, 0.1352, 0.2068, 0.1534]`. On D-75 observations,
tail RMS differences rise to `[0.3027, 0.3639, 0.2871]`. These are network outputs,
not physical actuator angles. Same-state comparisons establish policy drift
independently of state divergence, not the training cause of that drift.

Actual closed-loop signed mean lateral/vertical errors change from
`0.247 / 0.253 m` to `0.505 / 0.693 m`. Along-path speed decreases from
8.605 to 8.012 m/s. Average applied flap frequency changes only from 2.964 to
3.006 Hz. Signs follow the evaluator convention; no above/below interpretation
is assumed here.

### Memory is active but permits substantial C4 error

The installed RSL-RL `PPO.update` uses the appended memory observations for an
actor-mean MSE with detached stored targets. `attach_behavior_memory` enables this
auxiliary loss, not PPO data augmentation. D executed 5,792 auxiliary calls.
Mean logged auxiliary MSE rises from 0.00806 over iterations 25--50 to 0.01465
over 51--75. These loss values cannot be compared directly with PPO loss values
to infer gradient dominance.

On the fixed teacher memory, D-50 family-level action MSE is 0.00297--0.00432
for old families and 0.04580 for C4. At D-75 it is 0.00594--0.00806 versus
0.05335. Balanced family counts therefore do not imply equal behavior retention.
The memory source is the original full-C3 actor, not D-50. It contains successful
normal rollouts without deliberately injected recovery states; coverage away from
those trajectories is unverified.

Action standard deviations at D-50 and D-75 are respectively
`[0.2774, 0.1933, 0.1424, 0.1442]` and `[0.2686, 0.1934, 0.1411, 0.1419]`.
Variance growth is not supported; the deterministic evaluation also excludes sampled
action noise as its immediate failure mechanism.

### Conflict exists, but does not establish that old tasks overrode C4

At sampled updates 55/60/65/70/75, C1--C4 gradient cosines are all negative;
C3b--C4 cosines are negative at three of five probes. However, the dot product
of the actual actor update with the sampled C4 PPO-loss gradient is negative at
all five probes, with mean -0.02407. Locally, the aggregate update is a descent
direction for that sampled C4 surrogate too. This is not proof that the true
tracking objective improves; gradients use limited on-policy samples and estimated
advantages. It does rule out claiming simple gradient conflict as a proven sole cause.

C4 critic MSE decreases within these updates, but states/return targets change
between iterations. Neither critic correctness nor critic failure is established.
Mean C4 total reward drops 0.9194 -> 0.8742 and path reward 0.9691 -> 0.8531;
there is no evidence here that overall C4 reward improves while tracking deteriorates.

### Update control and episode feedback

The warm-start guard deliberately forces `algorithm.schedule = 'fixed'` in
`px4_like/rl_training_utils.py`; the effective learning rate remains 5e-5.
The configured `desired_kl=0.01` therefore does not enforce the adaptive learning-rate
rule. Mean measured C4 post-update KL is 0.02011 over probes 55--75. This compares
whole-update, task-specific KL with a minibatch scheduler setting; it is not a claim
that a formally enforced KL limit was violated.

D's curriculum updates only five times over the complete run. After update 50,
its completed-episode statistics stay at 100% success through update 70, and the
probabilities remain unchanged. With 1,200-step episodes and 48-step rollouts,
feedback arrives around 25 optimizer iterations apart. Later episode scores combine
behavior from multiple policy versions, so they lag current checkpoint performance.
This delay limits the curriculum's ability to arrest within-episode drift. It does
not alone explain failure: fixed-sampling C also regresses.

## Same-observation channel intervention

Two fresh CPU-native processes evaluated all 96 development cases with the same
1.2-second preview. At each step both actors see the same current observation;
this is closed-loop action replacement, not replay of another rollout's actions.

| Frequency actor | Tail actor | C4 successes | Mean horizontal / vertical error |
| --- | --- | --- | --- |
| D-50 | D-50 | 96/96 | 0.2710 / 0.2548 m |
| D-75 | D-75 | 0/96 | 0.5053 / 0.6936 m |
| D-75 | D-50 | 96/96 | 0.2450 / 0.2705 m |
| D-50 | D-75 | 0/96 | 0.6074 / 0.5795 m |

Both interventions have 100% survival and completion. Results are under
`logs/continual_c4/20260922_diagnosis/f75_tail50/` and `f50_tail75/`.
The evaluator's `launch.json` records exact commands and hashes. All 96 initial
observations and all recorded hybrid actions are checked by `swap_validation.json`.
Restoring the earlier tail policy is sufficient to restore development tracking
with the later frequency policy; replacing frequency alone is not. This localizes
the immediate behavioral degradation to tail-policy changes for these checkpoints
and cases. It does not identify an individual actuator or the optimizer-level cause,
nor does it promote the hybrid controller: its old-task retention was not evaluated.

## Interpretation and next bounded experiment

Current evidence supports systematic tail action-mean drift despite a soft memory
loss, with slow curriculum feedback. The channel intervention confirms the immediate
tail-policy mechanism; it does not identify PPO, critic sharing, one tail actuator,
or reward design as a sole training cause.

The next training experiment should first isolate update stability while keeping
physics, rewards, observation, geometry bounds and memory data fixed: compare the
current update against a smaller tail-update variant, monitor per-family fixed-memory
drift and current-policy tracking more frequently, and retain all six acceptance
suites. A task-specific behavior constraint is a candidate if smaller updates only
delay failure. Define the reference policy, constraint budget and rollback/optimizer
semantics before implementation; avoid selecting thresholds to make D-50 pass.
PCGrad or separate critics are not yet justified as the first change by these probes.

## Reproduction and validation

```bash
source /home/zn/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export PYTHONPATH="$PWD/source/flapping_bot:$PYTHONPATH"
python -m flapping_bot.analysis.c4_retention_diagnosis \
  --pilot logs/continual_c4/20260922_pilot \
  --memory logs/continual_c4/20260922_memory/memory.pt \
  --output logs/continual_c4/NEW_UNIQUE_DIAGNOSIS_DIRECTORY
```

The analysis completed with strict actor loads, all 192 expected traces, finite JSON
output and recorded-action reconstruction assertions. It refuses to overwrite outputs.
No new unit suite is needed for this read-only report workflow; actual artifact
consistency checks validate its principal claims. `git diff --check` is also required.
The two fresh-process intervention evaluations completed with finite summaries.
