# PureRL C3a split-actor promotion audit

- Date: 2026-08-29
- Branch: `feat/native-multibody-rl`
- Training run: `2026-08-29_10-07-47_pure_rl_c3a_split_frequency_actor_seed0_201iter`
- Policy: `PureRLSplitActorCritic`
- Authority plant: CPU-native measured multibody
- Current-stage contract: `pure_rl_spatial_c3a_v2`

## Decision

`model_200.pt` is the formally promoted C3a checkpoint and the required weights-only policy source for the first
C3b experiment. It is selected as the later member of the adjacent passing pair `model_175.pt` / `model_200.pt`.
The repository promotion helper returned `promoted=true`; both checkpoints passed C3a and the required frozen C1
and C2c retention suites without a recorded forgetting failure.

This promotion applies to the split actor architecture. The frequency actor receives the one-cycle-averaged,
phase-fixed 555-value transform, while the tail actor receives the full phase-aware 555-value observation. A
standard shared-actor runner must not load this checkpoint as if it used the baseline actor class.

## Training and lineage

The run used weights-only initialization from the governor-gap joint C3a `model_200.pt` at:

```text
logs/rsl_rl/flapping_bot_straight_flight/
2026-08-28_15-56-43_pure_rl_c3a_joint_reqappliedgap005_seed0_201iter/model_200.pt
```

The fresh run used seed 0, 256 CPU-native environments, 16 mini-batches, 201 iterations, checkpoint interval 25,
the fixed 15/35/50 C1/C2c/C3a mixture, C2c strong-climb probability 0.5, task-aware PPO, bounded warm start,
governor-gap weight 0.05, no distillation and no adaptive sampling. Shared source actor weights were mapped
strictly into independent `[256,128]` frequency and tail trunks; the critic remained `[256,128]`.

## Frozen evidence

All seven checkpoints from iteration 50 through 200 were evaluated in separate fresh CPU-native C3a, C1 and C2c
processes. The complete combined evidence is:

```text
logs/rsl_rl/flapping_bot_straight_flight/
2026-08-29_10-07-47_pure_rl_c3a_split_frequency_actor_seed0_201iter/
eval_authority_50_200_split_actor/checkpoint_evaluation.json
```

The adjacent promotion pair is:

| Checkpoint | C3a survival / completion / success | C1 success / termination | C2c survival | C2c climb / descent / recovery | All gates |
| --- | ---: | ---: | ---: | ---: | --- |
| `model_175.pt` | 1.000 / 1.000 / 1.000 | 1.000 / 0.000 | 0.96429 | 0.91667 / 1.000 / 0.96429 | pass |
| `model_200.pt` | 1.000 / 1.000 / 1.000 | 1.000 / 0.000 | 1.000 | 1.000 / 1.000 / 1.000 | pass |

Additional selected metrics:

| Checkpoint | C3a mean horizontal / vertical error (m) | C3a roll-limit terminations | C1 tail-limit fraction | C2c mean / P95 height error (m) | `+12 deg` success |
| --- | ---: | ---: | ---: | ---: | ---: |
| `model_175.pt` | 0.15873 / 0.12535 | 0 | 0.08658 | 0.25435 / 0.82806 | 12/16 |
| `model_200.pt` | 0.14928 / 0.08781 | 0 | 0.06267 | 0.25854 / 0.86332 | 16/16 |

The retention comparison used the selected stable C1 `model_1300.pt` as the C1 baseline and promoted C2c
`model_550.pt` as the longitudinal baseline. The standardized retention success rate is timeout rate for C1 and
overall survival rate for C2c. Relative to those baselines, both adjacent checkpoints remained within the allowed
five-percentage-point success drop; the C1 score and error-retention limits also passed.

The exact promotion result is:

```text
logs/rsl_rl/flapping_bot_straight_flight/
2026-08-29_10-07-47_pure_rl_c3a_split_frequency_actor_seed0_201iter/
eval_authority_50_200_split_actor/promotion_result_175_200.json
```

## Frequency-request diagnostic

On the fixed `+12 deg`, heading-0, reset-phase-0 trajectory, deterministic split `model_200.pt` produced five raw
requested-frequency zero crossings, all during the initial entry segment and all with values smaller than 0.05 in
absolute normalized action on at least one side. It produced zero such crossings during the active climb and zero
crossings with both sides above 0.05. Its requested-action mean absolute step delta was 0.01346, compared with
0.365--0.437 for earlier problematic joint policies. This supports removal of the high-amplitude,
wingbeat-synchronous inference-time flip on the checked trajectory; stochastic PPO rollout flip telemetry did not
become zero and is not presented as deployment evidence.

Diagnostic traces are under:

```text
eval/flip_diagnostic_shared_vs_split_model200_slope12_h0_p0/
```

## Scope and next gate

This result promotes C3a only. It does not establish C3b/C3c performance, wind robustness, multi-seed
reproducibility, aerodynamic-model validity or real-flight readiness. Because later curricula can still forget
earlier abilities, every C3b candidate must pass frozen C1, C2c and C3a retention; every C3c candidate must pass
C1, C2c, C3a and C3b retention. The next diagnostic should be a zero-shot C3b grid evaluation of this exact
`model_200.pt` before any C3b training recipe is selected.
