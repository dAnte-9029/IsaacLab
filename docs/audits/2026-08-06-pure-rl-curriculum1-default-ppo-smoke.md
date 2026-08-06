# PureRL curriculum-1 default PPO smoke

## Material Passport

- Date: 2026-08-06
- Branch: `feat/native-multibody-rl`
- Baseline commit: `e707e816`
- Task: `Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-Direct-v0`
- Experiment type: launch and optimizer smoke
- Verification status: completed
- Claim boundary: runtime integration only; this is not convergence or controller-performance evidence

## Objective

Verify that the accepted MeasuredPureRL defaults can pass through Hydra, create
the CPU-native measured multibody environment, collect physical rollouts,
perform PPO updates, save checkpoints and evaluate those checkpoints without
requiring the former native CPU, environment-count or reset-freeze command-line
flags.

## Pre-smoke configuration fix

The first launch stopped before environment construction with
`dataclasses.FrozenInstanceError: cannot assign to field
'cross_track_scale_m'`. Isaac Lab's Hydra bridge restores nested configuration
objects by recursively assigning their fields in place. The newly introduced
`PureRLRewardConfig` was frozen, so that framework-compatible restoration could
not complete.

After explicit user approval, the fix changed only `PureRLRewardConfig` from a
frozen dataclass to a mutable dataclass. Its field names, values and reward
formulas were unchanged. Reward and termination result objects remain frozen.
A regression test now invokes the same `update_class_from_dict` helper used by
the Hydra path, checks exact restoration of every default, verifies an
identical reward for identical inputs and confirms that mutating the restored
instance does not change a fresh or module-default configuration.

## Command

```bash
export TERM=xterm-256color
source /home/zn/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export PYTHONPATH="$PWD/source/flapping_bot${PYTHONPATH:+:$PYTHONPATH}"
./isaaclab.sh -p scripts/flapping_rl/train_and_watch.py \
  --task Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-Direct-v0 \
  --run-name curriculum1_default_smoke \
  --max-iterations 2 --save-interval 1 --seed 0 \
  --eval-num-envs 1 --episodes 1 --poll-s 5 \
  --eval-suite single --headless
```

The launcher resolved the omitted task defaults to CPU for both PhysX and
RSL-RL, 64 training environments, the project-local native extension and
isolated generated assets.

## Saved configuration

The saved `params/env.yaml` records:

- `device: cpu` and `num_envs: 64`;
- physics `dt: 0.0020833333333333333 s` and `decimation: 8`, giving a 60 Hz policy cadence;
- `replicate_physics: false`;
- randomized straight-line heading and flap phase;
- curriculum-1 reward and per-term telemetry enabled;
- command randomization and teacher guidance disabled;
- `freeze_steps_after_reset: 0`;
- `min_flap_hz: 0.0` and `max_flap_hz: 5.0`.

The saved `params/agent.yaml` records `device: cpu`. The resolved actor and
critic each consumed the 555-value policy observation; the actor produced four
actions.

## Training result

Training returned code zero after two PPO iterations:

| Iteration | Total steps | Throughput | Mean reward | Mean episode length | Value loss | Surrogate loss |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3,072 | 719 steps/s | 26.1064 | 32.4444 | 89.3582 | -0.00768 |
| 1 | 6,144 | 778 steps/s | 45.2738 | 62.8393 | 53.8293 | -0.01082 |

The termination telemetry was nonzero during both iterations. At iteration 1,
the terminated fraction was `0.0143` and the timeout fraction was `0.0010`, so
the rollout exercised automatic batched resets rather than only uninterrupted
episodes. Reward terms, loss terms, policy noise and plant diagnostics reported
finite values. The disabled-teacher diagnostic `Teacher/freq_hz` intentionally
reported `NaN`; it is not consumed by observations, rewards or PPO.

The two checkpoints contain 52 tensors each, all finite:

| Checkpoint | Iteration | Size | SHA-256 |
|---|---:|---:|---|
| `model_0.pt` | 0 | 4,228,359 bytes | `05257ccd5c04d0f831c015383d1cef37d4e8119fb657eb5369870f2fb0ab7b06` |
| `model_1.pt` | 1 | 4,228,359 bytes | `e3a542a6a17857fb10824a11889e31e357bd291393aabae206e7b3b24e9a5112` |

Distinct checkpoint hashes and recorded iterations confirm that optimizer work
was saved. Gradient norms are not emitted by the current RSL-RL runner and are
therefore not independently claimed.

## Watcher result

> Superseded for policy-quality interpretation: the one-environment legacy
> watcher result below only established checkpoint loading. The later
> `pure_rl_curriculum1_v1` fixed-grid evaluation is recorded in
> `2026-08-06-pure-rl-curriculum1-checkpoint-evaluation.md` and is the
> authoritative curriculum-1 result.

The watcher loaded and evaluated both checkpoints in a fresh one-environment
CPU process with the no-wind `single` suite:

| Checkpoint | Score | Mean absolute height error | Maximum absolute cross-track | Maximum tilt | Terminated |
|---|---:|---:|---:|---:|---:|
| `model_0.pt` | 9.34918 | 0.40464 m | 0.49904 m | 74.7845 deg | yes |
| `model_1.pt` | 9.26132 | 1.16498 m | 0.49568 m | 23.9780 deg | yes |

`model_0.pt` was selected as the smoke's best checkpoint. Both one-episode
evaluations terminated and no success gate was defined. This is expected for a
two-iteration launch smoke and must not be interpreted as learned stable flight
or PPO convergence.

## Artifacts

The uncommitted runtime artifacts are under:

```text
logs/rsl_rl/flapping_bot_straight_flight/
  2026-08-06_18-57-36_curriculum1_default_smoke/
```

They include `model_0.pt`, `model_1.pt`, TensorBoard events, saved environment
and agent YAML, per-checkpoint evaluation JSON, `eval/summary.csv` and
`eval/best_checkpoint.json`. Generated logs and checkpoints are not intended
for Git.

## Verification commands

The focused configuration/reward/launcher suite passed `44` tests. The related
PureRL action, observation, reward, runtime and launcher suite passed `81`
tests. Relevant files passed `py_compile`, and `git diff --check` passed.

## Conclusion

The Hydra compatibility failure is fixed without changing the experiment's
physical, action, observation, reward, termination or PPO contracts. The
canonical MeasuredPureRL defaults now complete the intended CPU PPO smoke and
watcher pipeline. Convergence, seed robustness and stable straight-flight
performance remain separate follow-up experiments.
