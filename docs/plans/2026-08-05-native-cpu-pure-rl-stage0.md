# Native CPU PureRL Stage 0 Plan

Date: 2026-08-05

Status: approved for implementation and a bounded smoke only. This plan does
not authorize a long training run, plant retuning, reward redesign or held-out
evaluation.

Update: the Stage 0 launch result remains valid, but its inherited mixed
elevon action interface was superseded for the Measured PureRL task by
`docs/plans/2026-08-06-pure-rl-action-step-response.md`.

## 1. Provenance and objective

- Branch: `feat/native-multibody-rl`
- Baseline: `flapping_rl` at `1200ca8d`
- Task: `Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-Direct-v0`
- Plant: canonical CPU native measured-wing multibody plant
- Control objective: establish a model-free PureRL training path before any
  curriculum expansion or domain randomization.
- Reference: Cai et al., *Learning-based Trajectory Tracking for Bird-inspired
  Flapping-Wing Robots*, local source at
  `/home/zn/flap-system-identification/docs/papers/Cai 等 - Learning-based Trajectory Tracking for Bird-inspired Flapping-Wing Robots.pdf`.

The paper informs the curriculum structure, history/preview observations and
model-free PPO framing. At this Stage 0 snapshot, the task still exposed the
four inherited high-level controls: flap frequency, rudder, elevon pitch and
elevon roll. The later direct-surface action decision preserves four channels
but replaces the last two with independent left and right elevon commands.

## 2. Stage hypothesis

The canonical native plant can complete environment creation, batched reset,
observation/reward computation, policy inference and PPO updates on CPU without
the native constraint failing or the learning pipeline producing non-finite
values.

This smoke does not test whether a useful controller can be learned. Two PPO
iterations are only a launch and data-path gate.

## 3. Why reset freeze is disabled

The canonical configuration freezes each reset for 240 physics steps. At
`dt=1/480 s` and decimation 4, this is 0.5 s or 60 policy steps. The current PPO
rollout contains 48 policy steps per environment, so its first rollout would be
entirely frozen. Stage 0 therefore explicitly sets
`env.freeze_steps_after_reset=0` rather than changing the task default.

## 4. Implementation scope

Allowed changes:

- add an explicit native CPU launch option to
  `scripts/flapping_rl/train_and_watch.py`;
- load `omni.flapping_bot.holonomic_constraint` at Kit startup;
- route both PhysX and the RSL-RL policy/optimizer to CPU for this mode;
- expose a non-negative reset-freeze override;
- add focused launcher-contract tests and this plan.

Out of scope:

- plant, mass, inertia, aerodynamic or mechanism changes;
- observation, action, reward, termination or PPO hyperparameter changes;
- GPU implicit-drive substitution;
- curriculum stages beyond straight flight;
- long training, checkpoint selection or performance claims.

Baseline behavior remains available: without `--native-cpu`, the existing
train/evaluation device arguments and portable Kit roots are unchanged.

## 5. Stage 0 command

After activating `env_isaaclab` and building the extension for the installed
Isaac Sim ABI if necessary:

```bash
./isaaclab.sh -p scripts/flapping_rl/train_and_watch.py \
  --task Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-Direct-v0 \
  --run-name native_cpu_pure_rl_p0_smoke \
  --native-cpu \
  --freeze-steps-after-reset 0 \
  --num-envs 64 \
  --eval-num-envs 1 \
  --max-iterations 2 \
  --save-interval 1 \
  --episodes 1 \
  --poll-s 10 \
  --headless
```

## 6. Acceptance gates

Stage 0 passes only if:

1. launcher contract tests pass;
2. the extension loads during fresh-process Kit startup;
3. the task creates 64 CPU environments and completes two PPO iterations;
4. losses, rewards and observations remain finite;
5. at least one checkpoint is written and the process exits successfully;
6. logs preserve task, seed, device and configuration provenance.

A watcher or evaluation failure is recorded separately from training startup;
it must not be recast as policy-quality evidence.

## 7. Subsequent curriculum, pending separate approval

Following Cai et al., later work should add difficulty in stages: stable
straight flight first, then climb/dive and speed variation, then turns and
general path tracking, and finally aerodynamic/physical/wind randomization.
Each stage must introduce its own training and held-out scenarios. Test cases
must remain sealed from reward, curriculum and checkpoint-selection decisions.

## 8. Resource note

The pre-run inventory reported 24 physical CPU cores, 125.47 GB RAM and ample
disk. The machine also has two RTX 4090 GPUs, but this P0 intentionally uses CPU
for both PhysX and PPO because the canonical native constraint rejects a
direct-GPU scene. The inventory was written outside the repository at
`/tmp/native_multibody_rl_resources.json`.

## 9. Material Passport

- Origin skill: `academic-research-suite`, experiment-agent
- Mode: run
- Created: 2026-08-05
- Inputs: repository code/configuration, canonical native-plant handoff and Cai
  et al. local paper
- Verification status: launcher contract and bounded Stage 0 runtime passed

## 10. Stage 0 execution result

The approved command completed successfully on 2026-08-05. The fresh process
created 64 CPU environments, completed two PPO iterations and 6,144 environment
steps, wrote `model_0.pt` and `model_1.pt`, and returned code 0. Reported
collection/update throughput was 1,388 and 1,835 steps/s for the two iterations;
the host warned that its CPU governor was set to `powersave`, so these numbers
are not adopted as a formal performance benchmark.

The logged value and surrogate losses, rewards and wing-drive diagnostics were
finite. `Teacher/enabled=0`, `Teacher/active=0` and
`Wind/curriculum_scale=0` confirm the intended straight-flight PureRL/no-wind
gate. `Teacher/freq_hz=nan` is an inactive-teacher diagnostic sentinel, not a
policy observation or training loss.

The one-environment watcher evaluated both checkpoints and exited successfully.
Its scores are startup evidence only; with two training iterations they are not
policy-quality or task-success results. The run directory is:

`logs/rsl_rl/flapping_bot_straight_flight/2026-08-05_16-36-32_native_cpu_pure_rl_p0_smoke`

The generated URDF converter metadata contains absolute paths into the primary
checkout, and this worktree does not contain the ignored generated USD. The
watcher therefore regenerated that ignored USD in `/home/zn/IsaacLab` and
temporarily rewrote two tracked converter metadata files there. The tracked
files were restored byte-for-byte and the primary checkout is clean; the
ignored generated USD remains a reproducible cache artifact. Before a formal
long run, asset-generation ownership and worktree isolation should be resolved
under separately approved `source/isaaclab_assets/` scope.
