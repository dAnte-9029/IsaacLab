# ADR-2026-08-29: Isolate PureRL flap frequency in a cycle-averaged actor branch

- Status: Accepted for controlled experiment
- Date: 2026-08-29
- Scope: one project-local PureRL policy architecture and one C3a training route

## Context

The C1-initialized governor-gap run produced one checkpoint, `model_200.pt`, that passed the frozen C3a, C1 and
C2c suites. A same-optimizer continuation to `model_225.pt` preserved C1 and C2c but failed C3a because one of 96
cases reached the roll limit. The run therefore has no adjacent all-suite passing pair.

The governor-gap objective reduced the requested/applied mismatch but did not eliminate the deterministic
wingbeat-synchronous requested-frequency sign reversal. The shared 60 Hz actor receives 30 sensor frames,
including sine and cosine flap phase, and emits both a slow flap-frequency request and three fast tail-surface
commands. The tail controls can legitimately use phase-dependent state, while the frequency request is executed
behind a symmetric 2 Hz/s governor. A shared hidden representation gives the frequency output a direct route to
wingbeat phase even though that signal cannot be followed physically at the requested rate.

Splitting only the final linear layer would be algebraically equivalent to the existing four output rows and
would not remove that route. The experiment therefore requires independent hidden trunks and different input
representations.

## Decision

Add the project-local `PureRLSplitActorCritic` policy with two independent `[256,128]` actor trunks:

```text
full normalized 555 observation -> tail trunk -> rudder, left elevon, right elevon
cycle-averaged phase-fixed 555 observation -> frequency trunk -> flap frequency
```

The slow frequency observation is derived inside the actor without changing the environment observation
contract. The most recent normalized actual frequency selects a one-cycle history window:

```text
cycle steps = clamp(round(60 Hz / actual frequency), 12, 30)
```

Sensor values and governor-applied actions are averaged over the newest window. The sign-aligned quaternion is
renormalized, every phase pair is replaced by the valid constant `(sin, cos) = (0, 1)`, and the averaged sensor
and action frames are repeated to retain 555 values. The existing 15-value path preview remains unchanged. The
frequency branch still runs at 60 Hz in this first experiment; no sample-and-hold is added.

Warm start is strict and deterministic. From the shared `[256,128]` four-action actor:

- both new trunks receive exact copies of both hidden layers;
- source output row zero initializes the frequency output;
- source output rows one through three initialize the tail output;
- critic and `log_std` are copied exactly;
- the optimizer and learning iteration start fresh.

The controlled route is `c3a_split_frequency_actor_v1`. It starts weights-only from the last all-suite passing
checkpoint:

```text
logs/rsl_rl/flapping_bot_straight_flight/
2026-08-28_15-56-43_pure_rl_c3a_joint_reqappliedgap005_seed0_201iter/model_200.pt
```

It runs seed 0, 256 environments, 16 mini-batches, 201 iterations and 25-iteration checkpoint cadence. It keeps
the 15/35/50 task mixture, strong-climb probability 0.5, task-aware PPO, bounded warm start, governor-gap weight
0.05, cubic frequency penalty 0.04, symmetric 2 Hz/s governor, critic `[256,128]`, no distillation and no adaptive
sampling. The generic low-pass and action-rate limiter remain disabled.

The policy class is registered only from the project task package; no RSL-RL or upstream Isaac Lab file is
modified. Frozen evaluation reconstructs the class through an explicit flag.

## Alternatives considered

- Splitting only the output layer was rejected because it is equivalent to selecting different rows of the
  existing output matrix after the same phase-aware trunk.
- Removing phase from the complete actor observation was rejected because tail commands may need fast
  phase-conditioned state.
- Disabling the physical frequency governor was rejected because it would expose the plant to the existing
  high-frequency request and change the accepted action interface.
- Adding cycle-level frequency sample-and-hold in the same experiment was deferred so representation isolation
  is tested as one variable. It is a reconsideration option only if sign reversal persists.
- Random initialization was rejected because it would discard the existing compatible C1/C2c/C3a behavior and
  confound the architecture comparison with exploration.

## Consequences

- Frequency output is exactly invariant to direct changes in all 30 phase pairs.
- Tail output retains the full phase-aware observation and independent capacity.
- The actor parameter count increases because both trunks have complete hidden layers; the critic is unchanged.
- Initial tail output equals the source actor on the full observation. Initial frequency output equals source
  action row zero evaluated on the transformed slow observation, not on the original phase-aware observation.
- Existing `ActorCritic` routes and checkpoint formats remain available. Split checkpoints require explicit
  reconstruction when saved config is intentionally ignored.
- This is a single-seed controlled experiment. Training startup or a single passing checkpoint is not promotion
  evidence.

## Assumptions

- The normalized 555-value order remains 30 by 14 sensor history, 30 by 4 applied-action history and 15 preview
  values.
- Actual frequency remains normalized affinely from 0--5 Hz to `[-1,1]` and the policy rate remains 60 Hz.
- Quaternion history remains sign aligned, so a cycle mean can be renormalized without introducing an artificial
  sign discontinuity.
- Keeping the preview unchanged does not reintroduce an explicit mechanical phase channel.

## Validation requirements

- Pure Tensor tests must prove shape, batch, dtype, one-cycle averaging and exact phase invariance of the
  frequency observation.
- A constructed tail branch must remain able to respond to phase.
- Warm-start tests must prove `new_tail(obs) == old_actor(obs)[1:4]` and
  `new_frequency(obs) == old_actor(slow_obs)[0]`, with critic and `log_std` unchanged.
- Launcher tests must freeze source checkpoint, fresh optimizer, policy class, reward and sampling recipe.
- Frozen evaluator tests must forward explicit split-policy reconstruction to every fresh suite process.
- A fresh CPU-native smoke must construct the registered policy, load the shared checkpoint through the strict
  converter and complete the first trainable update without traceback before the long run is handed off.
- Final evidence remains the frozen 96-case C3a, 16-case C1 and 112-case C2c suites plus the fixed positive
  12-degree response comparison.

## Reconsideration triggers

Add a separate cycle-level frequency sample-and-hold experiment only if the split frequency branch still produces
material wingbeat-synchronous sign reversal. Reconsider the phase-isolation assumption if removing the direct
phase route degrades maneuver onset or frozen path accuracy. Do not remove the governor, change the plant or
alter frozen gates to make the split architecture pass.
