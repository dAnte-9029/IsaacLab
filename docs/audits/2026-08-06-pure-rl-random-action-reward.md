# PureRL bounded random-action reward gate

Date: 2026-08-06

Status: accepted at the 3.0-second lifecycle window and reproduced exactly in
a second fresh process. No PPO training was started.

## Material Passport

- Origin skill: `academic-research-suite`, experiment-agent
- Mode: run and reproducibility validation
- Branch: `feat/native-multibody-rl`
- Baseline commit: `e707e816`
- Runtime plant: CPU native measured-wing multibody DeLaurier plant
- Physics/policy rates: 480/60 Hz
- Environment/action seeds: `20260806` / `20260807`
- Initial 1.5-second evidence: `/tmp/pure_rl_random_action_reward_20260806`
- Accepted 3.0-second evidence: `/tmp/pure_rl_random_action_reward_20260806_3s`
- Exact reproduction: `/tmp/pure_rl_random_action_reward_20260806_3s_repro`
- Verification status: accepted and exactly reproduced across 55 NPZ arrays

## Experiment contract

The 64 free-root environments were split by environment index into two action
families. The first family independently sampled every normalized action
channel over `[-1, 1]` at each 60 Hz policy step. The second used a deterministic
correlated random process with coefficient `0.25`; its frequency target used
the full normalized range while each tail target was restricted to
`[-0.8, 0.8]`. After automatic environment reset, correlated process state was
synchronized to the actual reset action before exploration resumed.

The gate retained requested and applied actions, returned observations,
rewards and done flags at every step. Every non-terminal state also retained
the preceding applied action, physical route-relative state and every
unweighted reward term for independent reward reconstruction. Step-level
telemetry retained all reward and termination channels.

## Initial duration diagnostic

The frozen 1.5-second command completed all 90 policy steps. Fourteen of 15
gates passed: 48 environments reset once and produced 1,282 post-reset samples,
but no environment terminated a second time. This result remains preserved as
a failed gate. It showed that reset and post-reset stepping worked, while the
window was too short for the stricter repeated-reset requirement.

The follow-up changed only duration to the predeclared maximum of 3.0 seconds.
Seed, action distributions, plant, task and all acceptance thresholds remained
unchanged.

## Accepted result

All 15 gates passed over 180 policy steps and 11,389 independently rebuilt
non-terminal samples:

- requested normalized actions remained within bounds;
- applied actions matched requested actions exactly on pre-reset samples;
- the IID family covered both extremes of all four action channels;
- all observations, rewards, state terms and telemetry remained finite;
- maximum absolute normalized observation was `1.3125`, below the safety clip;
- independent reward reconstruction error was zero;
- reward-total telemetry error was `1.27e-7`;
- termination telemetry error was zero;
- action-delta penalties remained within their mathematical range;
- all 64 environments reset at least once, 60 reset at least twice, and the
  run completed 131 reset events with 6,814 post-reset samples;
- no episode timeout occurred.

The returned reward ranged from `0.2286` to `0.9963`, with mean `0.7478`. The
IID full-range action family produced mean combined delta penalty `0.3364`,
while the correlated family produced `0.0100`, confirming that telemetry
distinguishes high-rate chatter from smooth exploration.

Termination telemetry recorded 73 height-error and 59 tilt cause events. Their
sum exceeds the 131 termination unions by one because one state crossed both
boundaries on the same step. There were no ground or cross-track events.
These uncontrolled failures are lifecycle coverage, not policy-quality data.

## Reproducibility

The accepted 3.0-second command was executed again in an independent fresh
process with identical seeds and configuration. The JSON summaries were
identical after removing timestamp/command provenance. All 55 aligned NPZ
arrays matched exactly with `np.array_equal`.

Use a fresh output directory for reproduction because the CLI refuses to
overwrite evidence:

```bash
conda activate env_isaaclab
export PYTHONPATH=$PWD/source/flapping_bot

./isaaclab.sh -p scripts/flapping_px4/validate_pure_rl_random_action_reward.py \
  --summary <output>/summary.json \
  --traces <output>/traces.npz \
  --num-envs 64 --duration-s 3.0 --seed 20260806 --headless
```

Isaac Sim emitted its existing headless-window, generated-schema and CPU
`powersave` governor warnings. None caused a gate failure. The Isaac Lab shell
wrapper does not reliably propagate the Python child's nonzero acceptance
status, so formal verification must inspect `summary.json` rather than relying
only on the wrapper exit code.

## Claim boundary and next gate

The fixed-action and bounded random-action gates now establish reward,
termination, observation and automatic-reset wiring under the current action
contract. They do not establish PPO learnability, convergence or stable flight.
The next separate stage is a short PPO smoke with explicit seed count,
iterations, logging requirements and launch-only acceptance criteria.
