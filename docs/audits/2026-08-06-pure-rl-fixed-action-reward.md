# PureRL curriculum-1 fixed-action reward gate

Date: 2026-08-06

Status: accepted. The reward, termination and telemetry wiring passed a
64-environment CPU-native fixed-action rollout. No random-action rollout or PPO
training was started.

## Material Passport

- Origin skill: `academic-research-suite`, experiment-agent
- Mode: run and validate
- Branch: `feat/native-multibody-rl`
- Baseline commit: `e707e816`
- Runtime plant: CPU native measured-wing multibody DeLaurier plant
- Physics/policy rates: 480/60 Hz
- Seed: `20260806`
- Generated evidence: `/tmp/pure_rl_fixed_action_reward_20260806_v2`
- Authoritative files: `summary.json` and `traces.npz`
- Verification status: fresh-process fixed-action gate accepted

## Experiment contract

The gate ran 64 free-root environments for 0.75 seconds. Environment index
modulo eight selected one constant action family: reset action, 0 Hz, 2.5 Hz,
5 Hz, positive rudder, positive left elevon, positive right elevon, or all
three tail commands at normalized `0.9`. Route heading and flap phase retained
the task's reset randomization.

Each non-terminal sample retained the physical state, applied action, returned
reward and every unweighted reward term. The analysis independently rebuilt
the reward from that state and the previous action. Step-level traces retained
every reward and termination telemetry scalar. Samples after an environment's
first automatic terminal reset were excluded from the per-environment reward
analysis.

## Results

All 15 acceptance gates passed. The returned environment reward, independently
rebuilt reward and logged mean reward agreed exactly at float32 resolution.
The returned termination fraction and logged termination fraction also agreed
exactly. Applied actions showed zero drift from the requested fixed actions.

The first action transition produced a maximum combined normalized delta
penalty of `0.64000005`; every later constant-action sample produced zero delta
penalty. Settled 0, 2.5 and 5 Hz groups tracked their requested frequency with
maximum error `4.3e-6 Hz`. Their cubic frequency penalties matched 0, 0.125 and
1 with maximum error `2.5e-6` and were strictly ordered. Ordinary action groups
had zero tail soft-limit penalty, while the normalized `0.9` group produced a
mean penalty of `0.24999979`, matching the configured soft-limit expression.
No target-speed reward term is present.

Forty-eight of 64 original environments reached termination during the
uncontrolled fixed-action rollout. All recorded termination events were caused
by the 75-degree tilt boundary; no ground, cross-track, height-error or timeout
event occurred. Four additional tilt events came from automatically restarted
episodes that remained in the simulator but were excluded from sample analysis.
This is expected diagnostic behavior and is not a policy-stability result.

## Reproduction

Use a fresh output directory because the CLI refuses to overwrite evidence:

```bash
conda activate env_isaaclab
export PYTHONPATH=$PWD/source/flapping_bot

./isaaclab.sh -p scripts/flapping_px4/validate_pure_rl_fixed_action_reward.py \
  --summary <output>/summary.json \
  --traces <output>/traces.npz \
  --num-envs 64 --duration-s 0.75 --seed 20260806 --headless
```

Isaac Sim emitted its existing headless-window, generated-schema and CPU
`powersave` governor warnings. None caused a gate failure.

## Claim boundary and next gate

This gate establishes that the fixed reward economics reach the environment
and telemetry correctly under interpretable constant actions. It does not
establish closed-loop stability, learnability, convergence or task success.
The next separate gate is a bounded random-action rollout; only after that
passes should a short PPO smoke be considered.
