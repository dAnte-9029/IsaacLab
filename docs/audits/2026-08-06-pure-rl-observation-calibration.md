# PureRL actor-observation engineering closure

Date: 2026-08-06

Status: the fixed-scale 555-value actor observation is integrated into the
Measured PureRL environment. Four calibration gates and the end-to-end CPU
runtime gate passed. No policy training was started.

## Material Passport

- Origin skill: `academic-research-suite`, experiment-agent
- Mode: run and validate
- Branch: `feat/native-multibody-rl`
- Baseline commit: `1200ca8d`
- Runtime plant: CPU native measured-wing multibody DeLaurier plant
- Physics/policy rates: 480/60 Hz
- Calibration seed: `20260806`
- Generated evidence: `/tmp/pure_rl_observation_normalized_20260806`
- Authoritative summaries: `synthetic_summary.json`, `reset_summary.json`,
  `scripted_summary.json`, and route-relative `boundary_v2_summary.json`
- Verification status: four fresh-process gates plus the 64-environment CPU
  integration test

## Integrated contract

Only `FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg` selects this
observation. Historical environments retain their 68-value observation and
their existing control interfaces.

The actor receives 555 values in this order:

1. 30 policy-rate sensor frames, oldest to newest. Each frame contains world
   quaternion `wxyz`, body ground velocity, body angular velocity, ideal
   body-x air velocity, actual flap frequency, and flap phase sine/cosine.
2. 30 applied direct-action frames, oldest to newest.
3. Five body-frame straight-line preview points at `0.12`, `0.24`, `0.36`,
   `0.48`, and `0.60 s`.

Reset histories repeat the first real sample instead of adding zero padding.
Quaternion signs are aligned to the preceding policy sample. The preview uses
current along-track ground speed clamped to `1--12 m/s`; it contains no target
speed. The critic remains symmetric and no Pitot noise, delay or filtering is
included in curriculum 1.

Fixed normalization is:

| Quantity | Mapping |
|---|---|
| quaternion, phase sine/cosine | unchanged |
| body ground velocity x | divide by `12 m/s` |
| body ground velocity y/z | divide by `5 m/s` |
| body angular velocity x/y/z | divide by `5 rad/s` |
| ideal body-x air velocity | divide by `12 m/s` |
| actual flap frequency | affine map `0--5 Hz` to `-1--1` |
| applied action | unchanged, already `-1--1` |
| body preview coordinates | divide by `5 m` |

A final `[-5, 5]` safety clip protects the network from exceptional states.
It is not used as a nominal scaling mechanism.

## Reset and route closure

Each Measured PureRL reset samples a straight-line heading uniformly over a
full revolution and rotates the root yaw, world initial velocity and route
tangent together. Cross-track reward and termination now use distance normal
to that route instead of world `y`. Each reset also samples flap phase over
`[0, 2*pi)` and initializes wing joint position, joint velocity, phase state,
frequency state and native constraint target consistently.

The base environment defaults both randomizations off, preserving historical
world-x and zero-phase reset behavior.

## Gate results

The saved samples remain in physical units. Each gate independently evaluated
the fixed normalization before the safety clip and rejected any sample that
would require clipping.

| Gate | Samples | Pre-clip max abs | Values/samples exceeding 5 | Result |
|---|---:|---:|---:|---|
| Synthetic envelope | 8,192 | `4.3062` | `0 / 0` | accepted |
| Randomized reset | 2,048 | `1.0000` | `0 / 0` | accepted |
| Scripted free-root dynamics | 3,378 | `1.8633` | `0 / 0` | accepted |
| Near-boundary injection | 64 | `4.0000` | `0 / 0` | accepted |

The reset gate observed 2,048 distinct sensor frames at `1e-6`, with both
route heading and phase randomization enabled. In the scripted uncontrolled
gate, 58 of 64 environments terminated during the requested 1.5 seconds;
automatic terminal-reset samples were excluded. This is expected diagnostic
coverage, not evidence of closed-loop flight stability. All 64 injected
near-boundary states remained inside the current termination boundaries.

The end-to-end runtime test used 64 CPU environments for 1.5 seconds with
three rounds of forced partial resets. Observations remained finite, had shape
`(64, 555)`, stayed within the safety bound, and reset-filled both histories.
The native mechanism retained maximum wing tracking error `0.0522 deg` and
left/right synchronization error below `0.000004 deg`.

## Reproduction

Run each calibration mode in a fresh process and use a new output directory;
the CLI refuses to overwrite prior evidence:

```bash
conda activate env_isaaclab
export PYTHONPATH=/home/zn/IsaacLab/.worktrees/native-multibody-rl/source/flapping_bot

./isaaclab.sh -p scripts/flapping_px4/calibrate_pure_rl_observation.py \
  --mode synthetic --sample-count 8192 --seed 20260806 \
  --summary <output>/synthetic_summary.json --samples <output>/synthetic_samples.npz

./isaaclab.sh -p scripts/flapping_px4/calibrate_pure_rl_observation.py \
  --mode reset --num-envs 64 --reset-batches 32 --seed 20260806 \
  --summary <output>/reset_summary.json --samples <output>/reset_samples.npz

./isaaclab.sh -p scripts/flapping_px4/calibrate_pure_rl_observation.py \
  --mode scripted --num-envs 64 --duration-s 1.5 --seed 20260806 \
  --summary <output>/scripted_summary.json --samples <output>/scripted_samples.npz

./isaaclab.sh -p scripts/flapping_px4/calibrate_pure_rl_observation.py \
  --mode boundary --num-envs 64 --seed 20260806 \
  --summary <output>/boundary_summary.json --samples <output>/boundary_samples.npz
```

Isaac Sim emitted its existing headless-window, generated-schema and CPU
`powersave` governor warnings. None caused a gate failure.

## Remaining boundary

The observation engineering layer is complete for curriculum 1. Reward,
termination economics, PPO hyperparameters and convergence criteria remain
separate training-design decisions. Wind, Pitot imperfections, domain
randomization and an asymmetric critic remain deferred until nominal
straight-flight learning is demonstrated.
