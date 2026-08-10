# PureRL Curriculum 2 Longitudinal Design

- Status: Approved
- Date: 2026-08-10
- Scope: no-wind longitudinal climb/descent curriculum on the measured CPU-native PureRL plant

## Objective

Curriculum 2 extends the successful C1 closed-loop straight-flight policy to level, climbing, and descending three-dimensional straight paths. The policy receives path geometry rather than a commanded horizontal or vertical speed. It chooses its own stable forward and vertical motion subject to path tracking, smoothness, energy, and safety rewards.

The shared `measured_pure_rl_shared_v1` action, observation, physics-step, policy-rate, frequency-governor, and actual-tail-state contracts remain unchanged. C1 rehearsal and retention evaluation are mandatory.

## Evidence used to choose the range

The three successful C1 policies produced approximately 6.5--7.3 m/s mean along-track velocity without an explicit speed target. The real-flight descriptive artifact reports train P05--P95 vertical speed of -1.28--1.55 m/s and 0.5 s altitude increments of approximately -0.61--0.60 m. The mission-setpoint artifact contains no distinct climb/descent waypoint-altitude changes, so it cannot identify a real task-angle distribution.

The existing path defaults of 3 m altitude change over 40 m correspond to a 4.29 degree flight-path angle. At the C1 speed, that geometry implies about 0.49--0.55 m/s vertical velocity and is a conservative starting point.

Cai et al. train their second curriculum on climbing and diving at variable forward and vertical speeds, with commands held for three seconds. This project reuses the staged difficulty and similar time scale, but intentionally keeps the previously approved geometry-only, no-target-speed policy contract.

## Geometry parameterization

Flight-path angle is the primary sampled difficulty variable. For horizontal path length `L` and signed flight-path angle `gamma`, altitude change is derived as:

```text
delta_h = L * tan(gamma)
```

Altitude change and length must not be sampled independently because that can create unintended steep paths.

Each non-rehearsal episode contains:

1. a 15--20 m level entry;
2. a 20--30 m constant-slope climb or descent;
3. at least 15--20 m of level recovery, followed by level continuation through the episode horizon.

The initial path heading remains uniform over `[0, 2*pi)`. Initial flap phase remains independently randomized. Wind and dynamics randomization remain disabled.

### Difficulty stages

| Stage | Sampled absolute flight-path angle | Approximate vertical speed at 6.5--7.3 m/s |
| --- | --- | --- |
| C2a | uniform from 1.5 to 4 degrees | 0.17--0.51 m/s |
| C2b | uniform from 2 to 6 degrees | 0.23--0.77 m/s |
| C2c | uniform from 2 to 8 degrees | 0.23--1.03 m/s |

Climb and descent signs are sampled with equal probability. Near-zero slope is excluded from non-rehearsal tasks because level flight is supplied explicitly by C1 rehearsal. Signed 10-degree paths are held out as extrapolation diagnostics and never drive promotion.

## Episode sampling

Each environment samples independently at reset. A seed determines a reproducible sequence and does not identify a fixed task or direction.

| Stage | C1 level rehearsal | Climb | Descent |
| --- | ---: | ---: | ---: |
| C2a | 50% | 25% | 25% |
| C2b | 30% | 35% | 35% |
| C2c | 25% | 37.5% | 37.5% |

Within a non-rehearsal task, absolute angle, entry length, and slope length are sampled uniformly inside the approved stage bounds. The distribution is fixed for a stage. Training results do not silently alter task weights or ranges.

## Three-dimensional velocity reward

For path heading `psi` and signed flight-path angle `gamma`, construct the orthonormal path basis in the z-up world frame:

```text
t_hat = [cos(gamma)*cos(psi), cos(gamma)*sin(psi), sin(gamma)]
n_h   = [-sin(psi), cos(psi), 0]
n_v   = [-sin(gamma)*cos(psi), -sin(gamma)*sin(psi), cos(gamma)]
```

Project world linear velocity `v` into:

```text
v_tangent = dot(v, t_hat)
v_lateral_normal = dot(v, n_h)
v_vertical_normal = dot(v, n_v)
```

The progress and velocity terms become:

```text
progress_reward = tanh(v_tangent / 3 m/s)
lateral_normal_reward = exp(-(v_lateral_normal / 1 m/s)^2)
vertical_normal_reward = exp(-(v_vertical_normal / 1 m/s)^2)
velocity_reward = 0.5 * (lateral_normal_reward + vertical_normal_reward)
```

The 3 m/s value is a saturation scale, not a target speed. The policy receives diminishing additional reward above that scale and is not asked to match a particular speed.

At zero flight-path angle, this basis reduces exactly to the existing C1 along-track, horizontal-normal, and world-vertical velocity components. Therefore the reward values on C1 rehearsal are unchanged. Initial C2 work keeps all other reward scales and weights unchanged to isolate the semantic change.

Path position reward continues to use horizontal cross-track and path-relative height error. Pitch retains only the existing excess-envelope penalty; the policy is not penalized for the moderate pitch needed to follow a slope.

Required telemetry adds:

- sampled task and stage;
- sampled signed slope angle;
- active path flight-path angle;
- tangent velocity;
- lateral-normal velocity;
- vertical-normal velocity.

## Evaluation grid

Evaluation is deterministic and separate from the randomized training sampler:

- flight-path angles: `0, +/-2, +/-4, +/-6, +/-8` degrees;
- headings: `0, 90, 180, 270` degrees;
- flap phases: `0, 90, 180, 270` degrees;
- level entry: 17.5 m;
- slope length: 25 m;
- episode duration: 12 s;
- no wind or dynamics randomization.

C2a evaluates `0, +/-2, +/-4` for 80 cases. C2b adds `+/-6` for 112 cases. C2c adds `+/-8` for 144 cases. Climb and descent metrics are always reported separately. Signed 10-degree cases are diagnostic-only.

## Promotion gates

Evaluate every 100 PPO iterations after a minimum 200-iteration dwell in the current stage. Promotion requires two consecutive checkpoints to pass:

- overall non-termination rate at least 95%;
- climb success rate at least 90%;
- descent success rate at least 90%;
- at least 95% of cases reach the slope end and enter recovery;
- mean absolute horizontal cross-track error no greater than 0.5 m;
- mean absolute path-relative height error no greater than 0.5 m;
- P95 absolute height error no greater than 1.5 m;
- reverse-motion fraction no greater than 1%;
- no NaN or Inf state, action, reward, or telemetry.

Frequency and tail saturation are reported in the first C2 experiment but are not new hard gates until a successful baseline establishes their distribution.

## C1 retention gate

Every C2 candidate checkpoint also runs the frozen C1 suite. Promotion requires:

- C1 success rate at least 95%;
- C1 score no more than five points below the corresponding source C1 checkpoint;
- C1 mean cross-track and height errors each no greater than `max(2 * source baseline, 0.25 m)`;
- C1 termination rate no greater than 5%.

A checkpoint that passes C2 but fails C1 retention is labeled as a forgetting failure and is not promoted. Rehearsal probability is not automatically changed within the run.

The retention matrix has checkpoint rows and C1, C2a, C2b, C2c, and signed-10-degree diagnostic columns. Each cell retains direction-specific and worst-heading/flap-phase metrics.

## Non-goals

- no target horizontal or vertical speed;
- no wind or dynamics randomization;
- no C3 turn, loiter, or composite paths;
- no automatic performance-adaptive task distribution;
- no reward-weight retuning in the initial implementation;
- no GPU-backend promotion or PPO run as part of implementation.
