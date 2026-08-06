# ADR: Freeze the PureRL curriculum-1 reward and termination contract

- Status: Accepted
- Date: 2026-08-06
- Scope: `FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg`
- Reference: Cai et al., *Learning-based Trajectory Tracking for Bird-inspired Flapping-Wing Robots*

## Context

Curriculum 1 must learn stable straight flight without receiving or tracking a
target forward speed and without a fixed trim controller. The inherited
straight-flight reward tracks both a `vx` command and a fixed pitch command.
Its absolute normalized-action penalty also has the wrong semantics for the
new direct action: zero normalized frequency means `2.5 Hz`, and a physically
necessary nonzero rudder or elevon trim would be penalized continuously.

Cai et al. combine position, angular-rate, balance and energy terms and keep
the reward structure fixed across their curriculum. Their position target is
time-parameterized and therefore contains an implicit speed objective. This
curriculum instead uses an unbounded geometric line and must separate forward
progress from path error.

## Decision

Only the Measured PureRL task selects the curriculum-1 reward. Historical
controller-facing environments retain the inherited reward and termination.
The new implementation is a standalone pure Tensor builder; the environment
only supplies route-relative state and applied-action history.

The positive terms are:

```text
0.40 * path_reward
+ 0.20 * progress_reward
+ 0.15 * velocity_reward
+ 0.15 * roll_reward
+ 0.10 * angular_rate_reward
```

`path_reward` is the average of Gaussian rewards for cross-track error with a
`1.5 m` scale and height error with a `1.0 m` scale. `progress_reward` is
`tanh(v_along / 3 m/s)`: reverse motion is negative and forward reward
saturates without defining a target speed. `velocity_reward` averages Gaussian
rewards for cross-track and vertical velocity, each with a `1 m/s` scale.
Roll uses a `20 degree` Gaussian scale. Root-body angular rate uses an
isotropic `3 rad/s` exponential scale.

The penalties are:

```text
- 0.03 * pitch_envelope_penalty
- 0.04 * flap_penalty
- 0.01 * frequency_action_delta_penalty
- 0.01 * tail_action_delta_penalty
- 0.02 * tail_action_limit_penalty
```

Pitch has no target inside `+-30 degrees`; excess pitch is smoothly penalized
with a `15 degree` scale. The flap penalty is the explicit proxy
`(actual_frequency / 5 Hz)^3`, not a claim of electrical or motor-shaft power.
Action-delta terms use consecutive applied policy actions and normalize a full
`-1` to `+1` jump to one. The tail-limit term is zero through `80 percent` of
normalized authority and rises smoothly to one at a hard limit. Constant
nonzero tail trim has no action-delta cost.

Termination occurs on ground contact, tilt above `75 degrees`, absolute
route-relative cross-track error above `3 m`, or absolute height error above
`3 m`. Tilt is computed from normalized projected gravity using `acos(-g_b.z)`
so an inverted vehicle cannot evade termination. Timeout remains separate.

Measured PureRL telemetry is enabled by default and reports every raw reward
term, weighted contribution, path and velocity state, actual frequency, and
each termination-cause fraction through `extras["log"]`. Telemetry can be
disabled explicitly without changing the reward.

## Alternatives considered

1. Retain the inherited `vx` and pitch targets. This would prescribe speed and
   trim instead of allowing the policy and plant to establish them.
2. Copy the paper's time-indexed position tracking. This introduces an implicit
   speed command that is absent from the approved observation and task.
3. Penalize absolute normalized action. This selects the action coordinate
   origin rather than physical energy and discourages required asymmetric
   trim.
4. Use the native inverse-dynamics power estimate directly. That quantity is a
   common-coordinate multibody estimate, not exact motor or electrical power,
   and has not been validated as a learning objective.
5. Add a constant alive reward or low-speed termination. The selected dense
   terms and loss of future return already reward survival, while a low-speed
   threshold would become another implicit speed target.

## Consequences

- Forward flight is encouraged without selecting one equilibrium speed.
- Pitch and asymmetric tail trim remain emergent within explicit safety
  envelopes.
- High actual flap frequency, action chatter and repeated hard-limit use are
  separately observable and tunable.
- Reward values are inspectable outside Isaac Sim, preserving batch, dtype and
  device behavior.
- The selected scales and weights are initial curriculum parameters, not a
  claim that PPO convergence or sim-to-real performance has been established.

## Assumptions

- The geometric line, commanded height and initial forward velocity provide a
  sufficient curriculum-1 task definition without a speed command.
- A cubic frequency proxy has the right monotonic energy preference for the
  first learning gate even though it is not calibrated power.
- Three-meter path-error and 75-degree tilt limits provide enough exploration
  room while removing clearly failed states.
- Root angular velocity and route-relative ground velocity are appropriate
  stability signals for the no-wind first curriculum.

## Validation requirements

- Pure tests must cover monotonic path shaping, saturating signed progress,
  pitch-envelope behavior, cubic actual-frequency cost, zero delta cost for a
  constant nonzero trim, soft tail-limit cost and all termination causes.
- Static tests must verify that only Measured PureRL selects the new contract.
- The 64-environment CPU native runtime gate must retain finite rewards,
  observation/mechanism guarantees and all required telemetry keys.
- Fixed-action, random-action and short multi-seed PPO gates remain separate
  follow-up experiments and must not be reported as completed by this ADR.

## Reconsideration triggers

Reconsider the contract if diagnostic rollouts show one weighted contribution
dominating by an order of magnitude, the policy exploits progress while
leaving the path, the frequency proxy drives systematic stall or maximum-rate
flapping, required control repeatedly enters the last 20 percent of tail
authority, or short PPO cannot improve survival and route error across seeds.
Any scale or weight change must retain per-term telemetry and record the
evidence that motivated it.
