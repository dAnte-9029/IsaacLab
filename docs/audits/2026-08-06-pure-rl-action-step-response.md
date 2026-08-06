# PureRL Direct-Action Step-Response Audit

Date: 2026-08-06

Branch: `feat/native-multibody-rl`

Baseline commit: `1200ca8df1f907a515aafde6deac03cf06d0418a`

Status: A/B fixed-root gates and C bounded free-root smoke passed. Generated
JSON and NPZ artifacts remain under `logs/` and are not committed.

## Implemented contract

The Measured PureRL task now uses:

```text
[frequency, rudder, left_elevon, right_elevon]
```

Frequency maps to `0--5 Hz`. Each tail action maps independently to the
corresponding runtime URDF joint interval after the `0.98` soft-limit factor.
The runtime interval was `[-0.7105, +0.7105] rad`, or approximately
`[-40.71, +40.71] deg`, for all three tail joints. The task uses `480 Hz`
physics and `60 Hz` policy commands.

Tail aerodynamics in this task consume actual PhysX joint positions. The
existing mixed pitch/roll action and command-angle tail-aero path remain the
default for controller-facing configurations.

## Tail-wrench scope

The URDF contains articulated `rudder`, `left_tail` and `right_tail` child
links. The aerodynamic tail model nevertheless represents the five tail
surfaces by one body-frame resultant force and one moment about base COM. That
equivalent wrench remains applied to `base_link`.

Consequently, this audit validates the whole-airframe tail force and moment,
including signs and left/right symmetry. It does not validate aerodynamic
hinge loads or an aerodynamic load-induced servo response because no tail
aerodynamic wrench is applied to the articulated tail links.

## Experiment configuration

Both fixed-root jobs used `dt=1/480 s`, command updates every eight physics
steps, 0.75 s dwell intervals and no flight controller. The outer normalized
action low-pass filter and slew-rate limit were disabled so the tests measured
the intrinsic native frequency state and PhysX implicit tail drives.

The frequency sequence was:

```text
0 -> 2.66958 -> 5 -> 2.66958 -> 0 Hz
```

Rudder, left elevon and right elevon independently used:

```text
0 -> +A -> 0 -> -A -> 0
```

with `A` equal to `10 deg`, `30 deg` and the runtime soft limit.

## A gate: fixed root without aerodynamics

Artifact summary:
`logs/flapping_rl/action_step_response/2026-08-06_no_aero_v2/summary.json`

| Channel / step | 10--90% time | 2% settling | Overshoot | Peak rate |
| --- | ---: | ---: | ---: | ---: |
| frequency, all four transitions | 0.0833 s | 0.1458 s | 0% | 69.25 Hz/s |
| each tail joint, 10 deg | 0.1208 s | 0.2146 s | 0% | 3.11 rad/s |
| each tail joint, 30 deg | 0.1208 s | 0.2146 s | 0% | 9.33 rad/s |
| each tail joint, soft limit | 0.1208 s | 0.2167 s | 0% | 10.00 rad/s |

The largest tail steady-state error was approximately `1.24e-5 rad`
(`0.00071 deg`). No soft-limit violation was observed. The `10 rad/s` peak for
the largest step agrees with the URDF velocity limit and is therefore a plant
limit, not an outer action slew setting.

Wing trajectory error remained below `0.06249 deg` and left/right
synchronization error below `1.8e-6 deg`.

## B gate: fixed root with 7 m/s aerodynamics

Artifact summary:
`logs/flapping_rl/action_step_response/2026-08-06_aero_7mps_v2/summary.json`

All frequency and tail response metrics matched A at the recorded precision.
This is expected: wing frequency is an ideal internal source and the tail
resultant wrench is applied to `base_link`, so tail aerodynamics do not load
the servo hinges. Actual joint lag still feeds forward into the aerodynamic
calculation.

The aerodynamic sign and symmetry gates passed:

- positive rudder produced `delta Fy=-0.17779 N` and
  `delta Mz=+0.06942 N m`;
- negative rudder produced `delta Fy=+0.17779 N` and
  `delta Mz=-0.06924 N m`;
- positive left and right elevon pulses produced opposed roll-moment
  increments (`+0.08207` and `-0.07551 N m`) and equal-sign pitch-moment
  increments (both approximately `+0.22357 N m`).

The small non-odd rudder drag and unequal left/right roll magnitudes are
consistent with the configured nonzero lateral COM offset and measured
left/right geometry; the primary control-effect signs are correct.

## C gate: one-second free-root smoke

Artifact summary:
`logs/flapping_rl/action_step_response/2026-08-06_free_smoke/summary.json`

The smoke started from the fixed-root wrench-balance seed and applied separate
positive 5 degree rudder/left/right pulses. All states and aerodynamic wrenches
remained finite. Maximum root linear speed was `7.18455 m/s` and maximum root
angular speed was `2.47482 rad/s`. Rudder yaw response and elevon
roll/pitch symmetry checks passed.

This is bounded one-second integration evidence. The reset candidate is not a
validated periodic orbit, and the smoke is not a straight-flight stability or
controller-performance result.

## Runtime regression

The existing 64-environment CPU PureRL runtime gate was changed from a fixed
180 policy steps to a fixed 1.5 s physical duration. At the new `60 Hz` policy
rate this is 90 steps; keeping 180 would unintentionally double the
uncontrolled flight interval from 1.5 to 3.0 s. With the corrected time basis,
the gate passed 93 forced partial resets, finite observation/reward/plant
signals, `0.05198 deg` maximum wing tracking error and approximately
`3.4e-6 deg` maximum synchronization error.

## Accepted action-shaping decision

At `60 Hz`, the policy observes roughly nine command intervals during the
frequency settling time and thirteen during a tail-servo settling time. Both
channels already have smooth intrinsic dynamics and no overshoot in this
model. An additional global `0.1 s` normalized-action LPF plus a
`2 normalized-unit/s` slew limit would therefore add a second lag layer and
would obscure the physical meaning of channel response.

The Measured PureRL configuration therefore disables both outer shapers by
setting `act_lpf_tau_s=0.0` and `act_rate_limit_per_s=0.0`, while retaining the
native `0.15 s` frequency settling model and PhysX tail drives. The inherited
`0.1 s` filter and `2 normalized-unit/s` slew defaults remain unchanged for
other configurations. If policy chatter appears, first use an action-difference
reward or measured hardware-specific per-channel limits rather than silently
restoring a global action lag. The complete frozen contract is recorded in
`docs/decisions/ADR-2026-08-06-pure-rl-direct-action-contract.md`.

## Commands

After activating `env_isaaclab`, prepending this worktree's project and asset
sources to `PYTHONPATH`, and using `./isaaclab.sh -p`:

```bash
./isaaclab.sh -p scripts/flapping_px4/validate_pure_rl_action_steps.py --mode no_aero \
  --summary logs/flapping_rl/action_step_response/2026-08-06_no_aero_v2/summary.json \
  --traces logs/flapping_rl/action_step_response/2026-08-06_no_aero_v2/traces.npz \
  --headless --device cpu

./isaaclab.sh -p scripts/flapping_px4/validate_pure_rl_action_steps.py --mode aero \
  --summary logs/flapping_rl/action_step_response/2026-08-06_aero_7mps_v2/summary.json \
  --traces logs/flapping_rl/action_step_response/2026-08-06_aero_7mps_v2/traces.npz \
  --airspeed-mps 7.0 --headless --device cpu

./isaaclab.sh -p scripts/flapping_px4/validate_pure_rl_action_steps.py --mode free_smoke \
  --summary logs/flapping_rl/action_step_response/2026-08-06_free_smoke/summary.json \
  --traces logs/flapping_rl/action_step_response/2026-08-06_free_smoke/traces.npz \
  --airspeed-mps 7.0 --duration-s 1.0 --headless --device cpu
```

Known non-fatal Isaac warnings remained the existing missing generated-schema,
headless GLFW, existing-prim and shutdown messages. All three worker summaries
reported `all_cases_accepted=true`.
