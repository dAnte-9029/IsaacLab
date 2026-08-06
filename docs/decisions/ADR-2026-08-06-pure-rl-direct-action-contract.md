# ADR: Freeze the Measured PureRL direct-action contract

- Status: Accepted
- Date: 2026-08-06
- Scope: `FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg`
- Evidence: `docs/audits/2026-08-06-pure-rl-action-step-response.md`

## Context

The first PureRL curriculum needs a policy interface that matches the physical
aircraft: one flapping-frequency command and three independently actuated tail
surfaces. The historical controller-facing interface instead combines the two
elevons into pitch and roll channels. It also applies a global normalized-action
low-pass filter and slew limit before the plant dynamics.

CPU step-response gates show that the measured native plant already contains
smooth dynamics: the frequency state settles in approximately `0.146 s`, while
the PhysX tail joints settle in approximately `0.215 s` without overshoot. A
second global filter and rate limit would obscure these channel-specific plant
responses and would make normalized action rate mean different physical rates
for frequency and angular commands.

## Decision

The Measured PureRL action vector is frozen as:

```text
[flap_frequency, rudder, left_elevon, right_elevon]
```

The four policy outputs are normalized to `[-1, 1]`. Flapping frequency maps
to `0--5 Hz`. Each tail-surface command maps independently to its runtime URDF
joint interval after the `0.98` soft-limit factor. Actual frequency remains a
separate plant state from target frequency.

Physics runs at `480 Hz`; the policy supplies one action every eight physics
steps at `60 Hz`. The Measured PureRL configuration sets
`act_lpf_tau_s=0.0` and `act_rate_limit_per_s=0.0`. The native `0.15 s`
frequency settling model, PhysX implicit tail drives and URDF joint velocity
limits provide the actuator dynamics.

Tail aerodynamics consume actual PhysX rudder and elevon joint positions. The
tail model continues to apply its equivalent resultant force and moment about
base COM to `base_link`; aerodynamic hinge loads are outside this contract.

The historical controller-facing configurations retain their mixed
`[frequency, rudder, elevon_pitch, elevon_roll]` action, command-angle tail
aerodynamics, `0.1 s` outer filter and `2 normalized-unit/s` slew limit. The
periodic-trim compatibility script explicitly selects that historical mixed
interface and command-angle source.

## Alternatives considered

1. Retain pitch/roll elevon mixing. This is convenient for the existing flight
   controller but does not match the three independently commanded physical
   servos and introduces an avoidable projection at the action boundary.
2. Retain the global low-pass filter and slew limit. This adds substantial
   dynamics on top of the measured channel responses and gives one normalized
   rate parameter unlike physical meanings across all four actions.
3. Apply tail aerodynamic loads to the articulated tail links. This would
   expose hinge loads and servo back-reaction, but requires a separate tail
   load-transfer and servo-model validation stage and is not necessary to
   freeze the policy interface.

## Consequences

- The policy controls the same four independent commands available on the
  physical vehicle.
- Frequency and tail commands retain distinct, inspectable plant dynamics.
- Action chatter is not hidden by a global command filter. Training must first
  address it through an explicit action-difference cost; hardware-specific
  channel limits may be added later with new evidence.
- Tail aerodynamic force and moment affect whole-aircraft motion, but the
  simulation does not claim aerodynamic hinge-load fidelity.
- Existing controller experiments and historical trim artifacts remain
  reproducible through their explicit mixed-action compatibility path.

## Assumptions

- The native frequency settling state is an adequate first approximation of
  the flapping drive response over `0--5 Hz`.
- The URDF joint position and velocity limits, implicit-drive parameters and
  the `0.98` soft-limit factor represent the current tail servo envelope well
  enough for the first curriculum.
- A `60 Hz` policy rate is sufficient to observe and control actuator responses
  whose measured settling times are approximately `0.15--0.22 s`.
- Aerodynamic hinge loads are not required for the first straight-flight
  learning stage.

## Validation requirements

- Pure contract tests must verify the direct channel order, frequency range,
  runtime URDF mapping, inverse mapping, actual-joint aerodynamic source,
  `480/60 Hz` cadence and disabled PureRL action shapers.
- Baseline contract tests must verify that the mixed interface and inherited
  action-shaping defaults remain unchanged.
- Fixed-root no-aero and aero step-response gates must remain finite, obey soft
  limits and preserve the expected control-effect signs.
- A bounded free-root smoke and the 64-environment CPU runtime gate must remain
  finite after configuration changes.

## Reconsideration triggers

Reconsider this decision if hardware measurements show materially different
frequency or tail-servo dynamics, policy chatter remains unacceptable after an
explicit action-difference cost is tuned, the policy cadence changes, hinge
loads become necessary for fidelity, or a different actuator interface is
required by the flight hardware. Any change to action order, normalization or
shaping requires a superseding ADR and new A/B step-response evidence.
