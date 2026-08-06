# PureRL Direct-Surface Action and Step-Response Plan

Date: 2026-08-06

Status: completed; action contract accepted and frozen.

## Objective

Close the low-level action path before redesigning Stage 1 observations,
rewards, terminations or PPO settings. The authoritative plant remains the CPU
native measured-wing multibody model.

## Approved action contract

The Measured PureRL task uses four normalized actions:

```text
[flap_frequency, rudder, left_elevon, right_elevon]
```

- flap frequency maps from `[-1, 1]` to `0--5 Hz`;
- rudder and each elevon map independently to the corresponding runtime URDF
  joint limit after the existing `0.98` soft-limit factor;
- physics remains `480 Hz` and the policy/action cadence becomes `60 Hz`
  (`decimation=8`);
- the tail aerodynamic model consumes actual PhysX rudder/elevon joint
  positions in this PureRL mode;
- the established controller and task configurations retain the existing
  `[frequency, rudder, elevon_pitch, elevon_roll]` action contract and
  command-angle tail-aero path.

The tail model produces one resultant force and one resultant moment about the
base COM. This equivalent tail wrench remains applied to `base_link`. The URDF
rudder and elevons are articulated child links, so this contract does not
represent aerodynamic hinge load or aerodynamic back-reaction on the implicit
joint drives.

## Step-response experiment

The experiment disables the outer normalized-action low-pass filter and slew
limit so it measures the intrinsic native frequency state and PhysX tail
servos.

### A. Fixed root, aerodynamics disabled

- physics `480 Hz`, policy commands updated at `60 Hz`;
- frequency sequence `0 -> 2.66958 -> 5 -> 2.66958 -> 0 Hz`;
- each tail surface independently receives
  `0 -> +A -> 0 -> -A -> 0`, with `A` equal to `10 deg`, `30 deg`, and the
  runtime URDF soft limit;
- record command, actual state, rise/fall/settling time, overshoot,
  steady-state error, maximum rate, wing trajectory error and left/right wing
  synchronization.

### B. Fixed root, representative aerodynamics

Repeat A with `7 m/s` body-forward air-relative flow, wing DeLaurier loads and
tail aerodynamics enabled. Record the tail force and moment about base COM in
addition to the actuator signals. Validate finite outputs, rudder sign and
left/right elevon mirror relations.

The comparison is a loaded whole-airframe aerodynamic calculation, not a
servo hinge-load test.

### C. Free-root bounded smoke

From the reproducible reset seed, run small direct-surface pulses for at most
one second. Acceptance means finite state, finite aerodynamic wrench, bounded
root rates and the expected initial response sign. It does not claim stable
flight or a trim orbit.

## Outputs and decision boundary

The runner writes a provenance-rich JSON summary and compressed NPZ traces and
refuses to overwrite existing outputs. Generated runs stay under `logs/`.

The accepted Measured PureRL configuration keeps the tested settings:
`act_lpf_tau_s=0.0` and `act_rate_limit_per_s=0.0`. The native frequency state
and PhysX tail drives remain the only actuator dynamics. Any future added
shaping must be an explicit superseding decision, must be rerun through A and
B, and must not duplicate those intrinsic dynamics.

## Completion criteria

1. Existing mixed-action baselines remain selectable and unchanged by default.
2. Pure tests cover limit mapping, inverse mapping, interface selection and
   response-metric calculations.
3. A and B complete in fresh CPU Isaac processes with finite traces.
4. C remains finite and bounded for its declared short horizon.
5. The report clearly separates whole-airframe wrench validation from actuator
   hinge-load claims.
