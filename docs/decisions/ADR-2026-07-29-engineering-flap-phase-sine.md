# ADR: Engineering Flap Phase Uses Neutral-Upstroke Sine Convention

- Status: Accepted
- Date: 2026-07-29
- Scope: Prescribed wing-stroke phase at the environment boundary
- Supersedes: the engineering-phase mapping portions of
  `ADR-2026-07-14-delaurier-prescribed-dynamic-twist.md` and
  `ADR-2026-07-14-delaurier-airflow-frame-convention.md`

## Context

The formal Isaac environment generated the prescribed stroke as
`q=Gamma*cos(phase)`, so phase zero was the positive stroke endpoint. The
flight-log and frozen DeLaurier-prior export contract instead uses
`q=Gamma*sin(phase)`, where phase zero is the neutral pose starting upstroke.
Passing a phase value between these paths without an explicit conversion
therefore introduced a 90 degree mismatch.

The mechanical convention is also the intended vehicle convention:

- phase `0`: neutral, starting upstroke;
- phase `pi/2`: positive/body-`+z` stroke endpoint;
- phase `pi`: neutral, starting downstroke;
- phase `3*pi/2`: negative/body-`-z` stroke endpoint.

## Decision

The default engineering stroke is:

```text
q     = Gamma*sin(psi)
q_dot = Gamma*omega*cos(psi)
q_ddot = -Gamma*omega^2*sin(psi)
```

Positive `q` continues to map to left URDF joint `+q` and right joint `-q`,
raising both physical wing span probes toward body-FLU `+z`.

DeLaurier's plunge expression remains
`h=-Gamma*y*cos(phi_D)`. The environment boundary therefore maps:

```text
phi_D = psi - pi/2
phi_D_dot = psi_dot
phi_D_ddot = psi_ddot
```

The default `dynamic_twist_phase_direction` remains `+1`; the default
`dynamic_twist_phase_offset_deg` becomes `-90`.

The previous `q=Gamma*cos(phase)` behavior remains available through the
explicit `legacy_cosine_endpoint_zero` configuration value. Reproducing that
baseline also requires a DeLaurier phase offset of `0 degree`.

## Consequences

- Online Isaac stroke generation and offline DeLaurier-prior export now share
  the same mechanical phase convention.
- Reset phase zero is a neutral wing pose rather than a stroke endpoint.
- The DeLaurier equations and their internal cosine plunge convention do not
  change; only the environment-to-model phase mapping changes.
- Existing trajectories, policies or plots recorded under the cosine convention
  must be labeled as legacy or converted by `psi=phi+pi/2`.

## Validation requirements

- Pure float64 tests at phase `0`, `pi/2`, `pi` and `3*pi/2` for
  `q`, `q_dot` and `q_ddot`.
- Regression coverage for the selectable legacy cosine convention.
- Engineering-to-DeLaurier phase and dynamic-twist quadrature tests with
  offset `-pi/2`.
- A real PhysX articulation pose test covering the four mechanical phase
  states, mirrored joints and motion directions.
- The existing offline exporter sine-phase contract remains unchanged.

## Reconsideration triggers

Revisit this decision if hardware encoder evidence establishes a different
zero or direction, or if prescribed joint motion is replaced by a measured
transmission state whose phase must be defined at another shaft.
