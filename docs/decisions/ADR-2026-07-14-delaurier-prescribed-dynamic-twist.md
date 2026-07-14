# ADR: DeLaurier Prescribed Dynamic Twist

- Status: Accepted
- Date: 2026-07-14
- Scope: DeLaurier wing kinematic input only

## Context

The environment historically supplied a full-span-uniform twist proxy whose amplitude was proportional to commanded flap rate `qd`. That proxy is not the prescribed dynamic twist used in the DeLaurier numerical example and obscures the relationship among stroke phase, span position and pitch derivatives. The no-dynamic-twist prior must remain the default baseline.

## Decision

Add an explicit `dynamic_twist_mode` with three mutually exclusive values:

- `disabled`（default）；
- `delaurier_linear_spanwise`；
- `legacy_qd_scaled_proxy`（compatibility only）。

For the DeLaurier mode, use

```text
delta_theta = -theta_tip * (y/R) * sin(phi_D)
delta_theta_dot = -theta_tip * (y/R) * cos(phi_D) * phi_D_dot
delta_theta_ddot = theta_tip * (y/R)
                    * (sin(phi_D) * phi_D_dot^2 - cos(phi_D) * phi_D_ddot)
theta = theta_bar + delta_theta
```

The current environment stroke is `q=Gamma*cos(current_phase)` and its plunge input is `h=-q*y`. Therefore the approved mapping is `phi_D=current_phase`, represented explicitly by direction `+1` and offset `0 rad`. The environment currently supplies zero phase acceleration within a physics step; the pure helper accepts non-zero acceleration.

Use `WingGeometry.R` as the authoritative geometric semi-span. If a caller has no explicit `R`, infer it from `max(y_i + 0.5*strip_width_i)`, never from the last strip center. `theta_tip` refers to the theoretical point `y=R`.

## Alternatives considered

- Keep the `qd`-scaled proxy as the main model: rejected because it is spanwise uniform and is not the DeLaurier numerical-example relationship.
- Remove the legacy proxy immediately: rejected to preserve reproducibility of the frozen `delaurier-strip-wrench-v1` behavior.
- Enable a non-zero twist by default: rejected because `theta_tip` has not been experimentally identified.
- Solve passive aeroelastic twist: outside this decision and requires structural parameters and validation.

## Consequences

- Default behavior remains `theta=theta_bar`, `theta_dot=theta_ddot=0`.
- A user must select `delaurier_linear_spanwise` and provide a tip amplitude to enable the new model.
- All DeLaurier force, moment, power and angle diagnostics consume the same generated `theta`, `theta_dot` and `theta_ddot` tensors.
- Historical behavior remains available only through an explicitly named legacy mode.

## Assumptions

- The numerical-example phase relationship is applicable as a prescribed kinematic prior.
- Left and right DeLaurier batches use the same scalar stroke phase and pitch convention; existing Wang-to-link reflection handles polar/axial parity.
- `theta_bar` is constant in time over the dynamic-twist derivative calculation.
- Instantaneous environment frequency is constant within one physics step, so `phi_D_ddot=0` at the current integration boundary.

## Validation requirements

- Exact disabled and zero-tip regression in float64.
- Span normalization and theoretical-tip semantics.
- Analytical derivative checks against independent centered finite differences.
- Non-zero phase-acceleration term.
- Engineering-phase to DeLaurier-phase quadrature cases.
- End-to-end changes in strip loads, integrated wrench and power.
- Left/right mirror and existing free-couple power-sign regression.
- Isaac reference-wrench regression to confirm the downstream application interface is unchanged.

## Reconsideration triggers

Revisit this decision if measured spanwise twist identifies a different shape or phase relationship, if a passive structural model replaces prescribed kinematics, or if the authoritative joint/wing frame convention changes.
