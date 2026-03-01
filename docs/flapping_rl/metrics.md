# FlappingBot Straight-Flight RL Metrics

The straight-flight tasks are set up as command-tracking problems with a small action space:

- Action: `[flap_frequency, elevator, rudder]` in normalized `[-1, 1]`
- Primary commands:
  - `vx_cmd` (body-frame forward speed)
  - `height_cmd` (world-frame z)

## Success Criteria (Practical)

- Forward speed tracking: `|vx - vx_cmd|` stays bounded without oscillation.
- Altitude tracking: `|z - height_cmd|` stays bounded (no slow sink or runaway climb).
- Lateral stability: small `y` drift and small `vy`.
- Attitude stability: tilt stays away from termination threshold.

## What To Plot

- `vx(t)` vs `vx_cmd`
- `z(t)` vs `height_cmd`
- `y(t)` and `vy(t)`
- Tilt proxy: `sqrt(g_bx^2 + g_by^2)` (from `projected_gravity_b`)
- Action traces: flap frequency, elevator, rudder

## Interpreting Rewards

The reward mixes:

- `r_vx`: forward speed tracking
- `r_height`: height tracking
- `r_tilt`: upright attitude

with penalties on lateral drift, angular rates, and action magnitude.

For paper-quality reporting, treat reward as a training signal, and report physical metrics (`vx`, `z`, drift, tilt)
on held-out seeds or fixed command settings.

