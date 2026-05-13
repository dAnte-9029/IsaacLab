# Reset Trim Toward Stable Design

Goal: move the default reset state closer to the observed long-straight stable flight state so the initial altitude dip is smaller, without changing warmup semantics or controller logic.

Scope:
- Only retune reset defaults used by straight/path-tracking envs.
- Do not change `freq` limits, controller gains, or scored-start rules in this step.

Candidate parameters:
- `reset_pitch_deg`
- `reset_flap_hz`
- `reset_elevon_pitch_deg`

Success criteria:
- Reduce the minimum `height_error_m` in the first 3 seconds of long `level_straight`.
- Reduce the distance from `height_error_m <= -0.5` back to `height_error_m >= 0.0`.
- Preserve stable completion behavior in long straight.

Non-goals:
- Solving turn/loiter residuals.
- Hiding startup debt by delaying scored start until height error crosses zero.
