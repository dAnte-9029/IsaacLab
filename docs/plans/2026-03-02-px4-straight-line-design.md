# PX4-Control-Law Straight-Line Baseline (DeLaurier)

## Goal
Build a non-SITL baseline where DeLaurier flapping dynamics are controlled by a PX4-inspired fixed-wing control law for one straight-line mission segment.

## Scope
- Use `Isaac-FlappingBot-StraightFlight-DeLaurier-Direct-v0` as the simulation backend.
- Keep robot visual model unchanged.
- Add a virtual roll channel in dynamics so action space becomes:
  - `[freq, elevator, rudder, roll]`
- Implement NPFG-like line guidance and heading-to-lateral-acceleration control in Python.
- Run closed-loop evaluation and log metrics (`cross-track`, `course error`, resets).

## Control Stack
1. **Line projection**: project current position onto mission line.
2. **Directional guidance**: compute `course_sp` + `lateral_accel_ff`.
3. **Heading loop**: map heading error to `lateral_accel_fb`.
4. **Roll target**: `roll_sp = atan((lateral_accel_fb + lateral_accel_ff) / g)`.
5. **Actuator mapping**:
   - `roll_sp` -> roll action channel
   - height/pitch loop -> elevator channel
   - course/yaw damping loop -> rudder channel
   - trim (or optional speed-hold) -> flapping frequency channel

## Validation Plan
1. Smoke test environment with 4-action interface.
2. Run `scripts/flapping_px4/fly_straight_line.py` on one env.
3. Inspect `summary.json`:
   - mean/p95 absolute cross-track error
   - mean/p95 absolute course error
   - reset count
4. Tune gains iteratively using trajectory CSV.
