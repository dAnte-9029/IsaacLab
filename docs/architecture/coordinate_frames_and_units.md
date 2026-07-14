# Coordinate Frames, Units and Reference Points

Date: 2026-07-13. `Observed` rows are direct code evidence. `Unresolved` is deliberately not filled with guesses.

## Frame definitions and conventions

| Frame/reference | Current evidence | Status |
|---|---|---|
| World `w` | root position/velocity use `root_pos_w`, `root_lin_vel_w`; gravity is `(0,0,-9.81)` | Observed (`straight_flight_env.py:170-175`) |
| Aerodynamic/body `b` | Tail module declares right-handed `x forward, y left, z up`; environment calls `root_lin_vel_b`, `root_ang_vel_b` aerodynamic inputs | Observed (`tail_aero.py:15`; `straight_flight_env.py:1248-1252`) |
| Flight-log FRD | A DeLaurier comment calls `v_air_b.z` “body-down” and FRD, but no conversion is made | Unresolved/conflicting (`straight_flight_env.py:1343-1348`) |
| Wing co-rotating/Wang `c` | `x` span, `y` normal, `z` chordwise toward LE | Observed (`qsm_delaurier1993.py:81-85`) |
| Engineering flap phase | `_phase` increases in the positive direction; `q=Gamma*cos(_phase)`, so phase zero is positive maximum stroke with zero rate | Observed (`straight_flight_env.py:_apply_action`) |
| DeLaurier phase `phi_D` | `phi_D=+current_phase+0`; this makes `h=-q*y=-Gamma*y*cos(phi_D)` | Observed/derived from implemented equations (`resolve_delaurier_phase`, `_compute_wing_delaurier_wrench`) |
| Wing-link `l` | Environment maps Wang axes to left/right wing link with two hard-coded matrices | Observed (`straight_flight_env.py:646-651`) |
| Tail surface | No persistent separate frame; span/chord axes and AC arms are declared in `TailSurfaceCfg` body coordinates | Observed (`tail_aero.py:349-364`) |
| Isaac articulation local | `set_external_force_and_torque(..., is_global=False)` applies in each body's local link frame | Observed (`articulation.py:962-1012`) |
| Body origin | Tail `lever_arm_body` doc says base origin; wing simple-QSM levers are body frame | Observed (`tail_aero.py:353-356`; `qsm.py:17-25`) |
| Base COM | Tail receives `body_com_pos_b`; DeLaurier closure uses `root_com_pos_w` | Observed (`straight_flight_env.py:1257-1265,1411-1414`) |
| Wing root | `DeLaurierStripWrench.moment_wang_about_wing_origin` is about the wing-root pitching-axis origin | Observed (`qsm_delaurier1993.py:DeLaurierStripWrench`) |
| Strip point | Normal forces use quarter/mid-chord points; chordwise forces use pitching-axis point by default | Observed (`integrate_delaurier_strip_wrench`) |
| Equivalent wing AC | area-weighted quarter chord in wing-link frame | Observed (`wing_equivalent_ac.py:12-27`) |

## Quaternion convention

`quat_from_euler_xyz`, `quat_apply` and `quat_apply_inverse` are used throughout. Robot initial quaternion is `(1,0,0,0)` (`isaaclab_assets/.../flapping_bot.py:53-57`). The inspected project code does not define component ordering or a standalone algebraic convention; this is an **Unresolved** project-level convention, even though Isaac Lab may define it upstream.

## Key tensor inventory

| Tensor/symbol | Shape | Unit | Frame/reference | Transform/notes | Status |
|---|---:|---|---|---|---|
| `root_pos_w` | `(N,3)` | m | world | subtract `scene.env_origins` for local mission position | Observed |
| `root_lin_vel_w` | `(N,3)` | m/s | world | ground velocity; wind is world frame | Observed |
| `root_lin_vel_b` | `(N,3)` | m/s | body/base local assumed | used as aerodynamic velocity before wind subtraction | Observed frame label; exact equality to base-link local unresolved |
| `root_ang_vel_b` | `(N,3)` | rad/s | body | tail point velocity and controllers | Observed |
| `wind_w`, `wind_b` | `(N,3)` | m/s | world/body | `wind_b=quat_apply_inverse(root_quat_w,wind_w)` | Observed |
| `v_air_b` | `(N,3)` | m/s | body | `root_lin_vel_b-wind_b` | Observed |
| `_phase`, `_freq` | `(N,)` | rad, Hz | scalar | advanced per physics step; controls prescribed wing motion | Observed |
| `q_cmd`, `qd_cmd`, `qdd_cmd` | `(N,)` | rad, rad/s, rad/s² | joint scalar | cosine waveform | Observed |
| DeLaurier strip fields `h,...,theta...` | `(B,N_strip)` | m/m·s⁻¹/m·s⁻²/rad/rad·s⁻¹/rad·s⁻² | co-rotating calculation | `B=2*N_env`; inputs to strip-load calculation | Observed |
| `DeLaurierTwistKinematics.theta`, derivatives and deltas | `(B,N_strip)` | rad, rad/s, rad/s² | Wang pitching scalar about `+x` | mean pitch plus optional prescribed linear-spanwise dynamic twist | Observed |
| `DeLaurierTwistKinematics.span_fraction` | `(B,N_strip)` | 1 | wing-root span coordinate | `y/R`; environment uses explicit `WingGeometry.R` | Observed |
| `DeLaurierTwistKinematics.phase`, rate, acceleration | `(B,1)` | rad, rad/s, rad/s² | DeLaurier scalar phase | current environment uses direction `+1`, offset `0`, acceleration `0` | Observed |
| `DeLaurierStripLoads` components | `(B,N_strip)` | N or N·m | Wang co-rotating | attached-flow raw components, strip width included | Observed |
| `DeLaurierStripWrench.force_wang`, `moment_wang_about_wing_origin` | `(B,3)` | N, N·m | Wang co-rotating | moment about wing-root pitching-axis origin | Observed |
| legacy `F_c`, `tau_c` | `(B,3)` | N, N·m | Wang co-rotating | compatibility wrapper; `tau_c` remains zero | Observed |
| `F_l`, `F_w`, `F_b` | `(B,3)`/`(N,3)` | N | link/world/body | Wang-to-link, link-to-world, world-to-body | Observed |
| `p_wing_w`, `r_w` | `(B,3)` | m | world | AC point; `r_w=p_wing_w-root_com_pos_w` | Observed |
| `tau_w` | `(B,3)` then `(N,3)` | N·m | world then body | `cross(r_w,F_w)` | Observed |
| tail `r_b` | `(N,3)` | m | base/body relative to COM when provided | AC arm | Observed |
| tail `force_b`, `torque_b` | `(N,3)` | N, N·m | body, about base COM | `cross(r_b,force_b)` | Observed |
| final `f_sum`, `t_sum` | `(N,1,3)` | N, N·m | base-link local assumed | applied to `base_link`, `is_global=False` | Observed / exact frame relation unresolved |
| controller actions | `(N,4)` | normalized | action space | `[frequency,rudder,elevon_pitch,elevon_roll]`, each clamped `[-1,1]` | Observed |

## Force and moment conventions

- Tail lift/drag and simple-QSM outputs are code-labelled body frame. Tail force is computed at surface AC; tail torque is about supplied base COM.
- DeLaurier raw normal/chord components and free pitching couples are initially co-rotating/Wang-frame strip loads. `strip_integrated` is the environment default and translates the integrated wing-root wrench to `root_com_pos_w`.
- `legacy_fixed_quarter_chord` retains the previous zero-moment compatibility wrapper and resultant-force closure about `root_com_pos_w`.
- Final force/torque are separately passed to a `base_link` local-frame Isaac API. No position argument is given; torque carries the selected reference effect.

## Left/right mirror conventions

- Equivalent AC returns `+span_center` for left and `-span_center` for right (`wing_equivalent_ac.py:25-26`).
- Wang-to-link mapping mirrors right span (`straight_flight_env.py:648-651`).
- DeLaurier batches repeat the same engineering phase and scalar `q/qd/qdd` for left/right. Prescribed dynamic twist uses the same local Wang `+x` pitch sign on both sides; no additional right-wing scalar sign is applied.
- Tail mixing defines left/right differential signs in the environment (`left=pitch+roll`, `right=pitch-roll`), and both elevon surfaces use `deflection_sign=-1` (`straight_flight_env.py:1156-1184`; `tail_aero.py:264-303`).
- The pure strip-wrench test verifies polar/axial parity under the current mirrored Wang-to-link matrices and static left/right symmetry; a full Isaac articulation reference test is still absent.

## Identified ambiguities

1. FLU versus FRD is contradictory at the DeLaurier `theta_a` calculation; code uses no conversion.
2. Quaternion component order and transform direction are not declared in project documentation.
3. The relationship among root frame, base-link frame, `root_com_pos_w`, and aerodynamic body reference is assumed but not asserted.
4. DeLaurier CSV `dhat` is read but active geometry uses `dhat=0`; whether the discarded field matters is unresolved.

## Required future assertions/tests

- Hand-computable FLU/FRD tests for `theta_a`, force and `My` signs.
- Assert quaternion order/direction at the project boundary.
- Assert `r` and `F` are in the same frame before every `cross(r,F)` closure.
- Test left/right mirrored DeLaurier loads yield expected net roll/yaw symmetry.
- Retain phase-map tests at top stroke, mid downstroke, bottom stroke and mid upstroke if stroke generation changes.
- Extend the current pure strip force/moment-conservation tests to an Isaac articulation one-force reference test.
- Test final base-link wrench frame/reference against a one-force Isaac articulation case.
