# Coordinate Frames, Units and Reference Points

Date: 2026-07-14. `Observed` rows are direct code evidence. `Unresolved` is deliberately not filled with guesses.

## Frame definitions and conventions

| Frame/reference | Current evidence | Status |
|---|---|---|
| World `w` | root position/velocity use `root_pos_w`, `root_lin_vel_w`; gravity is `(0,0,-9.81)` | Observed (`straight_flight_env.py:170-175`) |
| Aerodynamic/body `b` | Tail module declares right-handed `x forward, y left, z up`; environment calls `root_lin_vel_b`, `root_ang_vel_b` aerodynamic inputs | Observed (`tail_aero.py:15`; `straight_flight_env.py:1248-1252`) |
| DeLaurier section `D` | Right-handed `x` forward, `y` right, `z` down; `v_D=diag(1,-1,-1)v_FLU` | Observed (`delaurier_airflow.py`; ADR-2026-07-14) |
| Wing co-rotating/Wang `c` | `x` span, `y` normal, `z` chordwise toward LE | Observed (`qsm_delaurier1993.py:81-85`) |
| Engineering flap phase | `_phase` increases in the positive direction; `q=Gamma*cos(_phase)`. Positive `q` maps to left joint `+q` and right joint `-q`, raising both real span probes toward body `+z` | Observed (`startup_phase.py`; `test_delaurier_isaac_phase_pose_contract.py`) |
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

`quat_from_euler_xyz`, `quat_apply` and `quat_apply_inverse` are used throughout. Isaac articulation data explicitly document `body_link_quat_w` as `(w,x,y,z)`；robot initial quaternion `(1,0,0,0)` is identity (`articulation_data.py:body_link_quat_w`; `isaaclab_assets/.../flapping_bot.py`).

## Key tensor inventory

| Tensor/symbol | Shape | Unit | Frame/reference | Transform/notes | Status |
|---|---:|---|---|---|---|
| `root_pos_w` | `(N,3)` | m | world | subtract `scene.env_origins` for local mission position | Observed |
| `root_lin_vel_w` | `(N,3)` | m/s | world | ground velocity; wind is world frame | Observed |
| `root_lin_vel_b` | `(N,3)` | m/s | body/base local assumed | used as aerodynamic velocity before wind subtraction | Observed frame label; exact equality to base-link local unresolved |
| `root_ang_vel_b` | `(N,3)` | rad/s | body | tail point velocity and controllers | Observed |
| `wind_w`, `wind_b` | `(N,3)` | m/s | world/body | `wind_b=quat_apply_inverse(root_quat_w,wind_w)` | Observed |
| `v_air_b` | `(N,3)` | m/s | body | `root_lin_vel_b-wind_b` | Observed |
| `v_air_delaurier` | `(N,3)` | m/s | DeLaurier section `D` | FLU polar vector converted by `diag(1,-1,-1)` | Observed |
| `theta_a_env` | `(N,)` | rad | DeLaurier section incidence | `atan2(v_D.z, clamp(v_D.x))`; positive for vehicle air-relative velocity toward body-FLU `-z` | Observed |
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
- DeLaurier batches repeat the same physical engineering phase and scalar `q/qd/qdd` for left/right. URDF joint targets/velocities are mirrored as left `(+q,+qd)` and right `(-q,-qd)` because both revolute axes are joint `+x` while mesh spans are local `+y/-y`. Prescribed dynamic twist uses the same local Wang `+x` pitch sign on both sides; no additional right-wing aerodynamic scalar sign is applied.
- Tail mixing defines left/right differential signs in the environment (`left=pitch+roll`, `right=pitch-roll`), and both elevon surfaces use `deflection_sign=-1` (`straight_flight_env.py:1156-1184`; `tail_aero.py:264-303`).
- Pure strip-wrench tests verify polar/axial parity。`test_delaurier_isaac_phase_pose_contract.py` additionally verifies real mirrored link origins, span probes, chord/span directions, axial pitching directions and motion at four phases; `test_delaurier_isaac_wrench_reference.py` verifies the articulation wrench boundary.

## Phase-to-pose contract

Real PhysX pose validation uses a `0.65 m` representative span point in each real wing link and body FLU coordinates. With `Gamma=20 deg`:

| `phase` | left/right URDF joint | probe `z_b` | physical interpretation |
|---:|---|---:|---|
| `0` | `(+0.349066,-0.349066) rad`；zero rate | `+0.210428 m` | positive-body-`z` stroke endpoint |
| `pi/2` | approximately `(0,0)`；velocity `(-1.745329,+1.745329) rad/s` | `-0.012604 m` | midpoint moving toward body `-z`，downstroke |
| `pi` | `(-0.349066,+0.349066) rad`；zero rate | `-0.234115 m` | negative-body-`z` stroke endpoint |
| `3pi/2` | approximately `(0,0)`；velocity `(+1.745329,-1.745329) rad/s` | `-0.012604 m` | midpoint moving toward body `+z`，upstroke |

左右 probe 的 `x/z` 在 `2e-5` tolerance 内相等，`y` 等幅反号。该结果确认 `phi_D=current_phase`，因此 `dynamic_twist_phase_direction=1`、offset `0` 保持不变。

## Airflow and angle contract

Isaac/project body 使用 FLU，DeLaurier section 使用 FRD-like axes：

```text
v_D = diag(1,-1,-1) v_FLU
theta_a = atan2(w_D,u_D) = atan2(-v_FLU.z,v_FLU.x)
theta = theta_a + theta_w + delta_theta
```

因此水平来流 `theta_a=0`；vehicle air-relative velocity 具有 body-FLU `-z` 分量时 `theta_a>0`，具有 `+z` 分量时 `theta_a<0`。`theta_w` 和 `delta_theta` 均沿 Wang `+x` 正 pitch 定义，不在左右翼入口增加额外经验 sign。

## Identified ambiguities

1. DeLaurier current section reduction只使用 forward/down components；large sideslip 对 section incidence 的处理尚未定义。
2. DeLaurier CSV `dhat` is read but active geometry uses `dhat=0`; whether the discarded field matters is unresolved.

## Required future assertions/tests

- Assert `r` and `F` are in the same frame before every `cross(r,F)` closure.
- Test left/right mirrored DeLaurier loads yield expected net roll/yaw symmetry.
- Retain phase-map tests at top stroke, mid downstroke, bottom stroke and mid upstroke if stroke generation changes.
- Re-run mission baselines after the corrected mirrored joint-space mapping；pose integration passing does not by itself establish closed-loop performance.
