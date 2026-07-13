w

# Current Flapping Simulation Audit

## 1. Audit metadata

- Date: 2026-07-13
- Branch: `flapping_rl`
- Commit: `300045bf9a3ebbd338f1fd4f165016d8e6ccc0d8` (`feat: add measured DeLaurier flight config`)
- Working-tree status at audit start: `M AGENTS.md`; `A docs/AGENTS.md`; `A scripts/flapping_px4/AGENTS.md`; `A source/flapping_bot/AGENTS.md`; `AM source/flapping_bot/flapping_bot/physics/AGENTS.md`; `A source/flapping_bot/flapping_bot/px4_like/AGENTS.md`; untracked `artifacts/` and `docs/papers/ocr/`. These pre-existing changes were not modified by this audit.
- Audit scope: `source/flapping_bot/flapping_bot/{physics,px4_like,direct,path_tracking,config}`, `scripts/flapping_px4`, task registration, the required minimal Isaac Lab wrench API, and relevant tests/docs.
- Files intentionally excluded: upstream Isaac Lab implementation except `Articulation.set_external_force_and_torque`; non-project upstream code; asset/URDF/USD/mesh contents; generated `artifacts/`; full GPU/Isaac Sim rollouts; training and parameter sweeps.
- Required context result: root and nested `AGENTS.md` files listed by the task were found and read. `docs/PROJECT_STATE.md` was missing, therefore no active-handoff reference existed. There are no existing `docs/architecture/`, `docs/audits/`, or `docs/handoffs/` documents to read. Relevant existing plans were read.

## 2. Executive summary

**Observed.** The active DeLaurier plant is assembled in `FlappingBotStraightFlightEnv._apply_action`: it computes air-relative base velocity, calls five-surface tail aerodynamics and either simple QSM or DeLaurier wing aerodynamics, sums their body-frame wrenches, and buffers the net wrench with `Articulation.set_external_force_and_torque(..., body_ids=base_link, is_global=False)` (`straight_flight_env.py:1246-1317`). `FlappingBotPathTrackingEnv` inherits this plant unchanged.

**Observed.** DeLaurier itself returns aggregate force only: `F_c=[0,sum(dN),sum(dF_x)]` and `tau_c=0` (`physics/qsm_delaurier1993.py:212-215`). The environment supplies the present wing moment by treating each left/right aggregate force as applied at an area-weighted quarter-chord link point, then computing world-frame `r x F` about `root_com_pos_w` (`straight_flight_env.py:1404-1421`). No strip force or strip point is returned at this boundary.

**Observed.** Tail force and moment are more explicit: each of five surfaces computes local velocity at its aerodynamic center, force, and `cross(r_b, force_b)` about the supplied base COM, then the model sums them (`physics/tail_aero.py:423-473`, `519-535`).

**Observed.** The current source tree contains no runtime `effective_force`, `replace_fx_fz`, `replace_my`, `WingWrench`, or moment-closure implementation. Existing June plans describe such an intended or historical flow, but those plans are not source evidence that this checkout executes it.

## 3. Repository evidence reviewed

- Plant integration/configuration: `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`, especially `FlappingBotStraightFlightEnvCfg`, `_pre_physics_step`, `_apply_action`, `_compute_wing_delaurier_wrench`, and `_reset_idx`.
- Path task: `source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py`, `PX4LikePathTrackingController`, `PathManager`, and mission primitives.
- Wing models: `physics/qsm_delaurier1993.py`, `qsm.py`, `wing_geom_csv.py`, and `wing_equivalent_ac.py`.
- Tail model: `physics/tail_aero.py`.
- Controllers: `px4_like/{straight_line_controller,loiter_controller,path_tracking_controller,tecs,guidance}.py` and `controller_tuning_profiles.py`.
- Runtime scripts: `fly_straight_line.py`, `fly_loiter.py`, `fly_path_mission.py`, `run_tail_balance_grid_search.py`, `run_path_tracking_baseline_suite.py`, and `run_straight_height_recovery_sweep.py`.
- Registration/API: `direct/flapping_bot/__init__.py`, `source/isaaclab/isaaclab/assets/articulation/articulation.py:962-1012`, and `source/isaaclab_assets/isaaclab_assets/robots/flapping_bot.py`.
- Existing planning context: `docs/plans/2026-06-23-effective-force-model-freeze-and-isaac-integration-design.md`, `...integration.md`, and `docs/plans/2026-06-25-pitch-moment-fit-ood-zero-design.md`.

## 4. Current runtime entry points

| Purpose                            | Entry and task/env                                                                                                         | Controller/plant                                                                                                                                        | Outputs                                                                                         | Evidence status                                                        |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| Straight flight controller run     | `./isaaclab.sh -p scripts/flapping_px4/fly_straight_line.py --task Isaac-FlappingBot-StraightFlight-DeLaurier-Direct-v0` | Script constructs`PX4LikeStraightLineController`; task resolves to `FlappingBotStraightFlightEnv` with `FlappingBotStraightFlightDeLaurierEnvCfg` | timestamped`logs/flapping_px4/straight_line/.../trajectory_env0.csv`, `summary.json`        | Observed (`fly_straight_line.py:443-525,946-1119`)                   |
| Loiter/circle controller run       | `./isaaclab.sh -p scripts/flapping_px4/fly_loiter.py --task Isaac-FlappingBot-StraightFlight-DeLaurier-Direct-v0`        | Script constructs`PX4LikeLoiterController`; same straight environment/DeLaurier plant                                                                 | `logs/flapping_px4/loiter/.../trajectory_env0.csv`, `summary.json`                          | Observed (`fly_loiter.py:510-637,794-804,1018-1021`)                 |
| Fixed/random path mission          | `./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0`    | `FlappingBotPathTrackingEnv`; stateful `PX4LikePathTrackingController` is the environment teacher                                                   | `logs/flapping_px4/path_tracking/<mission>/<timestamp>/trajectory_env0.csv`, `summary.json` | Observed (`fly_path_mission.py:894-896,1004-1103,1388-1391`)         |
| Tail balance/trim search           | `run_tail_balance_grid_search.py` invokes path mission rollouts with tail-effectiveness overrides                        | same path task/plant; it ranks rollout summaries, not a separate tail model                                                                             | timestamped search root with CSV/JSON/Markdown summaries                                        | Observed (`run_tail_balance_grid_search.py:318-319,558-592,905-944`) |
| Controller tuning / recovery sweep | `run_straight_height_recovery_sweep.py`; command construction targets path mission                                       | TECS/controller arguments are swept; this is an experiment driver                                                                                       | `/tmp/.../sweep_summary.{json,csv}` by default                                                | Observed (`run_straight_height_recovery_sweep.py:89-94,203-212`)     |
| Baseline evaluation suite          | `run_path_tracking_baseline_suite.py`                                                                                    | calls path-mission cases; no new plant calculation                                                                                                      | timestamped suite manifest and per-case rollout directories                                     | Observed (`run_path_tracking_baseline_suite.py:41-64,91-195`)        |

No executed evidence was found for a separate current `tail trim` environment. The tail balance script is the current trim-like workflow. `FlappingBotEnv`/simple QSM remains package-exported but is not the default task used by the above PX4 scripts.

## 5. Wing-force computation chain

### Active DeLaurier chain

1. **State and wind — Observed.** `_apply_action` reads `root_quat_w`, `root_lin_vel_b`, and `root_ang_vel_b`; it transforms world wind into body coordinates with `quat_apply_inverse` and forms `v_air_b = v_b - wind_b` (`straight_flight_env.py:1248-1252`). Thus ground velocity is `root_lin_vel_w`, while aerodynamic input is intended as air-relative `root_lin_vel_b`.
2. **Kinematics — Observed.** `_pre_physics_step` maps normalized action zero to clamped flap frequency; `_apply_action` advances phase and constructs cosine `q_cmd`, `qd_cmd`, and `qdd_cmd` (`straight_flight_env.py:1143-1146,1196-1224`). The same commanded kinematics are used for both wings.
3. **Two-wing batching — Observed.** `_compute_wing_delaurier_wrench` repeats each environment twice in left/right order, creates spanwise `y`, computes `h=-q*y`, and prepares prescribed twist (`straight_flight_env.py:1328-1379`).
4. **DeLaurier inputs — Observed.** The routine derives `theta_a=atan2(v_air_b.z, clamp(v_air_b.x))`, `theta_bar`, and `U=clamp(v_air_b.x, min_airspeed)`, then calls `compute_aero_wrench_delaurier1993` (`straight_flight_env.py:1339-1398`). It does not pass lateral velocity, body angular velocity, or per-strip rigid-body point velocity into that function.
5. **Strip calculation and aggregation — Observed.** `compute_aero_wrench_delaurier1993` broadcasts `(B,N)` strip fields, computes `alpha`, `alpha_dot`, attached/separated normal loads and chordwise loads, then sums them into `F_c` (`qsm_delaurier1993.py:94-220`). With `return_terms=False` as called by the environment, strip arrays are not returned.
6. **Coordinates/force application — Observed.** Constant Wang-to-link matrices differ for left/right span mirroring (`straight_flight_env.py:646-651`); link force is rotated to world using each wing-link quaternion. The aggregate resultant per wing is placed at an area-weighted quarter-chord point in that link (`1400-1409`), summed, converted to root/body coordinates, and applied as a net base-link wrench (`1415-1444`, `1313-1317`).

### Simple-QSM baseline chain

`QuasiSteadyWingModel.compute_forces` accepts per-wing joint position/velocity plus body-frame root linear/angular velocity, derives a per-wing relative velocity, lift/drag, and `cross(lever, wing_force)` (`physics/qsm.py:82-202`). The environment selects this path only when `use_delaurier_wings=False` (`straight_flight_env.py:1283-1294`). Its `root_lin_vel` docstring says “body/world frame”, but the active caller supplies `v_air_b`; this ambiguity is recorded below.

## 6. Current wing-moment computation

- **Observed: DeLaurier intrinsic output.** The model explicitly sets `tau_c = zeros_like(F_c)` (`qsm_delaurier1993.py:212-215`). Although `dM_ac` and `dM_a` occur in its input-power calculation, they are not aggregated into returned torque (`196-220`). Therefore current DeLaurier does not directly output a nonzero aerodynamic moment.
- **Observed: environment resultant moment.** `FlappingBotStraightFlightEnv` calculates one force resultant for each wing, applies it at `_wing_application_point_link`, and computes `tau_w=cross(p_wing_w-root_com_pos_w,F_w)` (`straight_flight_env.py:1404-1416`). It therefore uses aggregate, not strip-level, force.
- **Observed: equivalent point.** `compute_area_weighted_quarter_chord_link_points` weights strip area `c*dx`, returns `left=(quarter_chord_x,+span_center,0)` and `right=(quarter_chord_x,-span_center,0)` in wing-link coordinates (`physics/wing_equivalent_ac.py:12-27`). It is used only on the DeLaurier initialization path (`straight_flight_env.py:628-651`).
- **Observed: reference.** The cross product is about `self._robot.data.root_com_pos_w`, then transformed by `quat_w` into the root/body frame. It is not documented as a wing-root or body-origin moment.
- **Observed: simple-QSM alternative.** The simple model calculates each `cross(lever_arm_body,force_b)` directly (`physics/qsm.py:193-200`); its configured levers are treated as body-frame moment arms but no explicit COM correction is supplied.
- **Inferred.** Because the final API receives a force plus torque for `base_link`, the DeLaurier torque is intended to be an equivalent base-body torque. Whether Isaac's local base-link frame is exactly the aerodynamic “body FLU” frame has not been asserted in project code.

## 7. Tail-force and tail-moment computation

1. **Controller/action mapping — Observed.** Action ordering is `[flap_frequency, rudder, elevon_pitch, elevon_roll]`; pitch/roll are mixed as `left=trim+pitch+roll`, `right=trim+pitch-roll`, then clamped to softened joint limits (`straight_flight_env.py:1143-1184`). Thus symmetric command is the average of left/right; differential command is half their difference (`1183-1184`).
2. **Actuation limits/rates — Observed.** The environment clamps normalized command to `elevon_max_deg=41`, `rudder_max_deg=25`, intersects with articulation limits, applies optional action low-pass and normalized action slew limit (`117-119`, `1129-1141`, `1176-1181`). Kinematic override writes joint state every physics step by default (`216`, `1233-1237`); no separate physical tail-servo rate model is used in this path.
3. **Tail local flow — Observed.** For every surface, `v_point=root_lin_vel_b + cross(root_ang_vel_b,r_b)`; it removes the spanwise velocity, determines AoA from rotated chord versus incoming flow, clamps AoA, and computes finite-wing lift/drag (`tail_aero.py:435-471`).
4. **Coefficients/empirical scales — Observed.** `TailSurfaceCfg` exposes area, geometry, `cl_alpha_per_rad`, `cd0`, `cd_k`, alpha limit and sign. Default lift slopes derive from a finite-wing formula; `cd0=0.02`, lift/efficiency constants and 25-degree limit are module constants (`tail_aero.py:53-58,215-244,349-381`). `TailAeroCfg` additionally exposes horizontal incidence, fixed/elevon effectiveness, elevon alpha limit and horizontal dynamic-pressure scale (`368-381`). Env defaults override two effects to `0.5` and `1.2` (`straight_flight_env.py:234-241,616-624`).
5. **Tail moment — Observed.** If `base_com_pos_b` is passed, every surface changes its base-origin AC arm to an arm relative to base COM, then sets `torque_b=cross(r_b,force_b)` (`tail_aero.py:435-473`). `compute_wrench` sums fixed-horizontal, split elevons, fixed-vertical and rudder wrenches (`475-535`); no extra intrinsic section moment exists.
6. **Virtual moments — Observed.** The environment can add `virtual_roll_moment_gain*q_dyn*roll_cmd-damping*p` and analogous pitch terms after tail moment computation (`straight_flight_env.py:1266-1278`). Both gains/dampings default to zero (`247-255`), so they are latent configuration coupling rather than default force-model behavior.
7. **Coupling — Observed.** Tail parameters are configured independently of wing coefficients, but wing and tail share `qsm_wings.air_density` in environment-level virtual moment and parasite-drag calculations. Controller code does not import or mutate tail coefficients.

## 8. PX4-like controller chain

**Observed.** The common four-action cascade is implemented by `PX4LikeStraightLineController.compute_actions` and duplicated/adapted by loiter/path controllers.

`line/circle/path query -> DirectionalGuidance -> airspeed-direction heading control -> lateral acceleration -> roll_sp -> elevon_roll`; simultaneously `altitude, altitude_rate, airspeed, bank/load factor -> PX4LikeTECS -> pitch_sp and throttle_sp -> flap frequency`; then `pitch_sp -> filtered pitch/rate PD+I -> elevon_pitch`, and `yaw-heading_from_airspeed plus yaw-rate damping -> rudder` (`straight_line_controller.py:344-603`; `loiter_controller.py:59-296`; `path_tracking_controller.py:30-288`).

- **Active loops.** Straight script owns a `PX4LikeStraightLineController`; loiter script owns a `PX4LikeLoiterController`; path-mission evaluation lets the environment execute its `PX4LikePathTrackingController` exactly once inside `env.step` (`fly_path_mission.py:1100-1111`). An environment-local straight controller exists only when `teacher_guidance_enabled` is true (`straight_flight_env.py:653-676`).
- **Reset.** `reset()` resets TECS and clears filtered pitch, rate, prior action and inner integrator state (`straight_line_controller.py:221-228`); environments call it during reset (`straight_flight_env.py:1579-1581`).
- **Integrator/anti-windup.** TECS contains pitch/throttle integrators guarded at output limits (`tecs.py:341-348,407-446`). The pitch inner loop leaks, clamps its integral, and blocks integration when raw output is saturated (`straight_line_controller.py:520-554`).
- **Saturation/rate controls.** Roll/pitch output is clamped to `[-1,1]`; roll/pitch command rates may be limited. TECS clamps pitch/throttle and applies optional filters/slew limits (`straight_line_controller.py:440-456,540-567`; `tecs.py:434-446`).
- **Gain provenance.** Dataclass defaults are controller profiles. External straight/loiter scripts apply `apply_controller_tuning_profile` based on truth/estimated state source; path teacher parameters are materialized in `FlappingBotPathTrackingEnvCfg` (`controller_tuning_profiles.py:37-63`, `path_tracking_env.py:580-675`).
- **Not active by default.** `PX4LikeLoiterController` is not called by `FlappingBotPathTrackingEnv`; path missions use `PX4LikePathTrackingController` with `PathManager` query. Conversely, a controller class existing in `px4_like` is not proof that a given task calls it.
- **Plant coupling.** The controller reads state/reference/wind and configured flap limits; it does not import DeLaurier or `TailAeroCfg`. Its gains may nevertheless be empirically tuned against the current tail/plant behavior; that is an experiment dependency, not a code-level coefficient dependency.

## 9. Straight-flight task chain

`fly_straight_line.py -> parse_env_cfg/task registry -> FlappingBotStraightFlightEnv -> external PX4LikeStraightLineController -> env.step(actions) -> _pre_physics_step -> _apply_action -> wing+tail net wrench -> base_link external wrench -> observation/reward/termination/log CSV+JSON`.

The base environment owns state collection, model orchestration, action mapping, wrench buffering, observation history, reward and terminations (`straight_flight_env.py:1090-1317,1585-1707`). It also contains explicit numerical/experiment stabilization mechanisms: reset freeze, kinematic appendage overrides, appendage mass scaling/redistribution, optional COM/inertia/total mass overrides, optional induced/parasite drag, action filters and latent virtual moments (`130,212-255,311-318,678-800`).

## 10. Loiter / circle-flight task chain

There are two distinct routes.

1. **Script loiter — Observed.** `fly_loiter.py` builds `PX4LikeLoiterController`, calls `navigate_circle`, and steps the straight environment. Its plant is the same straight environment and it records a loiter trajectory/summary (`fly_loiter.py:510-637,794-804`).
2. **Path-mission loiter — Observed.** `fly_path_mission.py` injects `Mission`/`PathManager`; `FlappingBotPathTrackingEnv` queries closest point, tangent, curvature and height; its teacher calls `PX4LikePathTrackingController.compute_actions_from_query`. Reward and termination add loiter radial/progress metrics (`path_tracking_env.py:947-1077,1278-1333,1512-1657`).

**Inferred.** These routes share the same underlying force application, but they are not the same controller entry point or reference construction. Results must identify which route was used.

## 11. Environment responsibilities and hidden coupling

### Observed responsibilities

- State collection and wind conversion: `straight_flight_env.py:1248-1252`.
- Model calls and wrench composition: `1246-1317`.
- Observation/reward/reset/termination: `1449-1707` and path overrides.
- Teacher action orchestration: `1021-1041`; path teacher overrides in `path_tracking_env.py:1278-1333`.
- Runtime diagnostics are limited to net wing/tail/net force/torque caches, executed action caches and controller diagnostic cache (`straight_flight_env.py:471-475,543-547`).

### Observed embedded model/stability content

The environment is not only a thin orchestration layer. It owns DeLaurier frame mapping, equivalent AC placement, induced/parasite drag, virtual tail moments, mass/COM/inertia overrides, forced initial hold, and action filtering. These are explicit in configuration, not hidden, but they affect plant behavior and must be logged for model comparisons.

## 12. Existing tests and validation

### Executed

```bash
PYTHONDONTWRITEBYTECODE=1 ./isaaclab.sh -p -m pytest -p no:cacheprovider -q \
  tests/test_delaurier_convention_contract.py tests/test_wing_equivalent_ac.py \
  tests/test_tail_aero.py tests/test_straight_line_controller.py \
  tests/test_px4_loiter_controller.py tests/test_px4_path_tracking_controller.py
```

Result: **32 passed in 2.26 s**. This covers DeLaurier convention expectation, equivalent AC arithmetic, tail surface signs/COM reference/compatibility fields, and pure controller behavior. It does not instantiate Isaac Sim or validate flight physics.

### Located but not run

- Physics/environment contracts: `test_straight_flight_env_reset_contract.py`, `test_path_tracking_env_contract.py`, `test_flapping_task_registration.py`.
- Script/evaluation contracts: `test_fly_straight_line_contract.py`, `test_fly_path_mission.py`, `test_eval_suites.py`, `test_tail_balance_grid_search.py`.
- Existing test inventory includes reset, mission, path, evaluator, controller-profile and suite-contract coverage. Most are unit/contract tests; named rollout scripts require Isaac Sim/GPU and write output directories.

Not run: Isaac Sim environment smoke, straight/loiter/path rollout, parameter sweep, training, and rendering. They are outside this read-only audit's low-cost validation scope and could create simulation outputs.

### Clear missing tests

- No current test found that reconstructs DeLaurier aggregate force/moment from exported per-strip loads (none are exported).
- No current test found for the complete final wing+tail+drag wrench frame/reference contract.
- No current source/test found for corrected-force, moment-closure, `replace_fx_fz`, or `replace_my` runtime contracts.
- No current test found that asserts DeLaurier's `theta_a` frame conversion against the environment FLU declaration.

## 13. Findings

### Blocking

#### F-01 — Corrected-force and moment-closure runtime interface is absent

- Severity: Blocking for corrected-force/moment-closure development; not a blocker for the baseline audit.
- Evidence: source-wide search found no runtime `effective_force`, `replace_fx_fz`, `replace_my`, correction artifact, or closure helper. Current final composition is direct `f_w_sum + f_tail + f_drag` and `tau_w_sum + tau_tail` (`straight_flight_env.py:1307-1317`).
- Consequence: there is no stable hook through which a corrected resultant or complete wing wrench can be inserted and no diagnostics for raw/corrected/applied quantities.
- Blocks development: Yes, any corrected model integration.
- Recommended follow-up: approve a strip-load contract first; do not choose a closure algorithm or edit plant code in this audit stage.

#### F-02 — DeLaurier strip data is discarded before wing wrench closure

- Severity: Blocking for force-consistent strip-based moment closure.
- Evidence: strip quantities exist only as locals in `compute_aero_wrench_delaurier1993` and it returns aggregate `F_c`, zero `tau_c`, power and separation; environment calls it with `return_terms=False` (`qsm_delaurier1993.py:212-262`, `straight_flight_env.py:1382-1398`).
- Consequence: a later closure cannot distinguish left/right or span/chord load redistribution from the existing public call result.
- Blocks development: Yes, for a strip-resolved closure. It does not block retaining the present single-equivalent-point baseline.
- Recommended follow-up: define exact per-wing/per-strip force, point, frame and reference metadata before implementation.

#### F-03 — Corrected `Fx/Fz` ownership is undecided

- Severity: Blocking for interface design.
- Evidence: old plan language describes a complete-vehicle `Fx/Fz` replacement, while current composition separates wing, tail and parasite drag (`docs/plans/2026-06-23-effective-force-model-freeze-and-isaac-integration-design.md`; `straight_flight_env.py:1299-1317`).
- Consequence: assigning a complete-vehicle correction to a wing-only closure would double-count or incorrectly attribute tail/drag unless an allocation rule is approved.
- Blocks development: Yes.
- Recommended follow-up: user must decide whether the correction target is wing-only, complete vehicle, or an explicitly allocated residual.

### Important

#### F-04 — Frame declaration conflict around DeLaurier AoA

- Severity: Important; blocks trustworthy FRD/FLU corrected-model integration until tested.
- Evidence: tail code declares body `x forward, y left, z up` (`tail_aero.py:15`), reset comments use the same convention (`straight_flight_env.py:132-145`), while DeLaurier input comment says `v_air_b.z` is “body-down” and calls it a flight-log FRD convention without conversion (`straight_flight_env.py:1343-1348`).
- Consequence: the sign of angle of attack and any FRD/FLU artifact feature conversion is ambiguous from code evidence.
- Blocks development: Yes for corrected-model integration; no for documenting current call path.
- Recommended follow-up: add hand-computable FLU/FRD convention tests and name conversion functions explicitly.

#### F-05 — Current wing moment is an equivalent-point closure, not a strip/load or intrinsic moment model

- Severity: Important.
- Evidence: `tau_c` is zero and environment uses one area-weighted quarter-chord point per wing (`qsm_delaurier1993.py:212-215`; `wing_equivalent_ac.py:12-27`; `straight_flight_env.py:1404-1421`).
- Consequence: it cannot support a claim that DeLaurier's aerodynamic pitching/rolling/yawing moment is fully represented. The result is a geometric closure assumption.
- Blocks development: No for baseline; yes for claiming a physically validated wing moment.
- Recommended follow-up: retain this as a named baseline closure and compare proposed alternatives only after approval.

#### F-06 — CSV `dhat` values are read but not used by active DeLaurier geometry construction

- Severity: Important design limitation.
- Evidence: `load_wing_geom_csv` reads `dhat` (`wing_geom_csv.py:27-46`); `build_wing_geometry_from_csv` ignores `_dhats` and sends its `dhat` argument (`99-127`); environment calls it with `dhat=0.0` (`straight_flight_env.py:628-635`).
- Consequence: equivalent quarter-chord point is based on zero pitch-axis offset regardless of CSV column value.
- Blocks development: No, unless the approved closure requires CSV pitch-axis fidelity.
- Recommended follow-up: decide whether the CSV `dhat` column is authoritative and add a contract test before changing behavior.

#### F-07 — Existing effective-force/moment plans conflict with current implementation status

- Severity: Important documentation/state conflict.
- Evidence: plans describe runtime modes and files such as `effective_force_correction.py`, while source and tests named by those plans are absent from this checkout. The design document's “Runtime Data Flow” is therefore not confirmed current implementation.
- Consequence: a future agent could treat planned behavior as active baseline.
- Blocks development: No, provided code remains authoritative; it blocks using plans as runtime evidence.
- Recommended follow-up: `PROJECT_STATE.md` now indexes this audit and marks strip-load contract as the single next task.

### Minor

#### F-08 — Simple-QSM root linear velocity frame is ambiguously documented

- Severity: Minor.
- Evidence: `compute_forces` doc says `root_lin_vel` is “body/world frame” (`physics/qsm.py:89-101`); active environment passes air-relative body velocity (`straight_flight_env.py:1248-1252,1286-1292`).
- Consequence: future callers could supply a world vector without an error.
- Blocks development: No.
- Recommended follow-up: tighten docstring/type-name and add a frame test in a future approved source change.

## 14. Unresolved human decisions

1. Are corrected `Fx/Fz` targets wing-only, complete-vehicle, or a specified component allocation?
2. Which moment-closure family is acceptable? No algorithm is selected by this audit.
3. Is the wing geometry CSV `dhat` authoritative for aerodynamic-center placement?
4. What is the authoritative FLU/FRD conversion and quaternion convention at the Isaac/body/artifact boundary?
5. Which baseline configuration (including tail effectiveness, COM/mass override, drag and virtual-moment settings) is frozen for comparison?

## 15. Current supported claims

- The code implements a batched DeLaurier force resultant and a five-surface tail force/moment model.
- The baseline environment applies a net base-link external wrench in Isaac Lab local-link mode.
- Current DeLaurier wing moment is generated by an equivalent application-point `r x F` closure, not returned directly by DeLaurier.
- PX4-like straight, loiter, and generic-path controller paths exist with saturation, reset and diagnostic logic.
- The listed 32 pure physics/controller tests passed in the stated environment.

## 16. Claims not supported by current implementation

- Complete six-axis or strip-resolved wing aerodynamic moment fidelity.
- Existence of a currently active corrected-force, `replace_fx_fz`, `replace_my`, or moment-closure runtime path.
- Aerodynamic validation from controller compensation, tail-effectiveness tuning, or a simulation rollout alone.
- Real-flight validation, sim-to-real evidence, or physical accuracy from this code audit.
- Frame-correct FRD/FLU artifact integration without an explicit convention test.

## 17. Recommended next stage

Design and approve the **strip-load contract** only. It must state the exact shapes, frames, application-point/reference semantics, diagnostic fields, disabled-feature baseline parity and tests. Do not integrate a corrected model, choose a moment-closure algorithm, retune tail parameters, or retune PID gains in that stage.
