# Project State

## Stable baseline

- Branch: `flapping_rl`
- Frozen tag: `delaurier-strip-wrench-v1`
- Current implemented plant: attached-flow DeLaurier strip resultant with strip-integrated wing moment about base COM; five-surface tail `r x F` moment; PX4-like controller/task stack. `delaurier-strip-wrench-v1` records the pre-switch always-active `qd`-scaled virtual twist path, with separation disabled, `d_hat=0`, `c_mac=0`, `dM_a` enabled, `strip_integrated` moment mode, and induced drag disabled. Default near-single-rigid-body mass properties are `0.90415 kg`, CG `(-0.12154, 0.00541, -0.01298) m` in `base_link`, and diagonal inertia `(0.02329, 0.02573, 0.04270) kg m^2` about that CG.

## Current completed stage

2026-07-29 the formal environment engineering phase changed to `q=Gamma*sin(phase)`, with phase zero neutral and starting upstroke. The DeLaurier boundary now uses `phi_D=phase-pi/2`; the legacy cosine stroke remains explicitly selectable. This supersedes only the phase-mapping portions of the 2026-07-14 decisions. Targeted non-Isaac validation passed 110 tests, and the headless Isaac phase-pose and wrench-reference tests each passed. Main record: `docs/decisions/ADR-2026-07-29-engineering-flap-phase-sine.md`.

## Active stage

The PhysX wing-multibody preparation branch now uses the shared mechanical sine phase contract used by the offline DeLaurier-prior exporter. The default candidate remains `dynamic_twist_mode="disabled"` with `dynamic_twist_tip_amplitude_deg=0.0`; `legacy_qd_scaled_proxy` and `legacy_cosine_endpoint_zero` are compatibility-only. Isaac body FLU is explicitly converted to the internal DeLaurier FRD-like section convention，and physical flap `q` maps to mirrored left/right URDF joint coordinates. This candidate has not replaced the immutable `delaurier-strip-wrench-v1` tag.

## Required reading

- `docs/handoffs/2026-07-13-current-simulation-audit.md`
- `docs/architecture/current_simulation_call_chain.md`
- `docs/architecture/coordinate_frames_and_units.md`
- `docs/architecture/wing_tail_controller_interfaces.md`
- `docs/decisions/ADR-2026-07-14-delaurier-prescribed-dynamic-twist.md`
- `docs/decisions/ADR-2026-07-14-delaurier-airflow-frame-convention.md`
- `docs/decisions/ADR-2026-07-29-engineering-flap-phase-sine.md`
- `docs/plans/closed_loop_model_integration_plan.md`

## Known Issues / Deferred Work

- **Simple-QSM right-wing mirror convention:** `straight_flight_env.py` now maps the physical flap coordinate into URDF joint space as `left=+q` and `right=-q`, while the simple-QSM backend may also apply opposite left/right hinge-axis signs. Passing raw joint velocity into that backend can therefore mirror the right-wing flapping direction twice. This does not affect the current DeLaurier wing-only main path or its offline comparison. Before simple QSM is reused as an RL baseline or comparison model, the joint/hinge convention must be corrected and covered by an explicit left/right symmetry test. This issue is recorded as unresolved; it is not claimed to be fixed.
- **Legacy `FlappingBotEnv`:** `flapping_env.py::FlappingBotEnv` is an early legacy/prototype environment. Formal development uses `straight_flight_env.py`; the legacy environment has not adopted the current mirrored left/right joint-space convention and is not used for the DeLaurier model, real-data offline comparison, or subsequent formal control experiments. It remains available for historical reproduction and is considered deprecated. Before any reactivation, its joint mapping and QSM conventions must be audited and updated. This deferred work does not block the current DeLaurier wing-only offline comparison.

## Known blockers

- No corrected-force adapter or corrected strip-distribution contract.
- Corrected `Fx/Fz` ownership and moment-closure assumption require human approval.
- The corrected mirrored joint-space mapping and `theta_a` frame fix change the pre-existing plant convention；closed-loop mission baselines have not yet been rerun.
- The sine phase switch changes reset pose and all phase-indexed online results; the updated headless PhysX phase-pose contract and closed-loop mission baseline must be rerun before promotion.

## Exact next task

Review and re-freeze the closed-loop baseline after the mirrored joint-space and airflow-frame corrections；do not start corrected-model integration, tail tuning or PID tuning in the same stage.
