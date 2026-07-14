# Project State

## Stable baseline

- Branch: `flapping_rl`
- Frozen tag: `delaurier-strip-wrench-v1`
- Current implemented plant: attached-flow DeLaurier strip resultant with strip-integrated wing moment about base COM; five-surface tail `r x F` moment; PX4-like controller/task stack. `delaurier-strip-wrench-v1` records the pre-switch always-active `qd`-scaled virtual twist path, with separation disabled, `d_hat=0`, `c_mac=0`, `dM_a` enabled, `strip_integrated` moment mode, and induced drag disabled. Default near-single-rigid-body mass properties are `0.90415 kg`, CG `(-0.12154, 0.00541, -0.01298) m` in `base_link`, and diagonal inertia `(0.02329, 0.02573, 0.04270) kg m^2` about that CG.

## Current completed stage

2026-07-14 DeLaurier prescribed linear-spanwise dynamic twist implemented with default-disabled and legacy compatibility modes. Pure physics, phase and configuration validation passes; the current Isaac reference-wrench rerun is environment-blocked. Main records: `docs/decisions/ADR-2026-07-14-delaurier-prescribed-dynamic-twist.md` and `docs/aerodynamics/delaurier_strip_wrench_refactor.md`.

## Active stage

The DeLaurier attached-flow strip-wrench implementation is complete. The current branch adds explicit prescribed dynamic-twist modes. The default candidate is `dynamic_twist_mode="disabled"` with `dynamic_twist_tip_amplitude_deg=0.0`; `delaurier_linear_spanwise` implements the numerical-example phase/span relationship, while `legacy_qd_scaled_proxy` is compatibility-only. This candidate has not replaced the immutable `delaurier-strip-wrench-v1` tag. Corrected-force integration remains pending approval.

## Required reading

- `docs/handoffs/2026-07-13-current-simulation-audit.md`
- `docs/architecture/current_simulation_call_chain.md`
- `docs/architecture/coordinate_frames_and_units.md`
- `docs/architecture/wing_tail_controller_interfaces.md`
- `docs/decisions/ADR-2026-07-14-delaurier-prescribed-dynamic-twist.md`
- `docs/plans/closed_loop_model_integration_plan.md`

## Known blockers

- No corrected-force adapter or corrected strip-distribution contract.
- FLU/FRD convention at the DeLaurier boundary is unresolved.
- Corrected `Fx/Fz` ownership and moment-closure assumption require human approval.
- Current headless Isaac regression rerun is blocked before test collection by a `DerivedDataCache` lock error followed by `cuInit` crash; the 124-test pure/config/phase suite passes.

## Exact next task

Restore a clean headless Isaac startup and rerun `tests/test_delaurier_isaac_wrench_reference.py` against this dynamic-twist diff; do not start corrected-model integration, tail tuning or PID tuning.
