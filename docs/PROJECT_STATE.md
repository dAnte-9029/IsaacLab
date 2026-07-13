# Project State

## Stable baseline

- Branch: `flapping_rl`
- Commit audited: `300045bf9a3ebbd338f1fd4f165016d8e6ccc0d8`
- Current implemented plant: DeLaurier aggregate wing resultant plus equivalent-AC `r x F` wing moment; five-surface tail `r x F` moment; PX4-like controller/task stack. Default near-single-rigid-body mass properties are `0.90415 kg`, CG `(-0.12154, 0.00541, -0.01298) m` in `base_link`, and diagonal inertia `(0.02329, 0.02573, 0.04270) kg m^2` about that CG.

## Current completed stage

2026-07-13 formal current-simulation audit completed, followed by approval of the measured whole-aircraft mass-property baseline and the DeLaurier strip-wrench refactor. Main records: `docs/audits/2026-07-13-current-simulation-audit.md` and `docs/aerodynamics/delaurier_strip_wrench_refactor.md`.

## Active stage

The DeLaurier attached-flow strip-wrench implementation is complete. Corrected-force integration remains pending approval.

## Required reading

- `docs/handoffs/2026-07-13-current-simulation-audit.md`
- `docs/architecture/current_simulation_call_chain.md`
- `docs/architecture/coordinate_frames_and_units.md`
- `docs/architecture/wing_tail_controller_interfaces.md`
- `docs/plans/closed_loop_model_integration_plan.md`

## Known blockers

- No corrected-force adapter or corrected strip-distribution contract.
- FLU/FRD convention at the DeLaurier boundary is unresolved.
- Corrected `Fx/Fz` ownership and moment-closure assumption require human approval.

## Exact next task

Design and approve the corrected-force distribution contract on top of the implemented DeLaurier strip loads; do not integrate a corrected model, tail tuning or PID tuning.
