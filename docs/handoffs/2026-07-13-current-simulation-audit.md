# Stage Handoff

## 1. Stage identity

2026-07-13 current flapping simulation code audit on branch `flapping_rl`, commit `300045bf9a3ebbd338f1fd4f165016d8e6ccc0d8`.

## 2. Objective

Record the implemented wing/tail/controller/task call chain and identify the smallest safe interface needed before corrected-force and moment-closure work.

## 3. Work completed

- Read required repository instructions, Git state, relevant code, scripts, tests and existing planning documents.
- Traced current DeLaurier wing force/resultant moment, five-surface tail wrench, PX4-like control and task paths.
- Ran 32 pure physics/controller tests successfully.
- Wrote the audit, architecture, coordinate/interface and proposed-plan documents.

## 4. Documents created or updated

- `docs/audits/2026-07-13-current-simulation-audit.md` — evidence and findings.
- `docs/architecture/current_simulation_call_chain.md` — runtime diagrams.
- `docs/architecture/coordinate_frames_and_units.md` — frame/reference inventory.
- `docs/architecture/wing_tail_controller_interfaces.md` — current and proposed boundaries.
- `docs/plans/closed_loop_model_integration_plan.md` — staged future work.
- `docs/PROJECT_STATE.md` — concise state index.

## 5. Key observed facts

- DeLaurier returns aggregate force and zero moment; environment uses per-wing equivalent quarter-chord `r x F` about `root_com_pos_w`.
- Tail sums five surface forces and `r x F` moments about supplied base COM.
- `FlappingBotPathTrackingEnv` inherits the straight environment plant.
- `fly_path_mission.py` uses the environment teacher once per `env.step`; separate loiter script uses `PX4LikeLoiterController`.
- Current source has no corrected-force/moment-closure runtime implementation.

## 6. Important inferred behavior

- The base-link local Isaac wrench is treated as the aerodynamic body-frame wrench, but project code lacks a direct assertion of that frame identity.
- Current wing torque is an equivalent-point closure rather than an intrinsic DeLaurier moment.

## 7. Unresolved questions

- Corrected `Fx/Fz` ownership: wing-only versus complete vehicle.
- Approved moment-closure assumption.
- Authoritative FLU/FRD conversion and `dhat` use.
- Frozen plant configuration for later comparisons.

## 8. Tests and commands executed

- `git status --short`; `git branch --show-current`; `git rev-parse HEAD`; `git log -5 --oneline`.
- The focused wrapper pytest command in the audit report: 32 passed in 2.26 s.
- No Isaac Sim smoke, rollout, sweep, training or rendering was run.

## 9. Current limitations

- No strip-load output means no force-consistent strip moment can yet be constructed.
- The DeLaurier FRD comment conflicts with surrounding FLU declaration.
- Existing effective-force/moment plan text is not evidence of a current runtime path.

## 10. Exact recommended next task

Design and approve the strip-load contract; do not yet integrate a corrected model, moment closure, tail tuning or PID tuning.

## 11. Allowed scope for next task

Read current physics/environment/tests and write design/ADR/contract documentation. Source modification only after the contract and human decisions are approved.

## 12. Prohibited scope for next task

No corrected-model integration, closure implementation, tail coefficient changes, PID tuning, reward/observation/reset/termination changes, task-ID changes, asset/upstream edits, or commits without separate authorization.

## 13. Required reading for next conversation

1. `AGENTS.md`, `docs/AGENTS.md`, physics and PX4-like nested `AGENTS.md`.
2. `docs/PROJECT_STATE.md`.
3. This handoff.
4. `docs/audits/2026-07-13-current-simulation-audit.md`.
5. `docs/architecture/coordinate_frames_and_units.md` and `wing_tail_controller_interfaces.md`.
