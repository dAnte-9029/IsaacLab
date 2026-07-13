# Closed-Loop Model Integration Plan

Date: 2026-07-13. This is a staged proposal derived from the current-code audit. It is not implementation authorization. No moment-closure algorithm is selected here.

## 1. Strip-load contract

- Objective: specify per-wing/per-strip loads, points, frames, references and aggregation invariants.
- Prerequisites: resolve authoritative FLU/FRD and base COM/reference semantics.
- Allowed files: `docs/`, then (only after approval) `source/flapping_bot/flapping_bot/physics/`, project tests.
- Prohibited files: controller, reward, mission, tail-coefficient, PID, upstream and asset files.
- Proposed API: `WingStripLoads` as described in `architecture/wing_tail_controller_interfaces.md`.
- Tests: shape/device/dtype, left/right ordering, force sum, point/reference metadata, hand-computable frame tests.
- Completion criteria: approved data contract and test plan; baseline still selectable.
- Baseline preservation: no change to default resultant force/moment.
- Human decisions: corrected-force ownership, FLU/FRD convention, CSV `dhat` authority.

## 2. Force-corrector adapter

- Objective: load/evaluate an approved corrected-force artifact without changing the default plant.
- Prerequisites: Stage 1 and an immutable artifact/schema with conventions and envelope policy.
- Allowed files: new physics adapter/config/data files, environment config wiring, focused tests/docs.
- Prohibited files: controllers, gains, tail model coefficients, rewards/missions, upstream.
- Proposed API: `CorrectedForceAdapter.predict(prior_inputs) -> correction/result with provenance`.
- Tests: schema/order/checksum/reference vectors, FRD/FLU conversion, disabled/shadow parity, finite/OOD fallback.
- Completion criteria: batched deterministic inference and diagnostics; no force application change in shadow.
- Baseline preservation: default `disabled`; disabled output matches current prior.
- Human decisions: artifact target ownership and out-of-envelope policy.

## 3. Moment-closure pure function

- Objective: define a pure, testable mapping from approved strip loads and corrected resultant to a wing moment.
- Prerequisites: Stages 1-2 and an approved closure assumption.
- Allowed files: physics data types/closure module, project tests/docs.
- Prohibited files: direct task reward/termination, controller/tail gains, upstream.
- Proposed API: `close_wing_moment(strip_loads, corrected_force_b, strategy) -> WingWrench`.
- Tests: resultant conservation, `sum(r x f)` reference equality, symmetry, zero/cancellation, no NaN, disabled parity.
- Completion criteria: approved closure identifier and explicit reference point; hand-calculable tests pass.
- Baseline preservation: equivalent-AC closure remains selectable as original baseline.
- Human decisions: closure algorithm and treatment of intrinsic aerodynamic moment.

## 4. Wing-wrench provider

- Objective: make wing force/moment production a named provider independent of environment orchestration.
- Prerequisites: Stages 1-3.
- Allowed files: physics provider/data type, environment call site, focused tests/docs.
- Prohibited files: controller architecture/gains, tail coefficients, mission/reward/asset/upstream files.
- Proposed API: `WingWrenchProvider.compute(state, wing_kinematics) -> WingWrench`.
- Tests: simple-QSM and DeLaurier provider contracts; disabled feature equivalence; CPU/CUDA/device preservation where available.
- Completion criteria: environment consumes only a complete explicit `WingWrench`.
- Baseline preservation: current DeLaurier path selected by default.
- Human decisions: whether simple-QSM needs the same strip contract now or only a compatible aggregate provider.

## 5. Environment integration

- Objective: wire the approved provider into final wrench composition and diagnostics.
- Prerequisites: Stage 4.
- Allowed files: `straight_flight_env.py`, optionally path-env inherited diagnostics, config/tests/docs.
- Prohibited files: physics formulas except provider interface, controller/tail/PID/reward/termination, upstream/assets.
- Proposed API: `final = wing_wrench + tail_wrench + drag`; one explicit external base-link wrench.
- Tests: disabled/shadow/application modes, only-approved components change, force/moment frame/reference checks, environment smoke if Isaac is available.
- Completion criteria: clear raw/corrected/applied diagnostics and no change to baseline task IDs/default mode.
- Baseline preservation: output parity when adapter disabled.
- Human decisions: whether correction is applied to wing-only or complete vehicle and how unchanged tail/drag are handled.

## 6. Plant freeze

- Objective: record one reviewed plant configuration for comparisons.
- Prerequisites: Stage 5 numerical contract and user approval.
- Allowed files: docs, explicit project configuration/manifest only after approval, tests that read it.
- Prohibited files: controller gains, tail geometry/coefficients, assets, rewards/missions during freeze task.
- Proposed API: immutable manifest containing commit, dirty-state hash, plant toggles, mass/COM/inertia, drag, tail settings and correction artifact hash.
- Tests: manifest completeness and deterministic config loading.
- Completion criteria: reviewed baseline/variant manifests.
- Baseline preservation: retain prior named baseline manifest.
- Human decisions: exact frozen comparison configuration.

## 7. Controller tuning harness

- Objective: compare fixed controller architecture/settings across frozen plant variants without tuning evaluation cases.
- Prerequisites: Stage 6.
- Allowed files: controller-tuning scripts, manifest/report docs, project tests.
- Prohibited files: plant equations, tail coefficients, reward/mission definitions, upstream.
- Proposed API: reproducible scenario/config/result manifest.
- Tests: CLI/config recording, controller reset, saturation diagnostics.
- Completion criteria: tuning scenarios and holdout scenarios separated with recorded gains.
- Baseline preservation: same architecture, constraints and objective for variants.
- Human decisions: tuning scenario set and acceptance criteria.

## 8. Straight-flight evaluation

- Objective: run frozen raw-versus-variant straight scenarios and report force, state, control and termination metrics separately.
- Prerequisites: Stages 6-7.
- Allowed files: evaluation scripts, output schemas, docs.
- Prohibited files: model/controller/tail gain edits while evaluating.
- Proposed API: result table with provenance, envelope/OOD, wrench, speed/height/attitude, effort and termination fields.
- Tests: deterministic seed/manifest/summary schema.
- Completion criteria: reviewed evidence; no claim beyond numerical/closed-loop demonstration.
- Baseline preservation: identical missions/seeds/constraints for compared variants.
- Human decisions: held-out scenario list and success gates.

## 9. Loiter evaluation

- Objective: evaluate script-loiter and/or path-mission loiter explicitly, without treating them as interchangeable.
- Prerequisites: Stage 8 acceptance or an approved exception.
- Allowed files: evaluation scripts/docs/tests.
- Prohibited files: controller/plant tuning during evaluation.
- Proposed API: route-labelled summary (`PX4LikeLoiterController` versus `PathTrackingController`) with radius, progress, altitude, saturation and termination metrics.
- Tests: route selection and summary schema.
- Completion criteria: mission-level comparison, not real-flight validation.
- Baseline preservation: same controller route per comparison.
- Human decisions: which loiter route is authoritative for the claim.

## 10. Uncertainty and sensitivity evaluation

- Objective: quantify sensitivity to correction, closure, frame assumptions, tail settings and initial/wind conditions.
- Prerequisites: reviewed straight/loiter evidence.
- Allowed files: dedicated evaluation scripts, docs, manifests and tests.
- Prohibited files: changing the frozen plant/controller during sensitivity execution.
- Proposed API: parameter/sample manifest and per-run diagnostic schema.
- Tests: reproducible sampling, bounds and aggregation.
- Completion criteria: uncertainty ranges and failure domains are reported separately from nominal metrics.
- Baseline preservation: original frozen plant included in every comparison.
- Human decisions: uncertain parameters, ranges and required confidence/coverage.
