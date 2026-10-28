# Bird-Scale Flapping-Wing Research on Isaac Lab

## Research objective

This fork supports a bird-scale flapping-wing aircraft, its aerodynamics and multibody plant, PX4-like control, and RL-based flight control. Project code primarily lives in `source/flapping_bot/`, `scripts/flapping_px4/`, project tests, and `docs/`.

Prioritize the scientific objective stated by the user. Approved work may substantially replace a physics model, controller, observation space, reward formulation, or training architecture. Do not force major ideas into small backwards-compatible patches, or invent extra validation stages and ablations.

## Relevant context only

- Read this file plus nested `AGENTS.md` files on paths actually being worked on. Do not scan and read every agent file in the tree.
- Consult `docs/PROJECT_STATE.md` and the specific active handoff/ADR when the task depends on current plant or RL assumptions. Do not automatically read all referenced history before unrelated tasks.
- Inspect relevant code and the current working-tree status before edits. Do not inspect old commits or repeatedly reread unchanged files unless resolving a concrete uncertainty.
- Preserve user changes, avoid unrelated refactors, and do not perform Git history-changing operations unless asked.

## Code boundaries and scientific integrity

- Prefer implementing project-specific functionality under `source/flapping_bot/` and `scripts/flapping_px4/`, not modifying upstream Isaac Lab. If a research design genuinely requires an upstream change, explain its impact and implement it within the authorized task rather than silently settling for an inferior workaround.
- Keep plant parameters, aerodynamic coefficients, mass/inertia, and controller parameters distinguishable. A better tracked trajectory does not by itself validate a physical model.
- Preserve coordinate frames, force/moment reference points, wing geometry/kinematics, actuator signs, and simulation timestep assumptions explicitly when relevant.
- In this fork, CPU-native multibody dynamics and GPU approximate or commanded-wing routes are not automatically physically equivalent. Do not compare or interchange them without identifying the actual model and the applicable limitations.
- Keep evaluation scenarios separate from parameter tuning where a held-out claim is made. Distinguish numerical tests, force-level comparisons, free-body simulations, controller evaluations, and real-flight validation.

## Execution and validation

- Use the configured `env_isaaclab` Conda environment and repository wrapper `./isaaclab.sh -p ...`; do not silently switch to another Python/Isaac Sim installation. Report an unavailable environment instead of installing dependencies without permission.
- Run focused checks for the parts changed. Numerical consistency checks are warranted for changed forces/moments or multibody coupling; a brief environment run may be warranted for environment wiring. Do not launch full test suites, broad sweeps, repeated GPU training, or long simulations unless needed for the requested scientific result.
- Preserve existing stable comparison baselines when needed for a study, but do not make feature flags, legacy compatibility, or redundant regression tests mandatory for every new research architecture.
- Avoid routine SHA256/checkpoint hash inventories, audit receipts, extra review agents, and exhaustive reports. Record only the selected model/plant configuration, experimental data, seed when relevant, and metrics required to understand the actual result.

## Completion

Implement the requested scope, report the changed behavior, actual results and consequential limitations briefly, and stop. A task request authorizes the necessary implementation work; seek clarification only for a material missing decision, not for routine stage approval.
