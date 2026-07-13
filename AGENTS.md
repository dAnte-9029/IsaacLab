
# Repository Agent Guidelines

## 1. Repository purpose

This repository is a fork of Isaac Lab containing a custom bird-scale flapping-wing simulation and control project.

The primary project code is under:

* `source/flapping_bot/`
* `scripts/flapping_px4/`
* project-specific tests and documentation

Treat upstream Isaac Lab code as externally maintained framework code.

## 2. Required context before work

Before planning or modifying project code:

1. Read this file.
2. Read `docs/PROJECT_STATE.md` if it exists.
3. Read the active handoff referenced by `docs/PROJECT_STATE.md`.
4. Read relevant architecture documents and ADRs referenced by the handoff.
5. Read the nearest nested `AGENTS.md` for every directory that may be modified.
6. Run `git status` and inspect recent commits.

If required context is missing or contradictory, report the inconsistency. Do not invent missing project decisions.

Do not rely on previous chat context as the source of truth. Repository code, committed documentation, tests, and approved ADRs are authoritative.

## 3. Modification boundaries

By default, modifications are allowed only under:

* `source/flapping_bot/`
* `scripts/flapping_px4/`
* project-specific files under `tests/`
* project documentation under `docs/`

Do not modify the following unless the user explicitly approves the exact files and reason:

* `source/isaaclab/`
* `source/isaaclab_tasks/`
* `source/isaaclab_assets/`
* upstream applications, framework APIs, or shared Isaac Lab infrastructure

Prefer adapters and extensions inside `source/flapping_bot/` over changes to upstream Isaac Lab.

Do not change unrelated files, perform broad formatting, rename public APIs, or reorganize directories unless explicitly requested.

## 4. Work process

For non-trivial tasks:

1. Inspect the current implementation.
2. State the proposed files and changes.
3. Identify assumptions and unresolved decisions.
4. Define tests and completion criteria.
5. Wait for approval when the task is design-sensitive or the user requested staged approval.
6. Implement only the approved scope.
7. Run targeted validation.
8. Review the final diff.
9. Report completed work, tests, limitations, and unverified items.

Do not silently expand scope.

Do not automatically commit, merge, rebase, reset, delete branches, or discard user changes unless explicitly instructed.

## 5. Git discipline

* One branch should address one cohesive feature, fix, test addition, or documentation task.
* Keep commits focused and independently understandable.
* Do not mix plant-model changes, controller changes, parameter tuning, and mission evaluation in one commit.
* Do not commit generated caches, simulation outputs, checkpoints, videos, large binaries, or temporary debug files.
* Preserve uncommitted user changes.
* Before finishing, show `git status` and summarize the diff.

Recommended commit format:

* `feat(physics): ...`
* `feat(control): ...`
* `fix(env): ...`
* `test(physics): ...`
* `docs(model): ...`
* `refactor(path): ...`

## 6. Build and test commands

Use the repository wrapper so commands run in the configured Isaac Lab environment.

Typical commands:

```bash
./isaaclab.sh -p <script.py> [arguments]
./isaaclab.sh -p -m pytest <test-path> -k <pattern>
./isaaclab.sh --format
```

On Windows, use the corresponding `isaaclab.bat` wrapper.

Prefer targeted tests first. Do not launch expensive GPU simulations, large sweeps, or training runs unless they are necessary and explicitly within scope.

## 7. Coding standards

* Use Python type hints for new public functions and data structures.
* Use 4-space indentation and concise Google-style docstrings.
* Use `snake_case` for variables and functions and `PascalCase` for classes.
* Avoid hidden global state and implicit device or dtype conversion.
* Avoid hard-coded paths and unexplained numerical constants.
* Preserve batch dimensions and GPU compatibility where relevant.
* Avoid noisy format-only diffs.

## 8. Baseline preservation

Existing baseline behavior must remain available unless its removal is explicitly approved.

When adding a new model or controller:

* add an explicit configuration or feature-selection path;
* retain the previous implementation for comparison;
* document whether default behavior changes;
* add regression checks where practical.

Do not modify physical parameters or controller gains merely to make a new implementation appear successful.

## 9. Completion report

At the end of a task, report:

* files added and modified;
* behavior changed;
* commands executed;
* tests passed or failed;
* tests not run and why;
* remaining limitations;
* whether baseline behavior changed;
* any decisions still requiring user approval.
