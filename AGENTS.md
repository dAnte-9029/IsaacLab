# Repository Guidelines

## Project Structure & Module Organization

- `source/` contains installable Python extensions:
  - `source/isaaclab/isaaclab/`: core framework (simulation, sensors, env APIs).
  - `source/isaaclab_tasks/isaaclab_tasks/`: prebuilt tasks/environments.
  - `source/isaaclab_assets/isaaclab_assets/`: curated assets/configs.
  - `source/flapping_bot/flapping_bot/`: custom flapping-wing project code (envs, physics, scripts).
- Tests typically live under each extension or in `tests/` (e.g., `tests/test_qsm.py`).
- Entry-point workflows and demos live in `scripts/` (e.g., `scripts/reinforcement_learning/`, `scripts/demos/`).
- Papers and references live in `docs/` (e.g., `docs/papers/`).

## Build, Test, and Development Commands

Use the repo wrapper to run with the correct Isaac Sim / Python environment:

- Install extensions: `./isaaclab.sh --install` (or `-i`).
- Run a script: `./isaaclab.sh -p scripts/.../foo.py --args ...`
- Format/lint (pre-commit): `./isaaclab.sh --format` (or `-f`).
- Run pytest (recommended targeted): `./isaaclab.sh -p -m pytest tests -k <pattern>`

## Coding Style & Naming Conventions

- Python 3.11, 4-space indentation, type hints, and short Google-style docstrings.
- Formatting via `pre-commit` (Black line length 120, isort, flake8, codespell). Avoid noisy reformat-only diffs.
- Naming: `snake_case` for functions/vars, `PascalCase` for classes, `test_*.py` for tests.

## Testing Guidelines

- Use `pytest`. Prefer small, deterministic unit tests for pure math/logic; mark IsaacSim-dependent tests with markers if needed.
- Keep tests runnable headlessly and without network access.

## Commit & Pull Request Guidelines

- Write concise, imperative commit messages (e.g., `feat: add Wang2016 QSM`).
- Keep commits focused; avoid mixing unrelated changes. Don’t commit generated caches or large binaries.
- PRs should describe motivation, link issues (e.g., `Fixes #123`), and include reproduction steps/commands; add screenshots for visual changes.

## Agent-Specific Notes

- Prefer extending custom code under `source/flapping_bot/` over modifying upstream modules under `source/isaaclab/` unless necessary.
- Keep task IDs and training entrypoints stable (e.g., `Isaac-FlappingBot-Direct-v0`, `scripts/reinforcement_learning/...`).

