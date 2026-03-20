"""Top-level package marker for the flapping_bot extension.

This repository contains both IsaacSim-dependent components (tasks/assets) and
IsaacSim-independent utilities (e.g., pure math models). To keep pure-Python
imports usable in environments without IsaacSim, this package avoids importing
IsaacSim modules at import time.

Use explicit imports when needed, for example:

- Pure math: `from flapping_bot.flapping_bot.physics.qsm_wang2016 import compute_aero_wrench`
- IsaacSim envs: `from flapping_bot.flapping_bot.direct.flapping_bot import FlappingBotEnv`
"""

from importlib import import_module
from pathlib import Path
from typing import Any

_PACKAGE_ROOT = Path(__file__).resolve().parent
_INNER_PACKAGE_ROOT = _PACKAGE_ROOT / "flapping_bot"

if _INNER_PACKAGE_ROOT.is_dir():
    _inner_path = str(_INNER_PACKAGE_ROOT)
    if _inner_path not in __path__:
        __path__.append(_inner_path)


def __getattr__(name: str) -> Any:
    # Backwards-compatible lazy re-export of the inner package.
    inner = import_module("flapping_bot.flapping_bot")
    return getattr(inner, name)


def __dir__() -> list[str]:
    inner = import_module("flapping_bot.flapping_bot")
    return sorted(set(globals().keys()) | set(getattr(inner, "__all__", [])))
