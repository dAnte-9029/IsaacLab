from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.flapping_px4.fly_straight_line import _resolve_initial_elevon_actions


SCRIPT_FILE = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "flapping_px4"
    / "fly_straight_line.py"
)


def _load_module() -> ast.Module:
    return ast.parse(SCRIPT_FILE.read_text(encoding="utf-8"))


def _find_controller_cfg_call(module: ast.Module) -> ast.Call:
    for node in ast.walk(module):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "PX4LikeStraightLineControllerCfg":
            return node
    raise AssertionError("PX4LikeStraightLineControllerCfg call not found")


class _DummyEnvCfg:
    reset_elevon_pitch_deg = -10.0
    reset_elevon_roll_deg = 5.0
    elevon_max_deg = 41.0


def test_resolve_initial_elevon_actions_defaults_to_zero_without_env_seed() -> None:
    args = SimpleNamespace(seed_controller_from_env_reset=False)

    pitch, roll = _resolve_initial_elevon_actions(args, _DummyEnvCfg())

    assert pitch == 0.0
    assert roll == 0.0


def test_resolve_initial_elevon_actions_normalizes_env_reset_trim_when_enabled() -> None:
    args = SimpleNamespace(seed_controller_from_env_reset=True)

    pitch, roll = _resolve_initial_elevon_actions(args, _DummyEnvCfg())

    assert pitch == pytest.approx(-10.0 / 41.0)
    assert roll == pytest.approx(5.0 / 41.0)


def test_fly_straight_line_controller_cfg_uses_resolved_initial_elevon_actions() -> None:
    call = _find_controller_cfg_call(_load_module())
    keyword_values = {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg is not None}

    assert isinstance(keyword_values["initial_elevon_pitch_action"], ast.Name)
    assert keyword_values["initial_elevon_pitch_action"].id == "initial_elevon_pitch_action"
    assert isinstance(keyword_values["initial_elevon_roll_action"], ast.Name)
    assert keyword_values["initial_elevon_roll_action"].id == "initial_elevon_roll_action"
