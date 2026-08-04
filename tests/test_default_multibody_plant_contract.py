"""Pure contract checks for the promoted default flapping plant."""

from __future__ import annotations

import ast
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_ENV_PATH = (
    _REPO_ROOT
    / "source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py"
)


def _class_definitions() -> dict[str, ast.ClassDef]:
    tree = ast.parse(_ENV_PATH.read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }


def _assignments(class_def: ast.ClassDef) -> dict[str, ast.expr]:
    assignments: dict[str, ast.expr] = {}
    for node in class_def.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assert node.value is not None
            assignments[node.target.id] = node.value
    return assignments


def _base_names(class_def: ast.ClassDef) -> list[str]:
    return [base.id for base in class_def.bases if isinstance(base, ast.Name)]


def _numeric_value(node: ast.expr) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _numeric_value(node.left) / _numeric_value(node.right)
    raise AssertionError(f"Expected a numeric literal expression, got {ast.dump(node)}")


def test_canonical_delaurier_config_selects_native_multibody_plant() -> None:
    classes = _class_definitions()
    canonical = classes["FlappingBotStraightFlightDeLaurierEnvCfg"]
    assert _base_names(canonical) == [
        "FlappingBotStraightFlightMeasuredWingMultibodyNativeHolonomicDeLaurierEnvCfg"
    ]

    assignments = _assignments(canonical)
    assert ast.literal_eval(assignments["min_flap_hz"]) == 2.0
    assert ast.literal_eval(assignments["max_flap_hz"]) == 5.0
    assert ast.literal_eval(assignments["reset_flap_hz"]) == 4.0
    assert ast.literal_eval(assignments["enable_tail_aero"]) is True


def test_native_multibody_default_freezes_validated_runtime_contract() -> None:
    classes = _class_definitions()
    native = classes["FlappingBotStraightFlightMeasuredWingMultibodyNativeHolonomicEnvCfg"]
    assignments = _assignments(native)

    assert ast.literal_eval(assignments["decimation"]) == 4
    sim_replace = assignments["sim"]
    assert isinstance(sim_replace, ast.Call)
    sim_keywords = {keyword.arg: keyword.value for keyword in sim_replace.keywords}
    assert _numeric_value(sim_keywords["dt"]) == 1.0 / 480.0
    assert ast.literal_eval(sim_keywords["device"]) == "cpu"

    scene_call = assignments["scene"]
    assert isinstance(scene_call, ast.Call)
    scene_keywords = {keyword.arg: keyword.value for keyword in scene_call.keywords}
    assert ast.literal_eval(scene_keywords["replicate_physics"]) is False


def test_commanded_kinematics_delaurier_baseline_remains_explicit() -> None:
    classes = _class_definitions()
    legacy = classes["FlappingBotStraightFlightCommandedKinematicsDeLaurierEnvCfg"]
    assert _base_names(legacy) == ["FlappingBotStraightFlightEnvCfg"]
    assert ast.literal_eval(_assignments(legacy)["use_delaurier_wings"]) is True
