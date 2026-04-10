from __future__ import annotations

import ast
from pathlib import Path


ENV_FILE = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "flapping_bot"
    / "flapping_bot"
    / "direct"
    / "flapping_bot"
    / "straight_flight_env.py"
)


def _load_module() -> ast.Module:
    return ast.parse(ENV_FILE.read_text())


def _find_class(module: ast.Module, class_name: str) -> ast.ClassDef:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"class {class_name} not found")


def _find_method(class_node: ast.ClassDef, method_name: str) -> ast.FunctionDef:
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    raise AssertionError(f"method {method_name} not found in {class_node.name}")


def _find_ann_assign(class_node: ast.ClassDef, field_name: str) -> ast.AnnAssign:
    for node in class_node.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == field_name:
            return node
    raise AssertionError(f"field {field_name} not found in {class_node.name}")


def test_straight_flight_reset_calls_base_reset_idx() -> None:
    module = _load_module()
    class_node = _find_class(module, "FlappingBotStraightFlightEnv")
    reset_idx = _find_method(class_node, "_reset_idx")

    for node in ast.walk(reset_idx):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "_reset_idx":
            continue
        if not isinstance(node.func.value, ast.Call):
            continue
        if not isinstance(node.func.value.func, ast.Name) or node.func.value.func.id != "super":
            continue
        return

    raise AssertionError("FlappingBotStraightFlightEnv._reset_idx must call super()._reset_idx(env_ids)")


def test_straight_flight_env_exposes_near_physical_elevon_authority() -> None:
    module = _load_module()
    class_node = _find_class(module, "FlappingBotStraightFlightEnvCfg")
    assign = _find_ann_assign(class_node, "elevon_max_deg")
    assert isinstance(assign.value, ast.Constant)
    assert float(assign.value.value) >= 40.5


def test_straight_flight_env_exposes_tail_aero_compatibility_fields() -> None:
    module = _load_module()
    class_node = _find_class(module, "FlappingBotStraightFlightEnvCfg")

    expected_fields = {
        "tail_horizontal_tail_incidence_bias_deg": 0.0,
        "tail_fixed_horizontal_effectiveness": 0.5,
        "tail_elevon_effectiveness": 1.2,
        "tail_elevon_alpha_limit_deg": 25.0,
        "tail_horizontal_tail_q_scale": 1.0,
    }

    found_fields: dict[str, float] = {}
    for field_name in expected_fields:
        assign = _find_ann_assign(class_node, field_name)
        assert isinstance(assign.value, ast.Constant)
        found_fields[field_name] = float(assign.value.value)

    assert found_fields == expected_fields
