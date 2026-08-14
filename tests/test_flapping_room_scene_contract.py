from __future__ import annotations

import ast
from pathlib import Path


SCENE_CFG_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "flapping_bot"
    / "flapping_bot"
    / "scenes"
    / "flapping_room_cfg.py"
)


def _keyword(call: ast.Call, name: str) -> ast.AST:
    return next(keyword.value for keyword in call.keywords if keyword.arg == name)


def _tuple_values(node: ast.AST) -> tuple[float, ...]:
    assert isinstance(node, ast.Tuple)
    return tuple(float(ast.literal_eval(element)) for element in node.elts)


def test_flapping_room_ground_is_local_static_geometry() -> None:
    module = ast.parse(SCENE_CFG_PATH.read_text(encoding="utf-8"))
    scene = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "FlappingRoomSceneCfg"
    )
    ground_assignment = next(
        node
        for node in scene.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "ground" for target in node.targets)
    )
    assert isinstance(ground_assignment.value, ast.Call)
    ground = ground_assignment.value
    spawn = _keyword(ground, "spawn")
    assert isinstance(spawn, ast.Call)
    assert isinstance(spawn.func, ast.Attribute)
    assert spawn.func.attr == "CuboidCfg"

    assert _tuple_values(_keyword(spawn, "size")) == (500.0, 500.0, 0.1)
    init_state = _keyword(ground, "init_state")
    assert isinstance(init_state, ast.Call)
    assert _tuple_values(_keyword(init_state, "pos")) == (0.0, 0.0, -0.05)

    collision_props = _keyword(spawn, "collision_props")
    assert isinstance(collision_props, ast.Call)
    assert isinstance(collision_props.func, ast.Attribute)
    assert collision_props.func.attr == "CollisionPropertiesCfg"

    physics_material = _keyword(spawn, "physics_material")
    assert isinstance(physics_material, ast.Call)
    assert ast.literal_eval(_keyword(physics_material, "static_friction")) == 0.5
    assert ast.literal_eval(_keyword(physics_material, "dynamic_friction")) == 0.5
