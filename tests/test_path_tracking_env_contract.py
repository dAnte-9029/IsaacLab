from __future__ import annotations

import ast
from pathlib import Path

from flapping_bot.direct.flapping_bot.path_tracking_env import (
    FlappingBotPathTrackingEnv,
    FlappingBotPathTrackingEnvCfg,
    PATH_TRACKING_RUNTIME_AVAILABLE,
)


REGISTRATION_FILE = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "isaaclab_tasks"
    / "isaaclab_tasks"
    / "direct"
    / "flapping_bot"
    / "__init__.py"
)


def _registered_env_cfgs() -> dict[str, str]:
    module = ast.parse(REGISTRATION_FILE.read_text())
    registered: dict[str, str] = {}
    for node in module.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not isinstance(call.func, ast.Attribute):
            continue
        if not isinstance(call.func.value, ast.Name) or call.func.value.id != "gym" or call.func.attr != "register":
            continue

        task_id = None
        env_cfg = None
        for kw in call.keywords:
            if kw.arg == "id" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                task_id = kw.value.value
            if kw.arg == "kwargs" and isinstance(kw.value, ast.Dict):
                for key_node, value_node in zip(kw.value.keys, kw.value.values, strict=True):
                    if isinstance(key_node, ast.Constant) and key_node.value == "env_cfg_entry_point":
                        if isinstance(value_node, ast.Name):
                            env_cfg = value_node.id
        if task_id is not None and env_cfg is not None:
            registered[task_id] = env_cfg
    return registered


def test_path_tracking_task_is_registered() -> None:
    registered = _registered_env_cfgs()
    assert "Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0" in registered


def test_path_tracking_task_uses_expected_env_cfg() -> None:
    registered = _registered_env_cfgs()
    assert registered["Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0"] == "FlappingBotPathTrackingEnvCfg"


def test_path_tracking_module_exports_task_symbols() -> None:
    assert FlappingBotPathTrackingEnv.__name__ == "FlappingBotPathTrackingEnv"
    assert FlappingBotPathTrackingEnvCfg.__name__ == "FlappingBotPathTrackingEnvCfg"
    assert isinstance(PATH_TRACKING_RUNTIME_AVAILABLE, bool)


def test_path_tracking_env_defaults_enable_teacher_guidance() -> None:
    module = ast.parse(
        (
            Path(__file__).resolve().parents[1]
            / "source"
            / "flapping_bot"
            / "flapping_bot"
            / "direct"
            / "flapping_bot"
            / "path_tracking_env.py"
        ).read_text()
    )
    for node in ast.walk(module):
        if not isinstance(node, ast.ClassDef) or node.name != "FlappingBotPathTrackingEnvCfg":
            continue
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) and item.target.id == "teacher_guidance_enabled":
                assert isinstance(item.value, ast.Constant) and item.value.value is True
                return
    raise AssertionError("teacher_guidance_enabled=True not found in FlappingBotPathTrackingEnvCfg")


def test_path_tracking_env_uses_fixed_five_point_preview_contract() -> None:
    module = ast.parse(
        (
            Path(__file__).resolve().parents[1]
            / "source"
            / "flapping_bot"
            / "flapping_bot"
            / "direct"
            / "flapping_bot"
            / "path_tracking_env.py"
        ).read_text()
    )

    preview_tuple_values: list[tuple[int, ...]] = []
    observation_space_value: int | None = None

    for node in ast.walk(module):
        if isinstance(node, ast.ClassDef) and node.name == "FlappingBotPathTrackingEnvCfg":
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) and item.target.id == "observation_space":
                    assert isinstance(item.value, ast.Constant) and isinstance(item.value.value, int)
                    observation_space_value = item.value.value

        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if not isinstance(target, ast.Attribute) or target.attr not in {
                "_path_preview_points_xyz",
                "_path_preview_points_body_xyz",
            }:
                continue
            call = node.value
            if not isinstance(call, ast.Call):
                continue
            if not isinstance(call.func, ast.Attribute) or call.func.attr != "zeros":
                continue
            shape_arg = call.args[0] if call.args else None
            if not isinstance(shape_arg, ast.Tuple):
                continue
            dims: list[int] = []
            for elt in shape_arg.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
                    dims.append(elt.value)
            if dims:
                preview_tuple_values.append(tuple(dims))

    assert observation_space_value == 96
    assert (5, 3) in {dims[-2:] for dims in preview_tuple_values}
