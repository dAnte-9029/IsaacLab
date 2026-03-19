from __future__ import annotations

import ast
from pathlib import Path


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
