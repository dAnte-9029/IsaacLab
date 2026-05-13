from __future__ import annotations

import ast
from pathlib import Path


REGISTRATION_FILE = Path(__file__).resolve().parents[1] / "source" / "isaaclab_tasks" / "isaaclab_tasks" / "direct" / "flapping_bot" / "__init__.py"

EXPECTED_TASKS = {
    "Isaac-FlappingBot-StraightFlight-DeLaurier-TeacherRL-Direct-v0": (
        "FlappingBotStraightFlightDeLaurierTeacherRLEnvCfg",
        "FlappingBotStraightFlightPPORunnerCfg",
    ),
    "Isaac-FlappingBot-StraightFlight-DeLaurier-WeakTeacherRL-Direct-v0": (
        "FlappingBotStraightFlightDeLaurierWeakTeacherRLEnvCfg",
        "FlappingBotStraightFlightPPORunnerCfg",
    ),
    "Isaac-FlappingBot-StraightFlight-DeLaurier-PureRL-Direct-v0": (
        "FlappingBotStraightFlightDeLaurierPureRLEnvCfg",
        "FlappingBotStraightFlightPPORunnerCfg",
    ),
    "Isaac-FlappingBot-PathTracking-DeLaurier-TeacherRL-Direct-v0": (
        "FlappingBotPathTrackingEnvCfg",
        "FlappingBotPathTrackingPPORunnerCfg",
    ),
    "Isaac-FlappingBot-PathTracking-DeLaurier-WeakTeacherRL-Direct-v0": (
        "FlappingBotPathTrackingWeakTeacherRLEnvCfg",
        "FlappingBotPathTrackingPPORunnerCfg",
    ),
    "Isaac-FlappingBot-PathTracking-DeLaurier-PureRL-Direct-v0": (
        "FlappingBotPathTrackingPureRLEnvCfg",
        "FlappingBotPathTrackingPPORunnerCfg",
    ),
    "Isaac-FlappingBot-PathTracking-DeLaurier-PrimitiveWeakTeacherRL-Direct-v0": (
        "FlappingBotPathTrackingPrimitiveWeakTeacherRLEnvCfg",
        "FlappingBotPathTrackingPPORunnerCfg",
    ),
    "Isaac-FlappingBot-PathTracking-DeLaurier-PrimitivePureRL-Direct-v0": (
        "FlappingBotPathTrackingPrimitivePureRLEnvCfg",
        "FlappingBotPathTrackingPrimitivePurePPORunnerCfg",
    ),
}


def _registered_task_cfgs() -> dict[str, tuple[str, str]]:
    module = ast.parse(REGISTRATION_FILE.read_text())
    registered: dict[str, tuple[str, str]] = {}
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
        runner_cfg = None
        for kw in call.keywords:
            if kw.arg == "id" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                task_id = kw.value.value
            if kw.arg == "kwargs" and isinstance(kw.value, ast.Dict):
                for key_node, value_node in zip(kw.value.keys, kw.value.values, strict=True):
                    if isinstance(key_node, ast.Constant) and key_node.value == "env_cfg_entry_point":
                        if isinstance(value_node, ast.Name):
                            env_cfg = value_node.id
                    if isinstance(key_node, ast.Constant) and key_node.value == "rsl_rl_cfg_entry_point":
                        if isinstance(value_node, ast.Name):
                            runner_cfg = value_node.id
        if task_id is not None and env_cfg is not None and runner_cfg is not None:
            registered[task_id] = (env_cfg, runner_cfg)
    return registered


def test_straight_flight_rl_task_variants_are_registered() -> None:
    registered = _registered_task_cfgs()
    missing = [task_id for task_id in EXPECTED_TASKS if task_id not in registered]
    assert missing == []


def test_straight_flight_rl_task_variants_use_expected_cfgs() -> None:
    registered = _registered_task_cfgs()
    actual = {task_id: registered[task_id] for task_id in EXPECTED_TASKS if task_id in registered}
    assert actual == EXPECTED_TASKS
