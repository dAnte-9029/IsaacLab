from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION_FILE = ROOT / "source/flapping_bot/flapping_bot/direct/__init__.py"


def test_project_local_registration_contains_exact_c3_tasks_and_entry_points() -> None:
    module = ast.parse(REGISTRATION_FILE.read_text())
    runner_cfg = next(
        node.value
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_RSL_RL_CFG" for target in node.targets)
    )
    assert ast.literal_eval(runner_cfg) == (
        "isaaclab_tasks.direct.flapping_bot.agents."
        "rsl_rl_ppo_straightflight_cfg:FlappingBotStraightFlightPPORunnerCfg"
    )
    registrations = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "register"
    ]
    ids = {
        ast.literal_eval(next(keyword.value for keyword in call.keywords if keyword.arg == "id")): call
        for call in registrations
    }
    for suffix in ("C3a", "C3b", "C3c"):
        task_id = f"Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-{suffix}-Direct-v0"
        assert task_id in ids
        source = ast.unparse(ids[task_id])
        assert "FlappingBotStraightFlightEnv" in source
        assert f"FlappingBotStraightFlightDeLaurierMeasuredPureRL{suffix}EnvCfg" in source
        assert "'rsl_rl_cfg_entry_point': _RSL_RL_CFG" in source


def test_registration_does_not_modify_upstream_task_module() -> None:
    text = REGISTRATION_FILE.read_text()
    assert "isaaclab_tasks.direct.flapping_bot.agents" in text
    assert "source/isaaclab_tasks" not in text
