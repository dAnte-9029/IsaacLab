from __future__ import annotations

import ast
from pathlib import Path


REGISTRATION_FILE = (
    Path(__file__).resolve().parents[1]
    / "source/isaaclab_tasks/isaaclab_tasks/direct/flapping_bot/__init__.py"
)


def test_longitudinal_tasks_register_exact_configs_and_shared_runner() -> None:
    module = ast.parse(REGISTRATION_FILE.read_text())
    registrations: dict[str, dict[str, str]] = {}
    for node in module.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not isinstance(call.func, ast.Attribute) or call.func.attr != "register":
            continue
        keywords = {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg is not None}
        task_id = ast.literal_eval(keywords["id"])
        kwargs = keywords["kwargs"]
        assert isinstance(kwargs, ast.Dict)
        values = {ast.literal_eval(key): value for key, value in zip(kwargs.keys, kwargs.values)}
        registrations[task_id] = {
            "env": values["env_cfg_entry_point"].id,
            "runner": values["rsl_rl_cfg_entry_point"].id,
        }

    for suffix, stage in (("C2a", "C2a"), ("C2b", "C2b"), ("C2c", "C2c")):
        task_id = f"Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-{suffix}-Direct-v0"
        assert registrations[task_id] == {
            "env": f"FlappingBotStraightFlightDeLaurierMeasuredPureRL{stage}EnvCfg",
            "runner": "FlappingBotStraightFlightPPORunnerCfg",
        }

    assert registrations["Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-Direct-v0"]["env"] == (
        "FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg"
    )

    assert registrations[
        "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-GpuImplicit-Direct-v0"
    ] == {
        "env": "FlappingBotStraightFlightDeLaurierMeasuredPureRLGpuImplicitEnvCfg",
        "runner": "FlappingBotStraightFlightPPORunnerCfg",
    }
    assert registrations[
        "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C2a-GpuImplicit-Direct-v0"
    ] == {
        "env": "FlappingBotStraightFlightDeLaurierMeasuredPureRLC2aGpuImplicitEnvCfg",
        "runner": "FlappingBotStraightFlightPPORunnerCfg",
    }
    assert registrations[
        "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-GpuPhaseMatched-Direct-v0"
    ] == {
        "env": "FlappingBotStraightFlightDeLaurierMeasuredPureRLGpuPhaseMatchedEnvCfg",
        "runner": "FlappingBotStraightFlightPPORunnerCfg",
    }
    assert registrations[
        "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C2a-GpuPhaseMatched-Direct-v0"
    ] == {
        "env": "FlappingBotStraightFlightDeLaurierMeasuredPureRLC2aGpuPhaseMatchedEnvCfg",
        "runner": "FlappingBotStraightFlightPPORunnerCfg",
    }
