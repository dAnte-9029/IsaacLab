from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / "source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py"
DIRECT_INIT_FILE = ROOT / "source/flapping_bot/flapping_bot/direct/flapping_bot/__init__.py"
PACKAGE_INIT_FILE = ROOT / "source/flapping_bot/flapping_bot/__init__.py"
PATH_MODULE = ROOT / "source/flapping_bot/flapping_bot/direct/flapping_bot/pure_rl_longitudinal_path.py"


def _module() -> ast.Module:
    return ast.parse(ENV_FILE.read_text())


def _class(module: ast.Module, name: str) -> ast.ClassDef:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found")


def _method(class_node: ast.ClassDef, name: str) -> ast.FunctionDef:
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"method {name} not found")


def _ann_assign(class_node: ast.ClassDef, name: str) -> ast.AnnAssign:
    for node in class_node.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            return node
    raise AssertionError(f"field {name} not found")


def _called_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if isinstance(child.func, ast.Name):
            names.add(child.func.id)
        elif isinstance(child.func, ast.Attribute):
            names.add(child.func.attr)
    return names


def _load_path_module():
    spec = importlib.util.spec_from_file_location("pure_rl_longitudinal_path_env_test", PATH_MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_c2_configs_select_only_the_approved_longitudinal_stage() -> None:
    module = _module()
    base = _class(module, "FlappingBotStraightFlightEnvCfg")
    measured_c1 = _class(module, "FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg")

    assert ast.literal_eval(_ann_assign(base, "pure_rl_longitudinal_stage_id").value) is None
    assert ast.literal_eval(_ann_assign(measured_c1, "pure_rl_longitudinal_stage_id").value) is None
    for suffix, stage_id in (("C2a", "c2a"), ("C2b", "c2b"), ("C2c", "c2c")):
        cfg = _class(module, f"FlappingBotStraightFlightDeLaurierMeasuredPureRL{suffix}EnvCfg")
        assert len(cfg.bases) == 1
        assert isinstance(cfg.bases[0], ast.Name)
        assert cfg.bases[0].id == "FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg"
        assert ast.literal_eval(_ann_assign(cfg, "pure_rl_longitudinal_stage_id").value) == stage_id
        assert all(
            not (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id in {
                "observation_space",
                "wind_enabled",
                "randomize_wind",
                "action_interface",
                "decimation",
            })
            for node in cfg.body
        )


def test_environment_routes_c2_reset_observation_reward_and_termination_through_path_contract() -> None:
    module = _module()
    env = _class(module, "FlappingBotStraightFlightEnv")

    reset_calls = _called_names(_method(env, "_reset_idx"))
    observation_calls = _called_names(_method(env, "_get_pure_rl_observations"))
    reward_calls = _called_names(_method(env, "_get_pure_rl_curriculum1_reward"))
    done_calls = _called_names(_method(env, "_get_dones"))
    assert "sample_longitudinal_path_batch" in reset_calls
    assert "write_longitudinal_path_batch_rows_" in reset_calls
    assert "_query_pure_rl_longitudinal_path" in observation_calls
    assert "compute_pure_rl_path_reward_terms" in reward_calls
    assert "_query_pure_rl_longitudinal_path" in done_calls
    assert "path_tracking_env" not in ENV_FILE.read_text()


def test_c2_configs_are_exported_from_both_project_package_surfaces() -> None:
    for file_path in (DIRECT_INIT_FILE, PACKAGE_INIT_FILE):
        text = file_path.read_text()
        for suffix in ("C2a", "C2b", "C2c"):
            assert f"FlappingBotStraightFlightDeLaurierMeasuredPureRL{suffix}EnvCfg" in text


def test_partial_path_batch_write_changes_only_selected_environments() -> None:
    path_module = _load_path_module()
    destination = path_module.PureRLLongitudinalPathBatch(
        task_id=torch.zeros(5, dtype=torch.int64),
        heading_rad=torch.zeros(5),
        signed_slope_rad=torch.zeros(5),
        entry_length_m=torch.full((5,), 15.0),
        slope_length_m=torch.full((5,), 20.0),
        initial_altitude_m=torch.full((5,), 10.0),
    )
    source = path_module.PureRLLongitudinalPathBatch(
        task_id=torch.tensor([1, 2], dtype=torch.int64),
        heading_rad=torch.tensor([0.2, -0.3]),
        signed_slope_rad=torch.tensor([0.1, -0.1]),
        entry_length_m=torch.tensor([17.0, 18.0]),
        slope_length_m=torch.tensor([24.0, 26.0]),
        initial_altitude_m=torch.tensor([11.0, 12.0]),
    )

    path_module.write_longitudinal_path_batch_rows_(
        destination=destination,
        env_ids=torch.tensor([1, 4], dtype=torch.int64),
        source=source,
    )

    assert destination.task_id.tolist() == [0, 1, 0, 0, 2]
    torch.testing.assert_close(destination.heading_rad, torch.tensor([0.0, 0.2, 0.0, 0.0, -0.3]))
    torch.testing.assert_close(destination.entry_length_m, torch.tensor([15.0, 17.0, 15.0, 15.0, 18.0]))
    torch.testing.assert_close(destination.initial_altitude_m, torch.tensor([10.0, 11.0, 10.0, 10.0, 12.0]))
