from __future__ import annotations

import ast
from pathlib import Path


CFG_FILE = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "isaaclab_tasks"
    / "isaaclab_tasks"
    / "direct"
    / "flapping_bot"
    / "agents"
    / "rsl_rl_ppo_straightflight_cfg.py"
)


def _load_module() -> ast.Module:
    return ast.parse(CFG_FILE.read_text())


def _find_class(module: ast.Module, class_name: str) -> ast.ClassDef:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"class {class_name} not found")


def _find_cfg_keyword_value(class_node: ast.ClassDef, attr_name: str, keyword_name: str) -> float:
    for node in class_node.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id != attr_name:
            continue
        if not isinstance(node.value, ast.Call):
            continue
        for keyword in node.value.keywords:
            if keyword.arg != keyword_name:
                continue
            if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, (float, int)):
                return float(keyword.value.value)
    raise AssertionError(f"{attr_name}.{keyword_name} not explicitly defined in {class_node.name}")


def test_path_tracking_ppo_cfg_reduces_policy_noise_vs_straight_flight() -> None:
    module = _load_module()
    straight = _find_class(module, "FlappingBotStraightFlightPPORunnerCfg")
    path_tracking = _find_class(module, "FlappingBotPathTrackingPPORunnerCfg")

    straight_noise = _find_cfg_keyword_value(straight, "policy", "init_noise_std")
    path_tracking_noise = _find_cfg_keyword_value(path_tracking, "policy", "init_noise_std")

    assert path_tracking_noise < straight_noise


def test_path_tracking_ppo_cfg_reduces_entropy_vs_straight_flight() -> None:
    module = _load_module()
    straight = _find_class(module, "FlappingBotStraightFlightPPORunnerCfg")
    path_tracking = _find_class(module, "FlappingBotPathTrackingPPORunnerCfg")

    straight_entropy = _find_cfg_keyword_value(straight, "algorithm", "entropy_coef")
    path_tracking_entropy = _find_cfg_keyword_value(path_tracking, "algorithm", "entropy_coef")

    assert path_tracking_entropy < straight_entropy


def test_path_tracking_ppo_cfg_uses_longer_rollouts_than_straight_flight() -> None:
    module = _load_module()
    straight = _find_class(module, "FlappingBotStraightFlightPPORunnerCfg")
    path_tracking = _find_class(module, "FlappingBotPathTrackingPPORunnerCfg")

    straight_rollout = None
    path_tracking_rollout = None
    for node in straight.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "num_steps_per_env":
                assert isinstance(node.value, ast.Constant)
                straight_rollout = int(node.value.value)
    for node in path_tracking.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "num_steps_per_env":
                assert isinstance(node.value, ast.Constant)
                path_tracking_rollout = int(node.value.value)

    assert straight_rollout is not None
    assert path_tracking_rollout is not None
    assert path_tracking_rollout > straight_rollout


def test_path_tracking_ppo_cfg_uses_less_myopic_discount_than_straight_flight() -> None:
    module = _load_module()
    straight = _find_class(module, "FlappingBotStraightFlightPPORunnerCfg")
    path_tracking = _find_class(module, "FlappingBotPathTrackingPPORunnerCfg")

    straight_gamma = _find_cfg_keyword_value(straight, "algorithm", "gamma")
    path_tracking_gamma = _find_cfg_keyword_value(path_tracking, "algorithm", "gamma")

    assert path_tracking_gamma > straight_gamma


def test_path_tracking_ppo_cfg_preserves_long_horizon_loiter_settings() -> None:
    module = _load_module()
    path_tracking = _find_class(module, "FlappingBotPathTrackingPPORunnerCfg")

    path_tracking_rollout = None
    for node in path_tracking.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "num_steps_per_env":
                assert isinstance(node.value, ast.Constant)
                path_tracking_rollout = int(node.value.value)

    path_tracking_gamma = _find_cfg_keyword_value(path_tracking, "algorithm", "gamma")
    path_tracking_lam = _find_cfg_keyword_value(path_tracking, "algorithm", "lam")

    assert path_tracking_rollout is not None
    assert path_tracking_rollout >= 192
    assert path_tracking_gamma >= 0.999
    assert path_tracking_lam >= 0.97
