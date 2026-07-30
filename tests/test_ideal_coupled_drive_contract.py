from __future__ import annotations

import ast
from pathlib import Path


_ASSET_SOURCE = (
    Path(__file__).parents[1]
    / "source"
    / "flapping_bot"
    / "flapping_bot"
    / "assets"
    / "ideal_coupled_drive.py"
)
_ENV_SOURCE = (
    Path(__file__).parents[1]
    / "source"
    / "flapping_bot"
    / "flapping_bot"
    / "direct"
    / "flapping_bot"
    / "straight_flight_env.py"
)


def _class(module: ast.Module, name: str) -> ast.ClassDef:
    return next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == name)


def _ann_assign(class_node: ast.ClassDef, name: str) -> ast.AnnAssign:
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name
    )


def test_formal_env_selects_measured_plant_and_disables_kinematic_override() -> None:
    module = ast.parse(_ENV_SOURCE.read_text(encoding="utf-8"))
    config = _class(module, "FlappingBotStraightFlightMeasuredWingMultibodyIdealCoupledEnvCfg")

    drive_variant = _ann_assign(config, "wing_drive_variant")
    kinematic_override = _ann_assign(config, "use_kinematic_joint_override")
    assert isinstance(drive_variant.value, ast.Name)
    assert drive_variant.value.id == "IDEAL_COUPLED_WING_DRIVE"
    assert ast.literal_eval(kinematic_override.value) is False
    assert any(
        isinstance(base, ast.Name) and base.id == "FlappingBotStraightFlightMeasuredWingMultibodyEnvCfg"
        for base in config.bases
    )


def test_multibody_delaurier_variant_selects_actual_per_wing_coupling() -> None:
    module = ast.parse(_ENV_SOURCE.read_text(encoding="utf-8"))
    base_config = _class(module, "FlappingBotStraightFlightEnvCfg")
    multibody_config = _class(
        module,
        "FlappingBotStraightFlightMeasuredWingMultibodyIdealCoupledDeLaurierEnvCfg",
    )

    default_coupling = _ann_assign(base_config, "wing_aero_coupling_mode")
    multibody_coupling = _ann_assign(multibody_config, "wing_aero_coupling_mode")
    use_delaurier = _ann_assign(multibody_config, "use_delaurier_wings")
    assert isinstance(default_coupling.value, ast.Name)
    assert default_coupling.value.id == "COMMANDED_BASE_EQUIVALENT"
    assert isinstance(multibody_coupling.value, ast.Name)
    assert multibody_coupling.value.id == "ACTUAL_PER_WING_LINK"
    assert ast.literal_eval(use_delaurier.value) is True
    assert any(
        isinstance(base, ast.Name)
        and base.id == "FlappingBotStraightFlightMeasuredWingMultibodyIdealCoupledEnvCfg"
        for base in multibody_config.bases
    )


def test_apply_action_targets_only_left_wing_driver_in_ideal_mode() -> None:
    source = _ENV_SOURCE.read_text(encoding="utf-8")
    assert "left_driver_joint_id = [int(self._joint_ids[self._IDX_LEFT_WING])]" in source
    assert "self._robot.set_joint_velocity_target(" in source
    assert "elif wing_drive_variant == IDEAL_COUPLED_WING_DRIVE:" in source


def test_asset_declares_one_driver_and_one_passive_right_wing() -> None:
    source = _ASSET_SOURCE.read_text(encoding="utf-8")
    assert '"wing_driver": ImplicitActuatorCfg(' in source
    assert 'joint_names_expr=["left_wing"]' in source
    assert '"passive_right_wing": ImplicitActuatorCfg(' in source
    assert 'joint_names_expr=["right_wing"]' in source
    assert "IDEAL_DRIVER_STIFFNESS_NM_PER_RAD = 2_000.0" in source
    assert "IDEAL_DRIVER_DAMPING_NM_S_PER_RAD = 20.0" in source
    assert "IDEAL_DRIVER_EFFORT_LIMIT_NM = 1_000.0" in source
