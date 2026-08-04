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


def test_prescribed_variant_uses_passive_wings_and_moving_physx_limit() -> None:
    asset_source = _ASSET_SOURCE.read_text(encoding="utf-8")
    env_source = _ENV_SOURCE.read_text(encoding="utf-8")
    module = ast.parse(env_source)
    config = _class(
        module,
        "FlappingBotStraightFlightMeasuredWingMultibodyPrescribedCoupledEnvCfg",
    )
    delaurier_config = _class(
        module,
        "FlappingBotStraightFlightMeasuredWingMultibodyPrescribedCoupledDeLaurierEnvCfg",
    )

    drive_variant = _ann_assign(config, "wing_drive_variant")
    coupling_mode = _ann_assign(delaurier_config, "wing_aero_coupling_mode")
    assert isinstance(drive_variant.value, ast.Name)
    assert drive_variant.value.id == "PRESCRIBED_COUPLED_WING_DRIVE"
    assert isinstance(coupling_mode.value, ast.Name)
    assert coupling_mode.value.id == "PRESCRIBED_PER_WING_LINK"
    assert '"passive_wings": ImplicitActuatorCfg(' in asset_source
    assert 'joint_names_expr=["left_wing", "right_wing"]' in asset_source
    assert "elif wing_drive_variant == PRESCRIBED_COUPLED_WING_DRIVE:" in env_source
    assert "self._robot.root_physx_view.set_dof_limits(" in env_source


def test_prescribed_branch_does_not_target_a_physical_wing_drive() -> None:
    module = ast.parse(_ENV_SOURCE.read_text(encoding="utf-8"))
    env_class = _class(module, "FlappingBotStraightFlightEnv")
    apply_action = next(
        node
        for node in env_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_apply_action"
    )
    prescribed_branch = next(
        node
        for node in ast.walk(apply_action)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and any(
            isinstance(comparator, ast.Name)
            and comparator.id == "PRESCRIBED_COUPLED_WING_DRIVE"
            for comparator in node.test.comparators
        )
    )
    calls = [
        node
        for statement in prescribed_branch.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert not any(call.func.attr == "write_joint_state_to_sim" for call in calls)
    assert not any(call.func.attr == "set_joint_velocity_target" for call in calls)


def test_ideal_torque_variant_uses_effort_and_hard_mimic_without_moving_limits() -> None:
    asset_source = _ASSET_SOURCE.read_text(encoding="utf-8")
    env_source = _ENV_SOURCE.read_text(encoding="utf-8")
    module = ast.parse(env_source)
    config = _class(
        module,
        "FlappingBotStraightFlightMeasuredWingMultibodyIdealTorqueCoupledEnvCfg",
    )
    delaurier_config = _class(
        module,
        "FlappingBotStraightFlightMeasuredWingMultibodyIdealTorqueCoupledDeLaurierEnvCfg",
    )

    drive_variant = _ann_assign(config, "wing_drive_variant")
    coupling_mode = _ann_assign(delaurier_config, "wing_aero_coupling_mode")
    assert isinstance(drive_variant.value, ast.Name)
    assert drive_variant.value.id == "IDEAL_TORQUE_COUPLED_WING_DRIVE"
    assert isinstance(coupling_mode.value, ast.Name)
    assert coupling_mode.value.id == "IDEAL_TORQUE_PER_WING_LINK"
    assert '"wing_driver": ImplicitActuatorCfg(' in asset_source
    assert "stiffness=0.0" in asset_source
    assert "damping=0.0" in asset_source
    assert "elif wing_drive_variant == IDEAL_TORQUE_COUPLED_WING_DRIVE:" in env_source
    assert "compute_ideal_torque_drive_effort(" in env_source
    assert "self._robot.set_joint_effort_target(" in env_source
