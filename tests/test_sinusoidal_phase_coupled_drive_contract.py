from __future__ import annotations

import ast
from pathlib import Path


_ROOT = Path(__file__).parents[1]
_ASSET_SOURCE = (
    _ROOT
    / "source"
    / "flapping_bot"
    / "flapping_bot"
    / "assets"
    / "ideal_coupled_drive.py"
)
_ENV_SOURCE = (
    _ROOT
    / "source"
    / "flapping_bot"
    / "flapping_bot"
    / "direct"
    / "flapping_bot"
    / "straight_flight_env.py"
)


def _class(module: ast.Module, name: str) -> ast.ClassDef:
    return next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _ann_assign(class_node: ast.ClassDef, name: str) -> ast.AnnAssign:
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == name
    )


def test_sinusoidal_phase_gate_is_explicit_and_aerodynamics_off() -> None:
    module = ast.parse(_ENV_SOURCE.read_text(encoding="utf-8"))
    config = _class(
        module,
        "FlappingBotStraightFlightMeasuredWingMultibodySinusoidalPhaseCoupledEnvCfg",
    )

    drive_variant = _ann_assign(config, "wing_drive_variant")
    kinematic_override = _ann_assign(config, "use_kinematic_joint_override")
    wing_aero = _ann_assign(config, "enable_wing_aero")
    tail_aero = _ann_assign(config, "enable_tail_aero")
    assert isinstance(drive_variant.value, ast.Name)
    assert drive_variant.value.id == "SINUSOIDAL_PHASE_SPEED_WING_DRIVE"
    assert ast.literal_eval(kinematic_override.value) is False
    assert ast.literal_eval(wing_aero.value) is False
    assert ast.literal_eval(tail_aero.value) is False


def test_sinusoidal_phase_delaurier_variant_is_explicit_and_per_wing() -> None:
    module = ast.parse(_ENV_SOURCE.read_text(encoding="utf-8"))
    config = _class(
        module,
        "FlappingBotStraightFlightMeasuredWingMultibodySinusoidalPhaseCoupledDeLaurierEnvCfg",
    )

    wing_aero = _ann_assign(config, "enable_wing_aero")
    delaurier = _ann_assign(config, "use_delaurier_wings")
    coupling_mode = _ann_assign(config, "wing_aero_coupling_mode")
    acceleration_source = _ann_assign(config, "wing_aero_acceleration_source")
    assert ast.literal_eval(wing_aero.value) is True
    assert ast.literal_eval(delaurier.value) is True
    assert isinstance(coupling_mode.value, ast.Name)
    assert coupling_mode.value.id == "SINUSOIDAL_PHASE_PER_WING_LINK"
    assert isinstance(acceleration_source.value, ast.Name)
    assert acceleration_source.value.id == "ACTUAL_JOINT_ACCELERATION"


def test_sinusoidal_phase_asset_uses_passive_effort_wings_and_hard_mimic() -> None:
    asset_source = _ASSET_SOURCE.read_text(encoding="utf-8")
    env_source = _ENV_SOURCE.read_text(encoding="utf-8")

    assert 'SINUSOIDAL_PHASE_SPEED_WING_DRIVE = "sinusoidal_phase_speed_drive"' in asset_source
    assert (
        "SinusoidalPhaseSpeedCoupledFlappingBotCfg = "
        "IdealTorqueCoupledFlappingBotCfg.replace()"
    ) in asset_source
    assert "SINUSOIDAL_PHASE_SPEED_WING_DRIVE," in env_source
    assert "apply_hard_opposed_wing_mimic(" in env_source


def test_sinusoidal_phase_branch_uses_effort_without_moving_limits_or_state_write() -> None:
    module = ast.parse(_ENV_SOURCE.read_text(encoding="utf-8"))
    env_class = _class(module, "FlappingBotStraightFlightEnv")
    apply_action = next(
        node
        for node in env_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_apply_action"
    )
    branch = next(
        node
        for node in ast.walk(apply_action)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and any(
            isinstance(comparator, ast.Name)
            and comparator.id == "SINUSOIDAL_PHASE_SPEED_WING_DRIVE"
            for comparator in node.test.comparators
        )
        and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "compute_sinusoidal_constraint_effort"
            for statement in node.body
            for call in ast.walk(statement)
        )
    )
    calls = [
        call
        for statement in branch.body
        for call in ast.walk(statement)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
    ]

    assert any(call.func.attr == "set_joint_effort_target" for call in calls)
    assert not any(call.func.attr == "set_dof_limits" for call in calls)
    assert not any(call.func.attr == "write_joint_state_to_sim" for call in calls)


def test_coupled_phase_core_excludes_transformed_physx_wing_inertia() -> None:
    source = _ENV_SOURCE.read_text(encoding="utf-8")

    assert "common_joint_inertia_kg_m2=0.0" in source
    assert (
        "constant_phase_inertia_kg_m2="
        "float(self.cfg.sinusoidal_phase_inertia_kg_m2)"
    ) in source
    assert "common_joint_external_torque_nm=-constraint.common_wing_effort_nm" in source
