from __future__ import annotations

import ast
from pathlib import Path


_ROOT = Path(__file__).parents[1]
_ASSET_SOURCE = (
    _ROOT / "source" / "flapping_bot" / "flapping_bot" / "assets" / "ideal_coupled_drive.py"
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
    return next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == name)


def _ann_assign(class_node: ast.ClassDef, name: str) -> ast.AnnAssign:
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == name
    )


def test_scheme_b_variants_are_explicit_and_baseline_safe() -> None:
    module = ast.parse(_ENV_SOURCE.read_text(encoding="utf-8"))
    no_aero = _class(
        module,
        "FlappingBotStraightFlightMeasuredWingMultibodyIdealInverseDynamicsPhaseCoupledEnvCfg",
    )
    aero = _class(
        module,
        "FlappingBotStraightFlightMeasuredWingMultibodyIdealInverseDynamicsPhaseCoupledDeLaurierEnvCfg",
    )

    assert _ann_assign(no_aero, "wing_drive_variant").value.id == "IDEAL_INVERSE_DYNAMICS_PHASE_WING_DRIVE"
    assert ast.literal_eval(_ann_assign(no_aero, "use_kinematic_joint_override").value) is False
    assert ast.literal_eval(_ann_assign(no_aero, "enable_wing_aero").value) is False
    assert ast.literal_eval(_ann_assign(no_aero, "enable_tail_aero").value) is False
    assert _ann_assign(aero, "wing_aero_coupling_mode").value.id == "IDEAL_INVERSE_DYNAMICS_PER_WING_LINK"
    assert _ann_assign(aero, "wing_aero_acceleration_source").value.id == "PRESCRIBED_ACCELERATION"


def test_scheme_b_uses_passive_wings_and_hard_opposed_mimic() -> None:
    asset_source = _ASSET_SOURCE.read_text(encoding="utf-8")
    env_source = _ENV_SOURCE.read_text(encoding="utf-8")

    assert (
        'IDEAL_INVERSE_DYNAMICS_PHASE_WING_DRIVE = "ideal_inverse_dynamics_phase_drive"'
        in asset_source
    )
    assert (
        "IdealInverseDynamicsPhaseCoupledFlappingBotCfg = "
        "IdealTorqueCoupledFlappingBotCfg.replace()"
    ) in asset_source
    assert "IDEAL_INVERSE_DYNAMICS_PHASE_WING_DRIVE," in env_source
    assert "apply_hard_opposed_wing_mimic(" in env_source


def test_scheme_b_reads_physx_inverse_dynamics_and_applies_effort() -> None:
    module = ast.parse(_ENV_SOURCE.read_text(encoding="utf-8"))
    env_class = _class(module, "FlappingBotStraightFlightEnv")
    apply_action = next(
        node
        for node in env_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_apply_action"
    )
    source_segment = ast.get_source_segment(_ENV_SOURCE.read_text(encoding="utf-8"), apply_action)
    assert source_segment is not None

    assert "get_generalized_mass_matrices" in source_segment
    assert "get_coriolis_and_centrifugal_compensation_forces" in source_segment
    assert "get_gravity_compensation_forces" in source_segment
    assert "reduce_common_inverse_dynamics" in source_segment
    assert "compute_aerodynamic_joint_hinge_torques" in source_segment
    assert "set_joint_effort_target" in source_segment


def test_scheme_b_does_not_add_a_moving_limit_or_per_step_wing_state_write() -> None:
    source = _ENV_SOURCE.read_text(encoding="utf-8")
    start = source.index("elif wing_drive_variant == IDEAL_INVERSE_DYNAMICS_PHASE_WING_DRIVE:")
    end = source.index("\n        else:", start)
    branch = source[start:end]

    assert "set_dof_limits" not in branch
    assert "write_joint_state_to_sim" not in branch
    assert "set_joint_position_target" in branch
    assert "set_joint_effort_target" not in branch
    assert "compute_desired_common_acceleration" in branch


def test_default_plant_and_drive_names_remain_unchanged() -> None:
    source = _ENV_SOURCE.read_text(encoding="utf-8")

    assert "plant_variant: str = NEAR_SINGLE_RIGID_BODY_PLANT" in source
    assert "wing_drive_variant: str = KINEMATIC_WING_OVERRIDE" in source
