from __future__ import annotations

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ASSET_FILE = ROOT / "source/flapping_bot/flapping_bot/assets/ideal_coupled_drive.py"
ENV_FILE = ROOT / "source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py"
DIRECT_INIT_FILE = ROOT / "source/flapping_bot/flapping_bot/direct/flapping_bot/__init__.py"
PACKAGE_INIT_FILE = ROOT / "source/flapping_bot/flapping_bot/__init__.py"


def _class(module: ast.Module, name: str) -> ast.ClassDef:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found")


def _assignment(class_node: ast.ClassDef, name: str) -> ast.AnnAssign:
    for node in class_node.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            return node
    raise AssertionError(f"field {name} not found on {class_node.name}")


def _literal_assignment(class_node: ast.ClassDef, name: str):
    return ast.literal_eval(_assignment(class_node, name).value)


def test_gpu_implicit_configs_change_only_the_backend_and_c2a_stage() -> None:
    module = ast.parse(ENV_FILE.read_text(encoding="utf-8"))
    gpu_c1 = _class(module, "FlappingBotStraightFlightDeLaurierMeasuredPureRLGpuImplicitEnvCfg")
    gpu_c2a = _class(module, "FlappingBotStraightFlightDeLaurierMeasuredPureRLC2aGpuImplicitEnvCfg")

    assert [base.id for base in gpu_c1.bases if isinstance(base, ast.Name)] == [
        "FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg"
    ]
    assert isinstance(_assignment(gpu_c1, "wing_drive_variant").value, ast.Name)
    assert _assignment(gpu_c1, "wing_drive_variant").value.id == "IDEAL_COUPLED_WING_DRIVE"
    assert isinstance(_assignment(gpu_c1, "wing_aero_coupling_mode").value, ast.Name)
    assert _assignment(gpu_c1, "wing_aero_coupling_mode").value.id == "ACTUAL_PER_WING_LINK"
    assert isinstance(_assignment(gpu_c1, "wing_aero_acceleration_source").value, ast.Name)
    assert _assignment(gpu_c1, "wing_aero_acceleration_source").value.id == "ACTUAL_JOINT_ACCELERATION"

    sim = _assignment(gpu_c1, "sim").value
    scene = _assignment(gpu_c1, "scene").value
    robot = _assignment(gpu_c1, "robot").value
    assert isinstance(sim, ast.Call) and isinstance(sim.func, ast.Name) and sim.func.id == "SimulationCfg"
    assert isinstance(scene, ast.Call) and isinstance(scene.func, ast.Name) and scene.func.id == "FlappingRoomSceneCfg"
    assert isinstance(robot, ast.Call) and isinstance(robot.func, ast.Attribute)
    assert isinstance(robot.func.value, ast.Name) and robot.func.value.id == "IdealCoupledFlappingBotCfg"

    sim_keywords = {keyword.arg: keyword.value for keyword in sim.keywords}
    scene_keywords = {keyword.arg: keyword.value for keyword in scene.keywords}
    assert ast.literal_eval(sim_keywords["device"]) == "cuda:0"
    assert ast.literal_eval(scene_keywords["replicate_physics"]) is True
    assert any(
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Attribute)
        and isinstance(node.targets[0].value, ast.Name)
        and node.targets[0].value.id == "scene"
        and node.targets[0].attr == "robot"
        and isinstance(node.value, ast.Constant)
        and node.value.value is None
        for node in gpu_c1.body
    )
    assert "retain_accelerations=False" in ast.unparse(robot)

    forbidden_policy_fields = {
        "action_interface",
        "tail_aero_deflection_source",
        "observation_space",
        "min_flap_hz",
        "max_flap_hz",
        "frequency_governor_maximum_rise_rate_hz_per_s",
        "frequency_governor_maximum_fall_rate_hz_per_s",
        "pure_rl_reward_cfg",
        "terminate_ground_height",
        "terminate_tilt_deg",
        "terminate_abs_y",
        "pure_rl_terminate_abs_height_error_m",
    }
    gpu_c1_fields = {
        node.target.id
        for node in gpu_c1.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert gpu_c1_fields.isdisjoint(forbidden_policy_fields)

    assert [base.id for base in gpu_c2a.bases if isinstance(base, ast.Name)] == [
        "FlappingBotStraightFlightDeLaurierMeasuredPureRLGpuImplicitEnvCfg"
    ]
    assert _literal_assignment(gpu_c2a, "pure_rl_longitudinal_stage_id") == "c2a"
    assert len([node for node in gpu_c2a.body if isinstance(node, ast.AnnAssign)]) == 1


def test_gpu_implicit_configs_are_exported_from_the_project_package() -> None:
    for file_path in (DIRECT_INIT_FILE, PACKAGE_INIT_FILE):
        text = file_path.read_text(encoding="utf-8")
        assert "FlappingBotStraightFlightDeLaurierMeasuredPureRLGpuImplicitEnvCfg" in text
        assert "FlappingBotStraightFlightDeLaurierMeasuredPureRLC2aGpuImplicitEnvCfg" in text


def test_phase_matched_implicit_drive_is_explicit_and_preserves_the_gpu_baseline() -> None:
    asset_text = ASSET_FILE.read_text(encoding="utf-8")
    env_text = ENV_FILE.read_text(encoding="utf-8")
    module = ast.parse(env_text)
    baseline = _class(module, "FlappingBotStraightFlightDeLaurierMeasuredPureRLGpuImplicitEnvCfg")
    phase_matched = _class(
        module,
        "FlappingBotStraightFlightDeLaurierMeasuredPureRLGpuPhaseMatchedEnvCfg",
    )
    phase_matched_c2a = _class(
        module,
        "FlappingBotStraightFlightDeLaurierMeasuredPureRLC2aGpuPhaseMatchedEnvCfg",
    )

    assert 'PHASE_MATCHED_IMPLICIT_WING_DRIVE = "phase_matched_implicit_drive"' in asset_text
    assert _assignment(baseline, "wing_drive_variant").value.id == "IDEAL_COUPLED_WING_DRIVE"
    assert [base.id for base in phase_matched.bases if isinstance(base, ast.Name)] == [
        "FlappingBotStraightFlightDeLaurierMeasuredPureRLGpuImplicitEnvCfg"
    ]
    assert _assignment(phase_matched, "wing_drive_variant").value.id == (
        "PHASE_MATCHED_IMPLICIT_WING_DRIVE"
    )
    assert _literal_assignment(
        phase_matched,
        "phase_matched_implicit_extra_lead_fraction",
    ) == pytest.approx(0.18)
    assert _assignment(phase_matched, "wing_aero_acceleration_source").value.id == (
        "PRESCRIBED_ACCELERATION"
    )
    assert len([node for node in phase_matched.body if isinstance(node, ast.AnnAssign)]) == 3
    assert [base.id for base in phase_matched_c2a.bases if isinstance(base, ast.Name)] == [
        "FlappingBotStraightFlightDeLaurierMeasuredPureRLGpuPhaseMatchedEnvCfg"
    ]
    assert _literal_assignment(phase_matched_c2a, "pure_rl_longitudinal_stage_id") == "c2a"
    assert "phase_matched_actuator_kinematics = compute_opposed_wing_kinematics" in env_text
    assert "actuator_q_cmd = phase_matched_actuator_kinematics.common_position_rad" in env_text
    assert "actuator_qd_cmd = phase_matched_actuator_kinematics.common_velocity_rad_s" in env_text
    assert "self._qdd_cmd = ideal_inverse_phase_step.next_kinematics" not in env_text
    assert "flap_position_rad=actuator_q_cmd" in env_text
    assert "flap_velocity_rad_s=actuator_qd_cmd" in env_text
    assert "phase_matched_implicit_extra_lead_fraction" in env_text
    assert "phase_matched_extra_lead_s" in env_text


def test_phase_matched_gpu_configs_are_exported_from_the_project_package() -> None:
    for file_path in (DIRECT_INIT_FILE, PACKAGE_INIT_FILE):
        text = file_path.read_text(encoding="utf-8")
        assert "FlappingBotStraightFlightDeLaurierMeasuredPureRLGpuPhaseMatchedEnvCfg" in text
        assert "FlappingBotStraightFlightDeLaurierMeasuredPureRLC2aGpuPhaseMatchedEnvCfg" in text


def test_environment_exposes_per_environment_reward_terms_for_selected_case_diagnostics() -> None:
    text = ENV_FILE.read_text(encoding="utf-8")

    assert "self._debug_last_pure_rl_reward_terms" in text
    for field_name in (
        "total_reward",
        "path_reward",
        "progress_reward",
        "velocity_reward",
        "roll_reward",
        "angular_rate_reward",
        "pitch_envelope_penalty",
        "flap_penalty",
        "frequency_slew_penalty",
        "tail_action_delta_penalty",
        "tail_action_limit_penalty",
    ):
        assert f'"{field_name}": terms.{field_name}' in text
