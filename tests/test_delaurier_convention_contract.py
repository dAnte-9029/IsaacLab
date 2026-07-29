from __future__ import annotations

from pathlib import Path


STRAIGHT_FLIGHT_ENV = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "flapping_bot"
    / "flapping_bot"
    / "direct"
    / "flapping_bot"
    / "straight_flight_env.py"
)


def test_delaurier_theta_a_explicitly_converts_isaac_flu_to_internal_frd() -> None:
    source = STRAIGHT_FLIGHT_ENV.read_text(encoding="utf-8")

    assert 'body_air_velocity_to_delaurier_section_velocity(v_air_b, body_frame="FLU")' in source
    assert "theta_a_env = compute_delaurier_axis_incidence(" in source
    assert 'body_frame="FLU"' in source
    assert "theta_a_env = torch.atan2" not in source


def test_delaurier_dynamic_twist_is_explicitly_opt_in() -> None:
    source = STRAIGHT_FLIGHT_ENV.read_text(encoding="utf-8")

    assert 'dynamic_twist_mode: str = "disabled"' in source
    assert "dynamic_twist_tip_amplitude_deg: float = 0.0" in source
    assert "compute_delaurier_dynamic_twist(" in source
    assert "theta = twist_kinematics.theta" in source
    assert "thetad = twist_kinematics.theta_dot" in source
    assert "thetadd = twist_kinematics.theta_ddot" in source
    assert "delaurier_enable_prescribed_twist" not in source


def test_environment_uses_sine_stroke_and_quadrature_delaurier_phase() -> None:
    source = STRAIGHT_FLIGHT_ENV.read_text(encoding="utf-8")

    assert "compute_prescribed_flap_kinematics(" in source
    assert "convention=self.cfg.flap_phase_convention" in source
    assert "flap_phase_convention: str = MECHANICAL_SINE_NEUTRAL_UPSTROKE" in source
    assert "map_symmetric_flap_coordinate_to_joint_space(" in source
    assert "h = -q.view(B, 1) * y" in source
    assert "dynamic_twist_phase_direction: float = 1.0" in source
    assert "dynamic_twist_phase_offset_deg: float = -90.0" in source
    assert "current_phase=current_phase" in source
