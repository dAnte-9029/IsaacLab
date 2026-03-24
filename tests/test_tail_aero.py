import math

import pytest
import torch

from flapping_bot.flapping_bot.physics.tail_aero import TailAeroCfg, TailAeroModel


def _forward_flight_inputs(batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    v_b = torch.tensor([[7.0, 0.0, 0.0]], dtype=torch.float32).expand(batch_size, 3).clone()
    w_b = torch.zeros_like(v_b)
    return v_b, w_b


def test_tail_aero_config_exposes_five_surfaces():
    cfg = TailAeroCfg()

    assert cfg.fixed_horizontal.name == "fixed_horizontal"
    assert cfg.left_elevon.name == "left_elevon"
    assert cfg.right_elevon.name == "right_elevon"
    assert cfg.fixed_vertical.name == "fixed_vertical"
    assert cfg.rudder.name == "rudder"

    assert cfg.fixed_horizontal.area == pytest.approx(0.0539)
    assert cfg.left_elevon.area == pytest.approx(0.02039)
    assert cfg.right_elevon.area == pytest.approx(0.02039)
    assert cfg.fixed_vertical.area == pytest.approx(0.01265)
    assert cfg.rudder.area == pytest.approx(0.0101404)

    assert cfg.left_elevon.lever_arm_body[1] > 0.0
    assert cfg.right_elevon.lever_arm_body[1] < 0.0
    assert cfg.fixed_vertical.lever_arm_body[2] > 0.0
    assert cfg.rudder.lever_arm_body[2] > 0.0

    for surf in (
        cfg.fixed_horizontal,
        cfg.left_elevon,
        cfg.right_elevon,
        cfg.fixed_vertical,
        cfg.rudder,
    ):
        assert surf.cl_alpha_per_rad > 0.0


def test_tail_aero_symmetric_elevons_generate_pitch_without_roll():
    cfg = TailAeroCfg()
    model = TailAeroModel(cfg, device=torch.device("cpu"))

    v_b, w_b = _forward_flight_inputs(batch_size=1)
    left = torch.tensor([math.radians(10.0)], dtype=torch.float32)
    right = torch.tensor([math.radians(10.0)], dtype=torch.float32)
    rudder = torch.zeros(1, dtype=torch.float32)

    F_b, tau_b = model.compute_wrench(
        root_lin_vel_b=v_b,
        root_ang_vel_b=w_b,
        left_elevon_rad=left,
        right_elevon_rad=right,
        rudder_rad=rudder,
    )

    assert torch.isfinite(F_b).all()
    assert torch.isfinite(tau_b).all()
    assert float(tau_b[0, 1]) > 0.0
    assert abs(float(tau_b[0, 0])) < 1.0e-5


def test_tail_aero_differential_elevons_generate_opposite_roll_moments():
    cfg = TailAeroCfg()
    model = TailAeroModel(cfg, device=torch.device("cpu"))

    v_b, w_b = _forward_flight_inputs(batch_size=2)
    left = torch.tensor([math.radians(10.0), math.radians(-10.0)], dtype=torch.float32)
    right = torch.tensor([math.radians(-10.0), math.radians(10.0)], dtype=torch.float32)
    rudder = torch.zeros(2, dtype=torch.float32)

    _, tau_b = model.compute_wrench(
        root_lin_vel_b=v_b,
        root_ang_vel_b=w_b,
        left_elevon_rad=left,
        right_elevon_rad=right,
        rudder_rad=rudder,
    )

    assert float(tau_b[0, 0]) != 0.0
    assert float(tau_b[0, 0]) == pytest.approx(-float(tau_b[1, 0]), rel=1.0e-5, abs=1.0e-6)
    assert abs(float(tau_b[0, 1])) < 0.01 * abs(float(tau_b[0, 0]))
    assert abs(float(tau_b[1, 1])) < 0.01 * abs(float(tau_b[1, 0]))


def test_tail_aero_rudder_generates_yaw():
    cfg = TailAeroCfg()
    model = TailAeroModel(cfg, device=torch.device("cpu"))

    v_b, w_b = _forward_flight_inputs(batch_size=3)
    left = torch.zeros(3, dtype=torch.float32)
    right = torch.zeros(3, dtype=torch.float32)
    rudder = torch.tensor([0.0, math.radians(10.0), math.radians(-10.0)], dtype=torch.float32)

    _, tau_b = model.compute_wrench(
        root_lin_vel_b=v_b,
        root_ang_vel_b=w_b,
        left_elevon_rad=left,
        right_elevon_rad=right,
        rudder_rad=rudder,
    )

    assert float(tau_b[1, 2]) != 0.0
    assert float(tau_b[1, 2]) == pytest.approx(-float(tau_b[2, 2]), rel=1.0e-5, abs=1.0e-6)


def test_tail_aero_zero_speed_gives_zero_loads():
    cfg = TailAeroCfg()
    model = TailAeroModel(cfg, device=torch.device("cpu"))

    v_b = torch.zeros(2, 3)
    w_b = torch.zeros(2, 3)
    left = torch.tensor([math.radians(15.0), math.radians(-15.0)], dtype=torch.float32)
    right = torch.tensor([math.radians(15.0), math.radians(-15.0)], dtype=torch.float32)
    rudder = torch.tensor([math.radians(15.0), math.radians(-15.0)], dtype=torch.float32)

    F_b, tau_b = model.compute_wrench(
        root_lin_vel_b=v_b,
        root_ang_vel_b=w_b,
        left_elevon_rad=left,
        right_elevon_rad=right,
        rudder_rad=rudder,
    )
    assert torch.allclose(F_b, torch.zeros_like(F_b), atol=1e-6)
    assert torch.allclose(tau_b, torch.zeros_like(tau_b), atol=1e-6)
