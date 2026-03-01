import math

import torch

from flapping_bot.flapping_bot.physics.tail_aero import TailAeroCfg, TailAeroModel


def test_tail_aero_elevator_and_rudder_torques_are_finite_and_signed():
    # Forward flight along +x in body frame (x forward, y left, z up).
    cfg = TailAeroCfg()
    model = TailAeroModel(cfg, device=torch.device("cpu"))

    v_b = torch.tensor([[7.0, 0.0, 0.0]]).expand(3, 3).clone()
    w_b = torch.zeros_like(v_b)

    # Elevator: positive deflection should produce a positive pitch moment (+tau_y).
    elevator = torch.tensor([0.0, math.radians(10.0), math.radians(-10.0)], dtype=torch.float32)
    rudder = torch.zeros(3, dtype=torch.float32)
    F_b, tau_b = model.compute_wrench(root_lin_vel_b=v_b, root_ang_vel_b=w_b, elevator_rad=elevator, rudder_rad=rudder)
    assert torch.isfinite(F_b).all()
    assert torch.isfinite(tau_b).all()
    assert float(tau_b[1, 1]) > 0.0
    assert float(tau_b[2, 1]) < 0.0

    # Rudder: positive deflection should produce a non-zero yaw moment (tau_z) with opposite sign for negative deflection.
    elevator = torch.zeros(3, dtype=torch.float32)
    rudder = torch.tensor([0.0, math.radians(10.0), math.radians(-10.0)], dtype=torch.float32)
    F_b, tau_b = model.compute_wrench(root_lin_vel_b=v_b, root_ang_vel_b=w_b, elevator_rad=elevator, rudder_rad=rudder)
    assert torch.isfinite(F_b).all()
    assert torch.isfinite(tau_b).all()
    assert float(tau_b[1, 2]) != 0.0
    assert float(tau_b[1, 2]) == -float(tau_b[2, 2])


def test_tail_aero_zero_speed_gives_zero_loads():
    cfg = TailAeroCfg()
    model = TailAeroModel(cfg, device=torch.device("cpu"))

    v_b = torch.zeros(2, 3)
    w_b = torch.zeros(2, 3)
    elevator = torch.tensor([math.radians(15.0), math.radians(-15.0)], dtype=torch.float32)
    rudder = torch.tensor([math.radians(15.0), math.radians(-15.0)], dtype=torch.float32)

    F_b, tau_b = model.compute_wrench(root_lin_vel_b=v_b, root_ang_vel_b=w_b, elevator_rad=elevator, rudder_rad=rudder)
    assert torch.allclose(F_b, torch.zeros_like(F_b), atol=1e-6)
    assert torch.allclose(tau_b, torch.zeros_like(tau_b), atol=1e-6)

