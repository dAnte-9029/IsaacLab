"""Independent geometric, derivative and symmetry checks for LE twist."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from flapping_bot.analysis.ullt_flight_replay import flight_kinematics
from flapping_bot.physics.ullt_geometry import load_csv_urdf_geometry
from flapping_bot.physics.ullt_state_space import ULLTBackend, ULLTGeometry
from flapping_bot.physics.ullt_twist import PrescribedTwistConfig, twist_wing
from flapping_bot.config.ullt import ULLTConfig


def geometry():
    root = Path(__file__).resolve().parents[1]
    return load_csv_urdf_geometry(
        root / "outputs_DeLaurier/right_wing_te_fit_poly5_gap50.csv",
        root / "source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf",
        sections_per_wing=6,
    )


def sample(t, config, amplitude=math.pi / 6):
    x = torch.zeros((1, 16), dtype=torch.float64)
    x[0, 0] = 1.1 + 12 * t + 1.5 * t * t
    x[0, 1], x[0, 2], x[0, 3] = 12 + 3 * t, 3, 8
    hinges = torch.tensor([[0, -0.056, 0], [0, 0.056, 0]], dtype=x.dtype)
    return flight_kinematics(geometry(), hinges, x, amplitude, config)


def test_zero_amplitude_exact_baseline():
    a = sample(0, PrescribedTwistConfig())
    b = sample(0, PrescribedTwistConfig(0, -0.5, -25))
    for first, second in zip(a, b):
        for name in vars(first):
            assert torch.equal(getattr(first, name), getattr(second, name))


def test_geometric_twist_about_le_and_shared_edges():
    g = geometry()
    local = ULLTGeometry(*(getattr(g, name)[:, 6:] for name in vars(g)))
    root = torch.tensor([0, 0.056, 0], dtype=torch.float64)
    z = torch.zeros(1, dtype=torch.float64)
    actual, motion = twist_wing(local, root, z, z, z, PrescribedTwistConfig(35, -0.5, 0))
    axis = np.array([0, np.cos(-0.019391), np.sin(-0.019391)])
    radius = np.max(np.abs((local.edge_end_w_m.numpy()[0] - root.numpy()) @ axis))
    for key in ("quarter_chord_w_m", "edge_start_w_m", "edge_end_w_m"):
        points = getattr(local, key).numpy()[0]
        expected = []
        for p in points:
            station = (p - root.numpy()) @ axis
            leading_edge = root.numpy() + station * axis
            s = abs(station) / radius
            theta = np.deg2rad(35) * (1.5 * s - 0.5 * s * s)
            expected.append(leading_edge + Rotation.from_rotvec(-theta * axis).apply(p - leading_edge))
        np.testing.assert_allclose(getattr(actual, key).numpy()[0], expected, atol=1e-12)
    torch.testing.assert_close(actual.edge_end_w_m[:, :-1], actual.edge_start_w_m[:, 1:])
    assert motion.acceleration_w_mps2.abs().max() == 0


def test_twist_flap_point_and_frame_derivatives():
    cfg = PrescribedTwistConfig(35, -0.5, -25)
    h = 1e-5
    gm, mm = sample(-h, cfg)
    g, m = sample(0, cfg)
    gp, mp = sample(h, cfg)
    translation = torch.tensor([8, 0, 0], dtype=torch.float64)
    for point, flow in (
        ("three_quarter_chord_w_m", "flow_three_quarter_w_mps"),
        ("edge_start_w_m", "flow_edge_start_w_mps"),
        ("edge_end_w_m", "flow_edge_end_w_mps"),
    ):
        velocity = (getattr(gp, point) - getattr(gm, point)) / (2 * h)
        torch.testing.assert_close(-getattr(m, flow), velocity + translation, atol=2e-7, rtol=2e-7)
    acceleration = -(mp.flow_three_quarter_w_mps - mm.flow_three_quarter_w_mps) / (2 * h)
    torch.testing.assert_close(m.acceleration_w_mps2, acceleration, atol=2e-6, rtol=2e-7)
    for name in ("chord_forward_w", "span_w", "normal_w"):
        derivative = (getattr(gp, name) - getattr(gm, name)) / (2 * h)
        torch.testing.assert_close(
            derivative, torch.linalg.cross(m.angular_velocity_w_rad_s, getattr(g, name)), atol=2e-7, rtol=2e-7
        )
    torch.testing.assert_close(
        m.angular_acceleration_w_rad_s2,
        (mp.angular_velocity_w_rad_s - mm.angular_velocity_w_rad_s) / (2 * h),
        atol=2e-6,
        rtol=2e-7,
    )


def test_mirrored_twist_loads_are_symmetric():
    g, m = sample(0, PrescribedTwistConfig(35, -0.5, -25))
    backend = ULLTBackend(ULLTConfig(model="four_state"))
    loads = backend.evaluate(g, m, backend.zero_state(g), torch.zeros((1, 3), dtype=torch.float64))
    assert abs(loads.force_w_n[0, 1]) < 1e-10
    assert abs(loads.moment_about_reference_w_nm[0, 0]) < 1e-10
    assert abs(loads.moment_about_reference_w_nm[0, 2]) < 1e-10


def test_twist_transport_with_body_rotation_and_batch():
    cfg = PrescribedTwistConfig(35, -0.5, -25)
    w = np.array([0.4, -0.7, 0.3])
    velocity = np.array([8.0, 0.2, -0.3])
    hinges = torch.tensor([[0, -0.056, 0], [0, 0.056, 0]], dtype=torch.float64)

    def moving(t):
        rotation = Rotation.from_rotvec(w * t).as_matrix()
        x = torch.zeros((2, 16), dtype=torch.float64)
        x[:, 0] = 1.1 + 12 * t + 1.5 * t * t
        x[:, 1], x[:, 2] = 12 + 3 * t, 3
        x[:, 3:6] = torch.from_numpy(rotation.T @ velocity)
        x[:, 9:12] = torch.from_numpy(w)
        g, m = flight_kinematics(geometry(), hinges, x, math.pi / 6, cfg)
        for value in vars(m).values():
            torch.testing.assert_close(value[0], value[1], atol=0, rtol=0)
        p = g.three_quarter_chord_w_m.numpy()[0] @ rotation.T + velocity * t
        v = -m.flow_three_quarter_w_mps.numpy()[0] @ rotation.T
        a = m.acceleration_w_mps2.numpy()[0] @ rotation.T
        return p, v, a

    h = 1e-5
    pm, vm, _ = moving(-h)
    _, v, a = moving(0)
    pp, vp, _ = moving(h)
    np.testing.assert_allclose(v, (pp - pm) / (2 * h), atol=2e-7, rtol=2e-7)
    np.testing.assert_allclose(a, (vp - vm) / (2 * h), atol=2e-6, rtol=2e-7)


@pytest.mark.parametrize("args", [(41, 0, 0), (35, 2, 0), (35, 0, float("nan"))])
def test_reject_invalid_twist(args):
    with pytest.raises(ValueError):
        PrescribedTwistConfig(*args)
