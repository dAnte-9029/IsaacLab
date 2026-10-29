"""Physics contracts: directions, finite-wing induction, link kinematics and moments."""
from dataclasses import replace
import math
import torch
import pytest
from flapping_bot.physics.control_aero_3d import *
from flapping_bot.physics.control_aero_kinematics import LinkKinematics3D
from flapping_bot.physics.ullt_geometry import elliptic_geometry, uniform_translation, rotation_xyz
from flapping_bot.analysis.crosswind_aero import CSV, URDF, CG


def sample(model="L3D", alpha=0.05, beta=0.0, speed=8.0):
    g = elliptic_geometry(num_sections=20, aspect_ratio=6, mid_chord_m=0.2)
    m = uniform_translation(g, speed)
    f = m.flow_three_quarter_w_mps.clone()
    f[..., 1] = -beta
    f[..., 2] = speed * alpha
    m = replace(m, flow_three_quarter_w_mps=f, flow_edge_start_w_mps=f, flow_edge_end_w_mps=f)
    return g, m, ControlAero3D(ControlAero3DConfig(backend=model, include_added_mass=False, span_friction_cd=0))


@pytest.mark.parametrize("model", ["L3D", "I3D"])
def test_lift_drag_circulation_and_low_speed(model):
    g, m, b = sample(model)
    out = b.evaluate(g, m, torch.zeros(1, 3, dtype=torch.float64))
    assert out.force.sum(1)[0, 2] > 0
    assert (out.gamma < 0).all() if model == "I3D" else True
    assert out.residual.max() < 1e-12
    # In still air no fictitious clamped-speed load.
    g, m, b = sample(model, speed=0)
    out = b.evaluate(g, m, torch.zeros(1, 3, dtype=torch.float64))
    assert out.force.abs().max() == 0
    for speed in (0.001, 0.1, -8.0):
        g, m, b = sample(model, speed=speed)
        assert torch.isfinite(b.evaluate(g, m, torch.zeros(1, 3, dtype=torch.float64)).force).all()


def test_induced_lift_reduction_and_drag():
    outputs = []
    for model in ("L3D", "I3D"):
        g, m, b = sample(model)
        o = b.evaluate(g, m, torch.zeros(1, 3, dtype=torch.float64))
        outputs.append(o.force.sum(1)[0])
    assert 0 < outputs[1][2] < outputs[0][2]
    # Remove geometric lift projection: induced force reduces propulsive x.
    assert outputs[1][0] < outputs[0][0]


def test_ideal_planar_first_order_sideslip():
    result = []
    for beta in (-0.001, 0.001):
        g, m, b = sample(beta=beta)
        result.append(b.evaluate(g, m, torch.zeros(1, 3, dtype=torch.float64)).force)
    torch.testing.assert_close(*result, atol=1e-12, rtol=0)


@pytest.mark.parametrize("model", ["L3D", "I3D"])
def test_reference_and_rotation(model):
    g, m, b = sample(model)
    ref = torch.zeros(1, 3, dtype=torch.float64)
    o = b.evaluate(g, m, ref)
    shift = ref.new_tensor([[0.3, -0.1, 0.2]])
    q = b.evaluate(g, m, shift)
    torch.testing.assert_close(q.moment, o.moment - torch.linalg.cross(shift[:, None].expand_as(o.force), o.force))
    r = rotation_xyz(ref.new_tensor([0.4, -0.3, 0.1]))
    gg = ULLTGeometry(*(v if k == "chord_m" else v @ r.T for k, v in vars(g).items()))
    mm = ULLTMotion(*(v @ r.T for v in vars(m).values()))
    oo = b.evaluate(gg, mm, ref)
    torch.testing.assert_close(oo.force, o.force @ r.T, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(oo.moment, o.moment @ r.T, atol=1e-10, rtol=1e-10)


def test_drag_dissipation_and_polar_continuity():
    v = torch.tensor([[8.0, 2.0, -1.0]], dtype=torch.float64)
    w = v * 0
    ref = CG[None]
    f, _ = distributed_body_drag(v, w, ref)
    assert (f * v).sum() < 0
    for angle in (15, 30, 90, 180):
        a = torch.tensor([math.radians(angle) - 1e-7, math.radians(angle) + 1e-7])
        p = flat_plate_polar(a, FlatPlatePolarConfig())
        assert abs(p.cl[0] - p.cl[1]) < 1e-5


def test_link_material_velocity_twist_derivatives_and_batch_reset():
    cfg = ControlAero3DConfig(twist_amplitude_deg=35)
    adapter = LinkKinematics3D(CSV, URDF, cfg, dtype=torch.float64)
    v = torch.tensor([[8.0, 0.5, -0.2], [8.0, 0.5, -0.2]], dtype=torch.float64)
    om = v * 0
    phase = torch.tensor([0.7, 0.7], dtype=torch.float64)
    g, m, rel = adapter.prescribed(v, om, CG[None].expand(2, -1), phase, 3.0)
    h = 1e-5
    gp, mp, _ = adapter.prescribed(v, om, CG[None].expand(2, -1), phase + 2 * math.pi * 3 * h, 3.0)
    gm, mm, _ = adapter.prescribed(v, om, CG[None].expand(2, -1), phase - 2 * math.pi * 3 * h, 3.0)
    numerical_v = (gp.three_quarter_chord_w_m - gm.three_quarter_chord_w_m) / (2 * h)
    torch.testing.assert_close(-m.flow_three_quarter_w_mps - v[:, None], numerical_v, atol=1e-6, rtol=1e-6)
    numerical_a = -(mp.flow_three_quarter_w_mps - mm.flow_three_quarter_w_mps) / (2 * h)
    torch.testing.assert_close(rel.acceleration_w_mps2, numerical_a, atol=1e-4, rtol=1e-6)
    backend = ControlAero3D(cfg)
    first = backend.evaluate(g, m, CG[None].expand(2, -1), rel)
    phase[0] = 0  # independently reset first environment
    g, m, rel = adapter.prescribed(v, om, CG[None].expand(2, -1), phase, 3.0)
    second = backend.evaluate(g, m, CG[None].expand(2, -1), rel)
    torch.testing.assert_close(first.force[1], second.force[1])


@pytest.mark.parametrize("model", ["L3D", "I3D"])
def test_real_wing_mirror_and_sum(model):
    cfg = ControlAero3DConfig(backend=model, twist_amplitude_deg=35)
    kin = LinkKinematics3D(CSV, URDF, cfg, dtype=torch.float64)
    v = torch.tensor([[8.0, 0.7, 0.0], [8.0, -0.7, 0.0]], dtype=torch.float64)
    omega = torch.tensor([[0.2, 0.0, 0.1], [-0.2, 0.0, -0.1]], dtype=torch.float64)
    ref = CG[None].expand(2, -1).clone()
    ref[:, 1] = 0
    g, m, rel = kin.prescribed(v, omega, ref, torch.full((2,), 0.65, dtype=torch.float64), 3.0)
    out = ControlAero3D(cfg).evaluate(g, m, ref, rel)
    torch.testing.assert_close(
        out.force[0].sum(0), out.force[1].sum(0) * v.new_tensor([1, -1, 1]), atol=1e-10, rtol=1e-10
    )
    torch.testing.assert_close(
        out.moment[0].sum(0), out.moment[1].sum(0) * v.new_tensor([-1, 1, -1]), atol=1e-10, rtol=1e-10
    )
    # Translate each separate wing to its own arbitrary COM then reconstruct.
    n = cfg.sections_per_wing
    centers = v.new_tensor([[-0.08, -0.3, 0.04], [-0.08, 0.3, 0.04]])
    f = out.force.reshape(2, 2, n, 3).sum(2)
    mb = out.moment.reshape(2, 2, n, 3).sum(2)
    mc = mb - torch.linalg.cross(centers[None] - ref[:, None], f)
    torch.testing.assert_close((mc + torch.linalg.cross(centers[None] - ref[:, None], f)).sum(1), out.moment.sum(1))


def test_elliptic_finite_wing_lift_slope():
    g = elliptic_geometry(num_sections=80, aspect_ratio=6, mid_chord_m=0.2)
    m = uniform_translation(g, 8.0)
    f = m.flow_three_quarter_w_mps.clone()
    f[..., 2] = 0.008
    m = replace(m, flow_three_quarter_w_mps=f)
    o = ControlAero3D(ControlAero3DConfig(backend="I3D", include_added_mass=False)).evaluate(
        g, m, torch.zeros(1, 3, dtype=torch.float64)
    )
    slope = float(o.force.sum(1)[0, 2] / (0.5 * 1.225 * 64 * (g.chord_m * g.width_m).sum() * 0.001))
    assert abs(slope / (2 * math.pi * 6 / 8) - 1) < 0.03
