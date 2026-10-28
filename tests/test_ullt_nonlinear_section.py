"""Independent polar, nonlinear-residual and low-angle compatibility checks."""

from dataclasses import replace
import math

import pytest
import torch

from flapping_bot.config.nonlinear_section import FlatPlatePolarConfig, flat_plate_prior
from flapping_bot.config.ullt import ULLTConfig
from flapping_bot.physics.flat_plate_section import flat_plate_polar
from flapping_bot.physics.ullt_geometry import elliptic_geometry, uniform_translation
from flapping_bot.physics.ullt_nonlinear import NonlinearConvergenceError, nonlinear_normalwash
from flapping_bot.physics.ullt_state_space import ULLTBackend


def case(angle_deg=25.0, batch=2, dtype=torch.float64):
    g = elliptic_geometry(num_sections=9, batch_size=batch, dtype=dtype)
    m = uniform_translation(g, 8.0)
    for flow in (m.flow_three_quarter_w_mps, m.flow_edge_start_w_mps, m.flow_edge_end_w_mps):
        flow[..., 2] = 8 * math.tan(math.radians(angle_deg))
    return g, m


@pytest.mark.parametrize("prior", ["early", "late"])
def test_polar_symmetry_bounds_and_low_angle(prior):
    p = flat_plate_prior(prior)
    alpha = torch.linspace(-math.pi, math.pi, 10001, dtype=torch.float64)
    r = flat_plate_polar(alpha, p, baseline_cd=0.03)
    assert torch.isfinite(r.cl).all() and torch.isfinite(r.dcl_dalpha).all()
    torch.testing.assert_close(r.cl, -r.cl.flip(0), atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(r.cd, r.cd.flip(0), atol=1e-13, rtol=1e-13)
    assert r.cd.min() >= 0.03 and r.cd.max() <= 0.03 + 2 * p.separated_lift_amplitude + 1e-12
    assert (
        r.cl.abs().max()
        <= max(2 * math.pi * math.sin(math.radians(p.transition_end_deg)), p.separated_lift_amplitude) + 1e-12
    )
    small = alpha.abs() < math.radians(p.transition_start_deg)
    assert torch.equal(r.cl[small], 2 * math.pi * alpha[small].sin())
    torch.testing.assert_close(
        flat_plate_polar(torch.tensor([math.pi / 2], dtype=alpha.dtype), p).cl,
        torch.zeros(1, dtype=alpha.dtype),
        atol=1e-14,
        rtol=0,
    )


@pytest.mark.parametrize("prior", ["early", "late"])
def test_analytic_polar_slope_and_smooth_boundaries(prior):
    p = flat_plate_prior(prior)
    a = torch.deg2rad(
        torch.tensor(
            [-48.0, -24.0, -17.0, 0.0, 17.0, 24.0, 48.0, p.transition_start_deg, p.transition_end_deg],
            dtype=torch.float64,
        )
    )
    h = 1e-6
    d = (flat_plate_polar(a + h, p).cl - flat_plate_polar(a - h, p).cl) / (2 * h)
    torch.testing.assert_close(d, flat_plate_polar(a, p).dcl_dalpha, atol=1e-7, rtol=1e-6)
    for boundary in (p.transition_start_deg, p.transition_end_deg):
        a = torch.tensor([math.radians(boundary)], dtype=torch.float64)
        assert abs(float(flat_plate_polar(a - h, p).dcl_dalpha - flat_plate_polar(a + h, p).dcl_dalpha)) < 1e-4


def test_normalwash_jacobian_independent_finite_difference():
    cfg = ULLTConfig(section_model="nonlinear_flat_plate")
    u = torch.tensor([[8.0, 8.0, 8.0]], dtype=torch.float64)
    n = u * torch.tan(torch.deg2rad(torch.tensor([[10.0, 22.0, 45.0]], dtype=u.dtype)))
    q = nonlinear_normalwash(u, n, cfg)
    h = 1e-5
    du = (nonlinear_normalwash(u + h, n, cfg)["equivalent"] - nonlinear_normalwash(u - h, n, cfg)["equivalent"]) / (
        2 * h
    )
    dn = (nonlinear_normalwash(u, n + h, cfg)["equivalent"] - nonlinear_normalwash(u, n - h, cfg)["equivalent"]) / (
        2 * h
    )
    torch.testing.assert_close(q["du"], du, atol=1e-8, rtol=1e-7)
    torch.testing.assert_close(q["dn"], dn, atol=1e-8, rtol=1e-7)
    assert q["equivalent"][0, 0] == n[0, 0] and q["du"][0, 0] == 0 and q["dn"][0, 0] == 1


@pytest.mark.parametrize("model", ["quasi_steady", "one_state", "four_state"])
def test_low_angle_backend_and_state_derivatives_match(model):
    g, m = case(3)
    linear = ULLTBackend(ULLTConfig(model=model, air_density_kg_m3=1, profile_drag_coefficient=0.03))
    nonlinear = ULLTBackend(replace(linear.config, section_model="nonlinear_flat_plate"))
    state = linear.zero_state(g)
    a = linear.evaluate(g, m, state, torch.zeros((2, 3), dtype=g.chord_m.dtype))
    b = nonlinear.evaluate(g, m, state, torch.zeros((2, 3), dtype=g.chord_m.dtype))
    for name in (
        "circulation_m2_s",
        "circulatory_force_w_n",
        "profile_force_w_n",
        "force_w_n",
        "moment_about_reference_w_nm",
    ):
        torch.testing.assert_close(getattr(a, name), getattr(b, name), atol=1e-11, rtol=1e-12)
    torch.testing.assert_close(
        linear.derivative(g, m, state), nonlinear.derivative(g, m, state), atol=1e-11, rtol=1e-12
    )
    assert b.nonlinear_iterations.max() == 0


@pytest.mark.parametrize("model", ["quasi_steady", "one_state", "four_state"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("angle", [50])
def test_batched_solve_closure_and_steady_memory(model, dtype, angle):
    g, m = case(angle, dtype=dtype)
    cfg = ULLTConfig(model=model, section_model="nonlinear_flat_plate", nonlinear_polar=flat_plate_prior("late"))
    backend = ULLTBackend(cfg)
    state = backend.steady_state(g, m)
    load = backend.evaluate(g, m, state, torch.zeros((2, 3), dtype=dtype))
    assert torch.all(load.nonlinear_residual_mps <= load.nonlinear_tolerance_mps)
    tol = 1e-3 if dtype == torch.float32 else 1e-5  # float32 residual floor amplified by 2U/c
    assert backend.derivative(g, m, state).abs().max() < tol if backend.state_count else True
    torch.testing.assert_close(load.force_w_n, load.section_force_w_n.sum(1))
    torch.testing.assert_close(load.moment_about_reference_w_nm, load.section_moment_about_reference_w_nm.sum(1))
    torch.testing.assert_close(load.force_w_n[0], load.force_w_n[1])
    # Check nonlinear residual independently, using the publicly returned physical flow.
    physical = m.flow_three_quarter_w_mps + load.induced_velocity_w_mps
    u = -(physical * g.chord_forward_w).sum(-1)
    n = (physical * g.normal_w).sum(-1)
    cl = flat_plate_polar(torch.atan2(n, u), cfg.nonlinear_polar).cl
    target = 0.5 * g.chord_m * torch.sqrt(u * u + n * n) * cl
    torch.testing.assert_close(load.circulation_m2_s, target, atol=tol, rtol=tol)


def test_iteration_failure_is_explicit_and_six_state_not_extended():
    with pytest.raises(ValueError, match="supports"):
        ULLTConfig(model="six_state", section_model="nonlinear_flat_plate")
    with pytest.raises(ValueError):
        FlatPlatePolarConfig(30, 15, 1)
    g, m = case(35)
    backend = ULLTBackend(
        ULLTConfig(
            model="quasi_steady",
            section_model="nonlinear_flat_plate",
            nonlinear_max_iterations=1,
            nonlinear_max_restarts=0,
        )
    )
    with pytest.raises(NonlinearConvergenceError, match="residual_mps"):
        backend.evaluate(g, m, backend.zero_state(g), torch.zeros((2, 3), dtype=g.chord_m.dtype))


@pytest.mark.parametrize("model", ["one_state", "four_state"])
def test_multiple_root_steady_initialization_preserves_memory_root(model):
    g, m = case(40, dtype=torch.float64)
    backend = ULLTBackend(
        ULLTConfig(model=model, section_model="nonlinear_flat_plate", nonlinear_polar=flat_plate_prior("late"))
    )
    state = backend.steady_state(g, m)
    assert backend.derivative(g, m, state).abs().max() < 1e-5


def test_quasi_steady_force_scale_drag_and_reference_shift():
    """Independent Kutta-Joukowski magnitude and translation-of-moment checks."""
    g, m = case(50)
    cfg = ULLTConfig(
        model="quasi_steady", section_model="nonlinear_flat_plate", nonlinear_polar=flat_plate_prior("late")
    )
    backend = ULLTBackend(cfg)
    ref = torch.zeros((2, 3), dtype=g.chord_m.dtype)
    load = backend.evaluate(g, m, backend.zero_state(g), ref)
    flow = m.flow_three_quarter_w_mps + load.induced_velocity_w_mps
    speed = torch.linalg.vector_norm(m.flow_three_quarter_w_mps, dim=-1)
    v = torch.linalg.vector_norm(flow, dim=-1)
    expected = cfg.air_density_kg_m3 * speed * load.circulation_m2_s * g.width_m
    torch.testing.assert_close(
        torch.linalg.vector_norm(load.circulatory_force_w_n, dim=-1), expected.abs(), atol=1e-5, rtol=1e-6
    )
    torch.testing.assert_close((load.circulatory_force_w_n * flow).sum(-1), torch.zeros_like(speed), atol=1e-10, rtol=0)
    expected_drag = 0.5 * cfg.air_density_kg_m3 * g.chord_m * g.width_m * speed * v * load.section_cd
    torch.testing.assert_close(torch.linalg.vector_norm(load.profile_force_w_n, dim=-1), expected_drag)
    assert ((load.profile_force_w_n * flow).sum(-1) >= 0).all()
    shift = torch.tensor([[0.2, -0.1, 0.3]], dtype=ref.dtype).expand_as(ref)
    shifted = backend.evaluate(g, m, backend.zero_state(g), shift)
    torch.testing.assert_close(
        shifted.moment_about_reference_w_nm,
        load.moment_about_reference_w_nm - torch.linalg.cross(shift, load.force_w_n),
    )
    linear = ULLTBackend(replace(cfg, section_model="linear"))
    baseline = linear.evaluate(g, m, linear.zero_state(g), ref)
    assert torch.equal(load.added_mass_force_w_n, baseline.added_mass_force_w_n)
    assert torch.equal(load.added_mass_couple_three_quarter_w_nm, baseline.added_mass_couple_three_quarter_w_nm)


@pytest.mark.parametrize("model", ["one_state", "four_state"])
def test_heterogeneous_batch_matches_independent_environments(model):
    g, m = case(50)
    for flow in (m.flow_three_quarter_w_mps, m.flow_edge_start_w_mps, m.flow_edge_end_w_mps):
        flow[0, :, 2] = 8 * math.tan(math.radians(3))
    cfg = ULLTConfig(model=model, section_model="nonlinear_flat_plate")
    backend = ULLTBackend(cfg)
    combined = backend.evaluate(g, m, backend.zero_state(g), g.chord_m.new_zeros((2, 3)))
    for i, angle in enumerate((3, 50)):
        single_g, single_m = case(angle, batch=1)
        single = backend.evaluate(single_g, single_m, backend.zero_state(single_g), g.chord_m.new_zeros((1, 3)))
        torch.testing.assert_close(combined.force_w_n[i], single.force_w_n[0], atol=1e-9, rtol=1e-10)
    assert combined.nonlinear_iterations[0] == 0
