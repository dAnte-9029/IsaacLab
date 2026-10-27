"""Analytical and independent frequency-domain checks for stage-1 ULLT."""

import math
import numpy as np
import torch
from dataclasses import replace
from pathlib import Path

import pytest
from flapping_bot.config.ullt import ULLTConfig
from flapping_bot.physics.ullt_geometry import (
    elliptic_geometry,
    load_csv_urdf_geometry,
    opposed_flapping_motion,
    rigid_motion,
    rotation_xyz,
    uniform_translation,
)
from flapping_bot.physics.ullt_state_space import (
    ULLTBackend,
    ULLTGeometry,
    ULLTState,
    added_mass_wrench,
    downwash_correction,
    trailer_influence,
)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def pitched(n=9, angle=0.05, dtype=torch.float64, device="cpu", batch=1):
    base = elliptic_geometry(num_sections=n, dtype=dtype, device=device, batch_size=batch)
    angles = base.chord_m.new_zeros((batch, 3))
    angles[:, 1] = -angle
    zeros = torch.zeros_like(angles)
    speed = zeros.clone()
    speed[:, 0] = 1
    return rigid_motion(
        base,
        rotation_w_from_reference=rotation_xyz(angles),
        origin_w_m=zeros,
        origin_velocity_w_mps=speed,
        origin_acceleration_w_mps2=zeros,
        angular_velocity_w_rad_s=zeros,
        angular_acceleration_w_rad_s2=zeros,
    )


def reference(geometry):
    return geometry.chord_m.new_zeros((geometry.chord_m.shape[0], 3))


def test_trailer_induction_matches_biot_savart_scalar_formula():
    g = elliptic_geometry(num_sections=7)
    m = uniform_translation(g, 1)
    w = trailer_influence(g, m, ULLTConfig())
    y = g.quarter_chord_w_m[0, :, 1]
    a, b = g.edge_start_w_m[0, :, 1], g.edge_end_w_m[0, :, 1]
    expected = (1 / (y[:, None] - b) - 1 / (y[:, None] - a)) / (4 * math.pi)
    torch.testing.assert_close(w[0, :, :, 2], expected)
    assert (w[0, :, :, 2].diagonal() < 0).all()
    assert torch.count_nonzero(w[..., :2]) == 0


@pytest.mark.parametrize("model", ["one_state", "four_state", "six_state", "quasi_steady"])
def test_steady_state_stationary_and_prandtl_limit(model):
    g, m = pitched(n=127, angle=1e-4)
    backend = ULLTBackend(ULLTConfig(model=model))
    state = backend.steady_state(g, m)
    torch.testing.assert_close(backend.derivative(g, m, state), torch.zeros_like(state.values_mps), atol=2e-11, rtol=0)
    loads = backend.evaluate(g, m, state, reference(g))
    b = 3 * math.pi / 4
    area = math.pi * b / 4
    slope = float(loads.force_w_n[0, 2]) / (0.5 * 1.225 * area * math.sin(1e-4))
    assert slope == pytest.approx(2 * math.pi / (1 + 2 / 3), rel=0.005)
    assert loads.force_w_n[0, 0] < 0  # induced drag on a stationary pitched wing
    assert abs(float(loads.force_w_n[0, 1])) < 1e-12


@pytest.mark.parametrize("model", ["one_state", "four_state"])
def test_two_dimensional_impulsive_start_matches_indicial_exponentials(model):
    # A single practically infinite-span strip eliminates finite-wing coupling.
    g = ULLTGeometry(
        torch.tensor([[[0.0, -5e8, 0.0]]], dtype=torch.float64),
        torch.tensor([[[0.0, 5e8, 0.0]]], dtype=torch.float64),
        torch.zeros((1, 1, 3), dtype=torch.float64),
        torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float64),
        torch.ones((1, 1), dtype=torch.float64),
    )
    m = uniform_translation(g, 1)
    flow = m.flow_three_quarter_w_mps.clone()
    flow[..., 2] = 0.02
    m = replace(m, flow_three_quarter_w_mps=flow)
    backend = ULLTBackend(ULLTConfig(model=model, include_added_mass=False))
    state = backend.zero_state(g)
    for k in range(100):
        state = backend.advance(state, lambda t: (g, m), k * 0.01, 0.01)
    tau = 2 * math.sqrt(1 + 0.02**2)
    if model == "one_state":
        expected_phi = 1 - 0.5 * math.exp(-0.25 * tau)
        expected_gamma = 1 - 0.8 * math.exp(-0.25 * tau)
    else:
        expected_phi = 1 - 0.165 * math.exp(-0.045 * tau) - 0.335 * math.exp(-0.300 * tau)
        expected_gamma = 1 + sum(
            a * math.exp(b * tau) for a, b in zip([-0.177, -0.609, 0.170, -0.259], [-0.045, -0.300, -0.433, -1.419])
        )
    loads = backend.evaluate(g, m, state, reference(g))
    magnitude = torch.linalg.vector_norm(loads.circulatory_force_w_n, dim=-1)
    assert float(magnitude[0, 0] / (0.5 * 1.225 * 2 * math.pi * math.sqrt(1.0004) * 0.02 * 1e9)) == pytest.approx(
        expected_phi, rel=1e-8
    )
    assert float(loads.circulation_m2_s[0, 0] / (math.pi * 0.02)) == pytest.approx(expected_gamma, rel=1e-8)


@pytest.mark.parametrize("model", ["one_state", "four_state"])
def test_against_independent_complex_frequency_domain_solution(model):
    # Derive harmonic response directly from transfer functions, not backend RHS.
    g = elliptic_geometry(num_sections=5)
    c = g.chord_m[0].numpy()
    y = g.quarter_chord_w_m[0, :, 1].numpy()
    a = g.edge_start_w_m[0, :, 1].numpy()
    b = g.edge_end_w_m[0, :, 1].numpy()
    w = (1 / (y[:, None] - b) - 1 / (y[:, None] - a)) / (4 * np.pi)
    omega, amplitude = 2.0, 1e-6
    if model == "one_state":
        ac, poles, ax = np.array([-0.8]), np.array([-0.25]), np.array([-0.5])
    else:
        ac, poles = np.array([-0.177, -0.609, 0.170, -0.259]), np.array([-0.045, -0.300, -0.433, -1.419])
        ax = np.array([-0.165, -0.335, 0.170, -0.259])
    decay = -2 / c[:, None] * poles
    transfer = 1 + (ac * 1j * omega / (decay + 1j * omega)).sum(-1)
    gamma = np.linalg.solve(np.diag(1 / (np.pi * c)) - transfer[:, None] * w, transfer * amplitude)
    ve = amplitude + w @ gamma
    x = -ax * decay / (decay + 1j * omega) * ve[:, None]
    backend = ULLTBackend(ULLTConfig(model=model, include_added_mass=False))
    state = ULLTState(torch.from_numpy(x.imag.copy()).unsqueeze(0))
    base_motion = uniform_translation(g, 1)

    def sample(t):
        flow = base_motion.flow_three_quarter_w_mps.clone()
        flow[..., 2] = amplitude * math.sin(omega * t)
        return g, replace(base_motion, flow_three_quarter_w_mps=flow)

    period, steps = 2 * math.pi / omega, 128
    errors = []
    for index in range(steps):
        time = index * period / steps
        observed = backend.evaluate(*sample(time), state, reference(g)).circulation_m2_s[0].numpy()
        errors.append(np.max(np.abs(observed - (gamma * np.exp(1j * omega * time)).imag)))
        state = backend.advance(state, sample, time, period / steps)
    assert max(errors) / np.max(np.abs(gamma)) < 2e-5


def test_downwash_correction_steady_cancellation_and_fit():
    g = elliptic_geometry(num_sections=9)
    cfg = ULLTConfig(model="six_state")
    w0, w1, w2, residual = downwash_correction(g, cfg)
    aw = (
        torch.tensor([-0.165, -0.335], dtype=torch.float64)
        * torch.exp(torch.tensor([0.135, 0.9], dtype=torch.float64))
        / (4 * math.pi)
    )
    gamma = torch.linspace(0.1, 1, 9, dtype=torch.float64)[None]
    xw = -(2 * gamma / g.chord_m)[..., None] * aw
    result = w0 @ gamma[..., None] + w1 @ xw[..., 0, None] + w2 @ xw[..., 1, None]
    torch.testing.assert_close(result, torch.zeros_like(result), atol=1e-14, rtol=0)
    assert float(residual.max()) < 0.006


def test_downwash_correction_matches_independent_finite_vortex_loop_kernel():
    # Unit circulation step at one source: compare spatial wake correction
    # against the exact rectangular-loop Biot-Savart kernel (paper Eq. 70).
    g = elliptic_geometry(num_sections=7)
    cfg = ULLTConfig(model="six_state")
    w0, w1, w2, _ = downwash_correction(g, cfg)
    source = 3
    chord = float(g.chord_m[0, source])
    tau = torch.linspace(0, 15, 301, dtype=torch.float64)
    a = (
        torch.tensor([-0.165, -0.335], dtype=torch.float64)
        * torch.exp(torch.tensor([0.135, 0.9], dtype=torch.float64))
        / (4 * math.pi)
    )
    b = torch.tensor([-0.045, -0.300], dtype=torch.float64)
    decay = torch.exp(tau[:, None] * b)
    # The source states grow from zero to -2*A/c; add W0 instantaneous term.
    x = -2 / chord * a * (1 - decay)
    actual = w0[0, :, source, None] + w1[0, :, source, None] * x[:, 0] + w2[0, :, source, None] * x[:, 1]
    y = g.quarter_chord_w_m[0, :, 1]
    dl = y - g.edge_start_w_m[0, source, 1]
    dr = y - g.edge_end_w_m[0, source, 1]
    length = chord / 2 * (1 + tau)

    def half(d):
        return (torch.sqrt(length.square() + d[:, None].square()) / length - 1) / (4 * math.pi * d[:, None])

    target = half(dr) - half(dl)
    # Remove the separate 2D wake already represented by sectional states.
    target[source] -= 2 / chord * (decay * a).sum(-1)
    assert float(torch.sqrt((actual - target).square().mean())) < 0.012


def test_added_mass_heave_and_anti_munk_moment():
    g = elliptic_geometry(num_sections=1, mid_chord_m=0.4)
    m = uniform_translation(g, 2)
    acceleration = torch.zeros_like(m.acceleration_w_mps2)
    acceleration[..., 2] = 3
    m = replace(m, acceleration_w_mps2=acceleration)
    cfg = ULLTConfig(added_mass_factor=0.85)
    force, couple = added_mass_wrench(g, m, cfg)
    mass = 1.225 * math.pi * 0.85 * 0.2**2 * float(g.width_m[0, 0])
    assert float(force[0, 0, 2]) == pytest.approx(-mass * 3)
    # At 3/4 chord, the mid-chord added-mass force is c/4 forward of origin.
    assert float(couple[0, 0, 1]) == pytest.approx(mass * 3 * 0.1)
    flow = m.flow_three_quarter_w_mps.clone()
    flow[..., 2] = -0.2
    moving = replace(m, flow_three_quarter_w_mps=flow, acceleration_w_mps2=torch.zeros_like(acceleration))
    _, steady_couple = added_mass_wrench(g, moving, cfg)
    torch.testing.assert_close(steady_couple, torch.zeros_like(steady_couple), atol=1e-14, rtol=0)


def test_frame_covariance_and_moment_reference_translation():
    g, m = pitched()
    backend = ULLTBackend(ULLTConfig(model="six_state"))
    state = backend.steady_state(g, m)
    output = backend.evaluate(g, m, state, reference(g))
    shift = torch.tensor([[0.3, -0.2, 0.4]], dtype=torch.float64)
    moved = backend.evaluate(g, m, state, shift)
    torch.testing.assert_close(
        moved.moment_about_reference_w_nm,
        output.moment_about_reference_w_nm - torch.linalg.cross(shift, output.force_w_n),
    )
    rotation = rotation_xyz(torch.tensor([0.3, -0.2, 0.8], dtype=torch.float64))

    def rotate(v):
        return v @ rotation.T

    transformed = ULLTGeometry(
        *(rotate(v) + shift[:, None] for v in (g.edge_start_w_m, g.edge_end_w_m, g.quarter_chord_w_m)),
        rotate(g.chord_forward_w),
        g.chord_m
    )
    motion = type(m)(*(rotate(v) for v in vars(m).values()))
    result = backend.evaluate(transformed, motion, state, shift)
    torch.testing.assert_close(result.force_w_n, rotate(output.force_w_n))
    torch.testing.assert_close(
        result.moment_about_reference_w_nm, rotate(output.moment_about_reference_w_nm), atol=1e-12, rtol=1e-9
    )


def test_partial_reset_isolation_and_evaluation_does_not_advance():
    g, m = pitched(batch=3)
    backend = ULLTBackend(ULLTConfig(model="six_state"))
    state = backend.steady_state(g, m)
    original = state.values_mps.clone()
    backend.evaluate(g, m, state, reference(g))
    assert torch.equal(original, state.values_mps)
    reset = state.reset(torch.tensor([1]))
    assert torch.count_nonzero(reset.values_mps[1]) == 0
    assert torch.equal(reset.values_mps[[0, 2]], original[[0, 2]])
    assert torch.equal(original, state.values_mps)
    with pytest.raises(ValueError):
        state.reset(torch.tensor([-1]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_and_batch_consistency(dtype):
    g, m = pitched(dtype=dtype, batch=2)
    backend = ULLTBackend(ULLTConfig(model="six_state"))
    state = backend.steady_state(g, m)
    output = backend.evaluate(g, m, state, reference(g))
    assert output.force_w_n.dtype == dtype
    torch.testing.assert_close(output.force_w_n[0], output.force_w_n[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable in this runtime")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_cuda_parity(dtype):
    g, m = pitched(dtype=dtype, batch=2)
    backend = ULLTBackend(ULLTConfig(model="six_state"))
    state = backend.steady_state(g, m)
    output = backend.evaluate(g, m, state, reference(g))
    gg = type(g)(*(v.cuda() for v in vars(g).values()))
    mm = type(m)(*(v.cuda() for v in vars(m).values()))
    result = backend.evaluate(gg, mm, ULLTState(state.values_mps.cuda()), reference(gg))
    torch.testing.assert_close(result.force_w_n.cpu(), output.force_w_n, atol=1e-5, rtol=1e-5)


def test_fail_closed_inputs_and_swept_six_state_gate():
    g, m = pitched()
    backend = ULLTBackend(ULLTConfig(model="six_state"))
    zero = torch.zeros_like(m.flow_three_quarter_w_mps)
    with pytest.raises(ValueError, match="Near-zero"):
        backend.evaluate(g, replace(m, flow_three_quarter_w_mps=zero), backend.zero_state(g), reference(g))
    with pytest.raises(ValueError, match="dtype"):
        backend.evaluate(g, m, ULLTState(backend.zero_state(g).values_mps.float()), reference(g))
    shifted = g.quarter_chord_w_m.clone()
    shifted[:, 1, 0] += 0.05
    swept = replace(g, quarter_chord_w_m=shifted)
    with pytest.raises(ValueError, match="unswept"):
        backend.steady_state(swept, m)
    single = ULLTBackend(ULLTConfig(model="one_state"))
    assert torch.isfinite(single.evaluate(swept, m, single.zero_state(swept), reference(g)).force_w_n).all()


def test_csv_urdf_geometry_keeps_complete_root_local_wings_and_gap():
    root = Path(__file__).resolve().parents[1]
    csv_path = root / "outputs_DeLaurier/right_wing_te_fit_poly5_gap50.csv"
    urdf = root / "source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf"
    geometry = load_csv_urdf_geometry(csv_path, urdf, sections_per_wing=10)
    assert geometry.chord_m.shape == (1, 20)
    torch.testing.assert_close(
        geometry.edge_end_w_m[0, 9], torch.tensor([-0.25 * 0.300083212, -0.056, 0.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        geometry.edge_start_w_m[0, 10], torch.tensor([-0.25 * 0.300083212, 0.056, 0.0], dtype=torch.float64)
    )
    assert float(geometry.edge_end_w_m[0, -1, 1]) == pytest.approx(0.056 + 0.7 * math.cos(0.019391))
    assert (geometry.quarter_chord_w_m[0, :, 1].abs() > 0.056).all()
    backend = ULLTBackend()
    motion = uniform_translation(geometry, 8)
    assert torch.isfinite(
        backend.evaluate(geometry, motion, backend.zero_state(geometry), reference(geometry)).force_w_n
    ).all()
    with pytest.raises(ValueError, match="unswept"):
        ULLTBackend(ULLTConfig(model="six_state")).steady_state(geometry, motion)
    # Nonzero dynamic loads must respect polar-vector mirror symmetry, too.
    origins = geometry.chord_m.new_tensor([[[0, -0.056, 0], [0, 0.056, 0]]])
    g, m = opposed_flapping_motion(geometry, origins, time_s=0.03, speed_mps=8, frequency_hz=4, amplitude_rad=0.5)
    state = backend.steady_state(g, m)
    loads = backend.evaluate(g, m, state, reference(g))
    reflection = geometry.chord_m.new_tensor([1, -1, 1])
    torch.testing.assert_close(loads.section_force_w_n[0, :10], loads.section_force_w_n[0, 10:].flip(0) * reflection)
    assert float(loads.force_w_n[0, 2].abs()) > 0.1
    assert float(loads.force_w_n[0, 1].abs()) < 1e-10
    assert float(loads.moment_about_reference_w_nm[0, [0, 2]].abs().max()) < 1e-10


def test_profile_drag_is_separate_and_dissipative_and_power_is_mechanical():
    g = elliptic_geometry()
    m = uniform_translation(g, 3)
    backend = ULLTBackend(ULLTConfig(profile_drag_coefficient=0.02))
    out = backend.evaluate(g, m, backend.zero_state(g), reference(g))
    drag = 0.5 * 1.225 * 9 * 0.02 * float((g.chord_m * g.width_m).sum())
    assert float(out.force_w_n[0, 0]) == pytest.approx(-drag)
    assert float(out.power_input_w[0]) == pytest.approx(3 * drag)
    assert torch.count_nonzero(out.circulatory_force_w_n) == 0


def test_rk4_uses_intermediate_motion_and_preserves_input_state():
    from flapping_bot.analysis.ullt_validation import paper_motion

    geometry = elliptic_geometry(num_sections=5)
    sample, _, _ = paper_motion("pitch", geometry)
    backend = ULLTBackend(ULLTConfig(model="six_state"))
    original = backend.zero_state(geometry)
    coarse = backend.advance(original, sample, 0, 0.01)
    fine = original
    for i in range(4):
        fine = backend.advance(fine, sample, i * 0.0025, 0.0025)
    assert torch.count_nonzero(original.values_mps) == 0
    assert torch.linalg.vector_norm(coarse.values_mps) > 0
    torch.testing.assert_close(coarse.values_mps, fine.values_mps, atol=1e-9, rtol=1e-5)


def test_singular_trailer_nonfinite_geometry_and_reverse_flow_rejected():
    g = elliptic_geometry()
    m = uniform_translation(g, 1)
    backend = ULLTBackend()
    with pytest.raises(ValueError, match="Singular"):
        backend.evaluate(replace(g, quarter_chord_w_m=g.edge_start_w_m), m, backend.zero_state(g), reference(g))
    with pytest.raises(ValueError, match="Non-finite"):
        backend.evaluate(replace(g, chord_m=g.chord_m * float("nan")), m, backend.zero_state(g), reference(g))
    with pytest.raises(ValueError, match="reverse"):
        backend.evaluate(g, uniform_translation(g, -1), backend.zero_state(g), reference(g))


@pytest.mark.parametrize("case", ["surge", "pitch", "heave_pitch"])
def test_prescribed_spatial_kinematics_against_finite_differences(case):
    from flapping_bot.analysis.ullt_validation import paper_motion

    sample, _, _ = paper_motion(case, elliptic_geometry())
    t, epsilon = 0.17, 1e-5
    before, mb = sample(t - epsilon)
    current, mc = sample(t)
    after, ma = sample(t + epsilon)
    v_fd = (after.three_quarter_chord_w_m - before.three_quarter_chord_w_m) / (2 * epsilon)
    a_fd = -(ma.flow_three_quarter_w_mps - mb.flow_three_quarter_w_mps) / (2 * epsilon)
    torch.testing.assert_close(v_fd, -mc.flow_three_quarter_w_mps, atol=2e-9, rtol=2e-8)
    torch.testing.assert_close(a_fd, mc.acceleration_w_mps2, atol=2e-9, rtol=2e-8)


def test_added_mass_heave_pitch_cross_terms_are_reciprocal():
    g = elliptic_geometry(num_sections=1, mid_chord_m=0.4)
    m = uniform_translation(g, 2)
    acceleration = torch.zeros_like(m.acceleration_w_mps2)
    angular_acceleration = torch.zeros_like(acceleration)
    acceleration[..., 2] = 3
    angular_acceleration[..., 1] = 5
    m = replace(m, acceleration_w_mps2=acceleration, angular_acceleration_w_rad_s2=angular_acceleration)
    force, moment = added_mass_wrench(g, m, ULLTConfig(added_mass_factor=0.85))
    width = float(g.width_m[0, 0])
    m33 = 1.225 * math.pi * 0.85 * 0.2**2 * width
    m35 = m33 * 0.1
    m55 = 1.225 * math.pi / 8 * 0.2**4 * width + m33 * 0.1**2
    assert float(force[0, 0, 2]) == pytest.approx(-m33 * 3 + m35 * 5)
    assert float(moment[0, 0, 1]) == pytest.approx(m35 * 3 - m55 * 5)
