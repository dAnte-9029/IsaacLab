import math

import torch

from flapping_bot.flapping_bot.physics.qsm_wang2016 import WingGeometry, compute_aero_wrench
from flapping_bot.flapping_bot.physics.virtual_twist import (
    VirtualTwistCfg,
    VirtualTwistState,
    solve_quasi_static_eta_tip,
    step_virtual_twist,
)


def _geom(**overrides):
    base = {
        "R": 0.10,
        "N": 50,
        "c": 0.05,
        "dhat": 0.25,
        "aspect_ratio": 2.0,
    }
    base.update(overrides)
    return WingGeometry.from_input(base)


def test_shapes_and_translation_sign():
    # Choose ω_yc and ω_zc both non-zero so α̃ = 45deg and translation force is non-zero.
    geom = _geom()
    F_c, tau_c = compute_aero_wrench(
        phi=0.0,
        theta=0.0,
        eta=0.0,
        phid=1.0,
        thetad=1.0,
        etad=0.0,
        phidd=0.0,
        thetadd=0.0,
        etadd=0.0,
        wing_geom=geom,
        rho=1.225,
        include_wagner=False,
    )
    assert tuple(F_c.shape) == (3,)
    assert tuple(tau_c.shape) == (3,)
    # ω_zc = φ̇ > 0 => sgn(ω_zc)=+1 => Fy_trans is negative due to eq. (2.11).
    assert float(F_c[1]) < 0.0
    assert float(tau_c[2]) < 0.0


def test_translation_at_90deg_aoa_is_finite_and_nonzero():
    # With θ=η=0 and φ̇!=0: ω_zc=φ̇, ω_yc=0 => α̃=π/2.
    # Using eq. (2.6) and (2.10) together, CF_y tends to 2A (finite), so translation force should be non-zero.
    geom = _geom()
    F_c, tau_c = compute_aero_wrench(
        phi=0.0,
        theta=0.0,
        eta=0.0,
        phid=2.0,
        thetad=0.0,
        etad=0.0,
        phidd=0.0,
        thetadd=0.0,
        etadd=0.0,
        wing_geom=geom,
        rho=1.225,
        include_wagner=False,
    )
    assert torch.isfinite(F_c).all()
    assert torch.isfinite(tau_c).all()
    assert float(F_c[1]) != 0.0


def test_rotation_and_coupling_zero_when_omega_x_is_zero():
    # ω_xc = η̇ - φ̇ sinθ. Set η̇=0, φ̇=0 => ω_xc=0.
    geom = _geom(enable_translation=False, enable_added_mass=False)
    F_c, tau_c = compute_aero_wrench(
        phi=0.0,
        theta=0.0,
        eta=0.0,
        phid=0.0,
        thetad=2.0,
        etad=0.0,
        phidd=0.0,
        thetadd=0.0,
        etadd=0.0,
        wing_geom=geom,
        rho=1.225,
        include_wagner=False,
    )
    assert torch.allclose(F_c, torch.zeros(3), atol=1e-6)
    assert torch.allclose(tau_c, torch.zeros(3), atol=1e-6)


def test_added_mass_zero_when_accelerations_zero():
    # With θ=η=0, φ̇ non-zero gives ω_zc != 0 but ω_xc=ω_yc=0.
    # With φ̈=0 => α_zc=0, α_xc=0 => added-mass term should be zero.
    geom = _geom(enable_translation=False, enable_rotation=False, enable_coupling=False, enable_added_mass=True)
    F_c, tau_c = compute_aero_wrench(
        phi=0.0,
        theta=0.0,
        eta=0.0,
        phid=1.0,
        thetad=0.0,
        etad=0.0,
        phidd=0.0,
        thetadd=0.0,
        etadd=0.0,
        wing_geom=geom,
        rho=1.225,
        include_wagner=False,
    )
    assert torch.allclose(F_c, torch.zeros(3), atol=1e-6)
    assert torch.allclose(tau_c, torch.zeros(3), atol=1e-6)


def test_distributed_twist_shape_zero_masks_eta():
    # If eta_shape is all zeros, a non-zero eta_tip should behave like eta=0 everywhere.
    geom0 = _geom(eta_shape=[0.0] * 50)
    F0, T0 = compute_aero_wrench(
        phi=0.1,
        theta=0.2,
        eta=0.0,
        phid=1.0,
        thetad=0.5,
        etad=2.0,
        phidd=0.0,
        thetadd=0.0,
        etadd=0.0,
        wing_geom=geom0,
        rho=1.225,
        include_wagner=False,
    )
    F1, T1 = compute_aero_wrench(
        phi=0.1,
        theta=0.2,
        eta=0.7,  # should be masked out by eta_shape
        phid=1.0,
        thetad=0.5,
        etad=2.0,
        phidd=0.0,
        thetadd=0.0,
        etadd=0.0,
        wing_geom=geom0,
        rho=1.225,
        include_wagner=False,
    )
    assert torch.allclose(F0, F1, atol=1e-6)
    assert torch.allclose(T0, T1, atol=1e-6)


def test_subspan_matches_full_for_full_range():
    geom = _geom()
    sub = geom.subspan(0.0, geom.R)
    F0, T0 = compute_aero_wrench(
        phi=0.0,
        theta=0.0,
        eta=0.2,
        phid=1.0,
        thetad=0.7,
        etad=0.3,
        phidd=0.0,
        thetadd=0.0,
        etadd=0.0,
        wing_geom=geom,
        rho=1.225,
        include_wagner=False,
    )
    F1, T1 = compute_aero_wrench(
        phi=0.0,
        theta=0.0,
        eta=0.2,
        phid=1.0,
        thetad=0.7,
        etad=0.3,
        phidd=0.0,
        thetadd=0.0,
        etadd=0.0,
        wing_geom=sub,
        rho=1.225,
        include_wagner=False,
    )
    assert torch.allclose(F0, F1, atol=1e-6)
    assert torch.allclose(T0, T1, atol=1e-6)


def test_virtual_twist_quasi_static():
    cfg = VirtualTwistCfg(mode="quasi_static", stiffness=2.0)
    state = VirtualTwistState.zeros((), device=torch.device("cpu"), dtype=torch.float32)
    eta, etad, etadd = step_virtual_twist(state, tau_x=torch.tensor(1.0), dt=0.01, cfg=cfg)
    assert torch.allclose(eta, torch.tensor(0.5))
    assert torch.allclose(etad, torch.tensor(0.0))
    assert torch.allclose(etadd, torch.tensor(0.0))


def test_solve_quasi_static_eta_tip_fixed_point():
    cfg = VirtualTwistCfg(mode="quasi_static", stiffness=2.0, eta_limit=1.0)
    # tau(eta) = 1 - eta, solution: eta = (1-eta)/2 => 3/2 eta = 1/2 => eta=1/3
    def tau_of_eta(eta):
        return 1.0 - eta

    eta = solve_quasi_static_eta_tip(torch.tensor(0.0), tau_of_eta=tau_of_eta, cfg=cfg, max_iters=10, relax=1.0)
    assert torch.allclose(eta, torch.tensor(1.0 / 3.0), atol=1e-3)
