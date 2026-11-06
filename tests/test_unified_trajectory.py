"""Focused numerical and sampling checks independent of Isaac Sim."""
from dataclasses import replace
import numpy as np
import pytest
from flapping_bot.path_tracking.unified_trajectory import (
    GeometryLimits, SamplingConfig, Trajectory, TrajectorySpec, clamped_knots,
    generate_batch, reversal_locations, sample_trajectory, screen,
)


def constant(k=0., g=0., length=60.):
    knots = clamped_knots(length, 10.)
    n = len(knots)-4
    return TrajectorySpec(knots, (k,)*n, (g,)*n, length)


def test_straight_and_exact_helix():
    s = np.linspace(0, 60, 121)
    for path in generate_batch([constant(), constant(.03, .15)]):
        k, g = path.spec.kappa_coeff[0], path.spec.gamma_coeff[0]
        q = path.query(s)
        expected = (np.c_[s, s*0, s*0+20] if k == 0 else
                    np.c_[np.cos(g)*np.sin(k*s)/k, np.cos(g)*(1-np.cos(k*s))/k, 20+s*np.sin(g)])
        np.testing.assert_allclose(q['position'], expected, atol=2e-12)
        np.testing.assert_allclose(np.linalg.norm(q['tangent'], axis=1), 1, atol=1e-14)
        np.testing.assert_allclose(q['curvature'], abs(k*np.cos(g)), atol=1e-14)


def test_analytic_derivatives_and_frames():
    path = sample_trajectory(7, SamplingConfig(required_reversals=("kappa", "gamma")))
    s, h = np.linspace(.1, 59.9, 100), 1e-4
    q, plus, minus = path.query(s), path.query(s+h), path.query(s-h)
    for value, derivative in [('position', 'tangent'), ('tangent', 'tangent_prime'),
                              ('kappa', 'kappa_prime'), ('gamma', 'gamma_prime'), ('heading', 'kappa')]:
        np.testing.assert_allclose((plus[value]-minus[value])/(2*h), q[derivative], atol=2e-8)
    frame = np.stack([q['tangent'], q['normal'], q['up']], axis=-1)
    np.testing.assert_allclose(np.swapaxes(frame, -1, -2)@frame, np.broadcast_to(np.eye(3), frame.shape), atol=1e-14)
    for knot in np.unique(path.spec.knots)[1:-1]:
        a, b = path.query(knot-1e-7), path.query(knot+1e-7)
        for key in ('position', 'tangent', 'kappa_prime', 'gamma_prime'):
            np.testing.assert_allclose(a[key], b[key], atol=3e-7)


@pytest.mark.parametrize('name,value', [('kappa', .08), ('gamma', .5)])
def test_amplitude_rejection(name, value):
    spec = constant(**({'k': value} if name == 'kappa' else {'g': value}))
    assert name in screen(Trajectory(spec), GeometryLimits())


@pytest.mark.parametrize('name', ['kappa', 'gamma'])
def test_derivative_rejection(name):
    spec = constant(length=5.)
    spec = replace(spec, **{name+'_coeff': (0., .1, -.1, 0.)})
    limits = replace(GeometryLimits(), kappa=1., gamma=1., kappa_prime=.001, gamma_prime=.001)
    assert name+'_prime' in screen(Trajectory(spec), limits)


def test_height_intersection_and_invalid_continuity():
    assert 'height' in screen(Trajectory(constant(g=.3)), replace(GeometryLimits(), max_height=25.))
    assert 'nonlocal_intersection' in screen(Trajectory(constant(k=.2, length=40.)), replace(GeometryLimits(), kappa=1.))
    spec = constant()
    with pytest.raises(ValueError):
        Trajectory(replace(spec, knots=(0.,)*4+(20., 20., 30., 40., 50.)+(60.,)*4))


def test_unified_representative_maneuvers_and_repeatability():
    specs = [constant(), constant(g=.1), constant(g=-.1), constant(k=.02)]
    for k_sign, g_sign in [(1, 0), (0, 1), (1, 1)]:
        spec = constant()
        coeff = np.r_[np.ones(4), 0., -np.ones(4)]
        specs.append(replace(spec, kappa_coeff=tuple(.02*k_sign*coeff), gamma_coeff=tuple(.1*g_sign*coeff)))
    for path in generate_batch(specs):
        assert screen(path, GeometryLimits()) == []
    cfg = SamplingConfig(required_reversals=("kappa", "gamma"))
    for seed in (0, 7, 31):
        a, b = sample_trajectory(seed, cfg), sample_trajectory(seed, cfg)
        assert a.spec == b.spec
        assert screen(a, GeometryLimits()) == []
        events = reversal_locations(a, cfg)
        assert all(events[name] for name in cfg.required_reversals)
        assert all(.15*cfg.length_m < s < .85*cfg.length_m for roots in events.values() for s in roots)
    constant_cfg = replace(cfg, transition_strength=0., required_reversals=())
    p = sample_trajectory(3, constant_cfg)
    assert np.ptp(p.spec.kappa_coeff[4:]) == 0
    with pytest.raises(RuntimeError):
        sample_trajectory(0, replace(cfg, max_attempts=1, axis_active_probability=1., nonzero_density=1.), replace(GeometryLimits(), kappa=0.))


def test_projection_preview_endpoint():
    path = Trajectory(constant())
    assert path.project(np.array([4.3, 1., 20.]), 4., 1.) == pytest.approx(4.3)
    assert path.project(np.array([50., 0., 20.]), 4., 1.) <= 5.
    np.testing.assert_allclose(path.query([60., 70.])['position'], [[60, 0, 20]]*2)


def test_json_replay_float32_reset_origin():
    import json
    path = sample_trajectory(9, origin=np.array([10., 20., 20.], dtype=np.float32))
    record = json.loads(json.dumps(path.spec.to_dict()))
    replay = Trajectory(TrajectorySpec(**record))
    np.testing.assert_array_equal(path.positions, replay.positions)


@pytest.mark.parametrize('entry', [0., 5., 12.])
def test_horizontal_entry_is_exact_and_smooth_before_random_maneuver(entry):
    cfg = SamplingConfig(initial_straight_m=entry, required_reversals=('kappa', 'gamma'))
    path = sample_trajectory(7, cfg, heading=.8)
    q = path.query(np.linspace(0, entry, 11))
    np.testing.assert_allclose(q['tangent'], np.tile([np.cos(.8), np.sin(.8), 0.], (11, 1)), atol=1e-14)
    for key in ('kappa', 'gamma', 'kappa_prime', 'gamma_prime'):
        np.testing.assert_allclose(q[key], 0., atol=1e-14)
    for spline in (path.k, path.g):
        assert abs(spline.derivative(2)(entry)) < 1e-14
        assert abs(spline(entry+1e-4)) < 1e-10
    assert all(reversal_locations(path, cfg).values())
    assert path.sampling_attempts < 20


def test_coefficient_density_persistence_and_coupling_controls():
    for cfg in (SamplingConfig(nonzero_density=0.), SamplingConfig(axis_active_probability=0.)):
        path = sample_trajectory(3, cfg)
        assert not np.any(path.spec.kappa_coeff) and not np.any(path.spec.gamma_coeff)
    cfg = SamplingConfig(coefficient_change_probability=0., axis_active_probability=1., nonzero_density=1.)
    path = sample_trajectory(3, cfg)
    assert np.ptp(path.spec.kappa_coeff[4:]) == 0 and np.ptp(path.spec.gamma_coeff[4:]) == 0
    for coupling in (-1., 1.):
        cfg = replace(cfg, coupling=coupling, coefficient_change_probability=1.)
        path = sample_trajectory(4, cfg)
        np.testing.assert_allclose(np.array(path.spec.kappa_coeff)/cfg.kappa_amplitude*coupling,
                                   np.array(path.spec.gamma_coeff)/cfg.gamma_amplitude, atol=1e-14)
    with pytest.raises(ValueError, match='maneuver span'):
        sample_trajectory(0, SamplingConfig(initial_straight_m=55., required_reversals=('kappa',)))


def test_default_sampler_has_checkable_simple_and_complex_mass():
    # This is a geometry measurement of one shared sampler, not family selection.
    from scripts.flapping_px4.benchmark_unified_trajectory import coverage
    cfg = SamplingConfig()
    assert cfg.required_reversals == ()
    paths = [sample_trajectory(seed, cfg) for seed in range(64)]
    counts = coverage(paths, cfg)
    assert all(counts[name] > 0 for name in ('exact_straight', 'near_straight', 'one_direction',
                                            'sustained', 'reversal', 'coupled_3d')), counts
    assert max(path.sampling_attempts for path in paths) == 1
