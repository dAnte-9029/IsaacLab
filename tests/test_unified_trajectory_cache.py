"""Analytic oracle checks for cached queries/projection and reset isolation."""
from dataclasses import replace

import numpy as np
import torch

from flapping_bot.path_tracking.unified_trajectory import SamplingConfig, Trajectory, reversal_locations, sample_trajectory
from flapping_bot.path_tracking.unified_trajectory_cache import TrajectoryCache


def fixture_paths():
    cfg = SamplingConfig(required_reversals=('kappa', 'gamma'))
    return [sample_trajectory(seed, cfg, origin=(123.4*i, -57.*i, 20.), heading=.7*i)
            for i, seed in enumerate((0, 7, 31))]


def test_cache_matches_analytic_geometry_and_spatial_derivatives():
    paths = fixture_paths()
    cache = TrajectoryCache(3, 60.)
    cache.update_rows(torch.arange(3), paths)
    s = torch.linspace(.031, 59.967, 227, dtype=torch.float64).expand(3, -1)
    got = cache.query(s)
    exact = [p.query(s[0].numpy()) for p in paths]
    for key in got:
        tolerance = 2e-6 if key == 'position' else 2e-7
        np.testing.assert_allclose(got[key], np.stack([q[key] for q in exact]), atol=tolerance, rtol=0)
    h = 1e-4
    derivative = (cache.positions(s+h)-cache.positions(s-h))/(2*h)
    torch.testing.assert_close(derivative, got['tangent'], atol=2e-6, rtol=0)
    for key in ('kappa', 'gamma'):
        derivative = (cache.query(s+h)[key]-cache.query(s-h)[key])/(2*h)
        torch.testing.assert_close(derivative, got[key+'_prime'], atol=1e-9, rtol=0)
    endpoints = cache.positions(torch.tensor([[-1., 60., 80.]]*3))
    for i, path in enumerate(paths):
        np.testing.assert_allclose(endpoints[i], path.query([-1, 60, 80])['position'], atol=1e-10)


def test_mid_route_reversal_projection_preview_and_local_window():
    paths = fixture_paths()
    cache = TrajectoryCache(3, 60.)
    cache.update_rows(torch.arange(3), paths)
    cfg = SamplingConfig()
    roots = torch.tensor([reversal_locations(path, cfg)['kappa'][0] for path in paths], dtype=torch.float64)
    for offset in (-1., 0., 1.):
        s = roots+offset
        q = cache.query(s)
        points = q['position']+.15*q['normal']
        projected = cache.project(points, s-.2, torch.full((3,), .7))
        expected = [path.project(point.numpy(), float(x-.2), .7) for path, point, x in zip(paths, points, s)]
        np.testing.assert_allclose(projected, expected, atol=1e-5, rtol=0)
        preview_s = projected[:, None]+torch.tensor([2., 5., 10.])
        preview = cache.positions(preview_s)
        for i, path in enumerate(paths):
            np.testing.assert_allclose(preview[i], path.query(preview_s[i].numpy())['position'], atol=2e-6)
    far_points = cache.query(torch.full((3,), 50.))['position']
    projected = cache.project(far_points, torch.full((3,), 10.), torch.full((3,), .5))
    assert torch.all((projected >= 9.5) & (projected <= 10.5))


def test_partial_cache_reset_and_no_scipy_in_hot_path(monkeypatch):
    paths = fixture_paths()
    cache = TrajectoryCache(3, 60.)
    cache.update_rows(torch.arange(3), paths)
    protected = cache.values[1:].clone()
    moved = Trajectory(replace(paths[0].spec, origin=(50., 70., 30.)))
    cache.update_rows(torch.tensor([0]), [moved])
    torch.testing.assert_close(cache.values[1:], protected, atol=0, rtol=0)
    def forbidden(*args, **kwargs):
        raise AssertionError('Analytic per-env query called inside policy hot path')
    monkeypatch.setattr(Trajectory, 'query', forbidden)
    monkeypatch.setattr(Trajectory, 'project', forbidden)
    s = torch.tensor([20., 30., 40.])
    p = cache.query(s)['position']
    projected = cache.project(p, s, torch.full((3,), .7))
    torch.testing.assert_close(projected, s.double(), atol=1e-8, rtol=0)
    p[1, 0] = float('nan')
    projected = cache.project(p, s, torch.full((3,), .7))
    assert projected[1] == s[1]
