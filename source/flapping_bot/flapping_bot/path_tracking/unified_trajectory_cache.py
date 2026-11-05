"""Reset-time analytic tables and batched CPU Torch trajectory queries.

Cubic Hermite interpolation uses analytic spatial derivatives. No SciPy or
per-environment Python loop occurs in query/project. Geometry stays z-up/world.
"""
import math

import numpy as np
import torch

from .unified_trajectory import Trajectory


class TrajectoryCache:
    """Uniform shared arc grid for an environment batch with one configured length."""

    def __init__(self, num_envs: int, length_m: float, spacing_m: float = .2, device="cpu"):
        if not math.isfinite(length_m) or not math.isfinite(spacing_m) or min(length_m, spacing_m) <= 0:
            raise ValueError("Cache length and spacing must be finite and positive")
        self.length = length_m
        self.count = math.ceil(length_m/spacing_m)+1
        self.ds = length_m/(self.count-1)
        self.device = torch.device(device)
        self.grid = np.linspace(0, length_m, self.count)
        # Float64 preserves geometry under world translations. Env converts outputs
        # to its observation/state dtype at the boundary, not inside interpolation.
        self.values = torch.zeros(num_envs, self.count, 6, dtype=torch.float64, device=self.device)
        self.slopes = torch.zeros_like(self.values)
        self.rows = torch.arange(num_envs, device=self.device)[:, None]

    def update_rows(self, ids: torch.Tensor, paths: list[Trajectory]) -> None:
        """Populate only reset rows; SciPy work is confined to this method."""
        values, slopes = [], []
        for path in paths:
            if path.spec.length_m != self.length:
                raise ValueError("Cache rows must share configured trajectory length")
            q = path.query(self.grid)
            values.append(np.c_[q['position'], q['heading'], q['kappa'], q['gamma']])
            slopes.append(np.c_[q['tangent'], q['kappa'], q['kappa_prime'], q['gamma_prime']])
        self.values[ids] = torch.as_tensor(np.stack(values), dtype=torch.float64, device=self.device)
        self.slopes[ids] = torch.as_tensor(np.stack(slopes), dtype=torch.float64, device=self.device)

    def _interpolate(self, s: torch.Tensor, position_only: bool = False):
        s = s.to(device=self.device, dtype=torch.float64).clamp(0, self.length)
        index = torch.floor(s/self.ds).long().clamp(0, self.count-2)
        t = ((s-index.to(torch.float64)*self.ds)/self.ds).unsqueeze(-1)
        values = self.values[..., :3] if position_only else self.values
        slopes = self.slopes[..., :3] if position_only else self.slopes
        y0, y1 = values[self.rows, index], values[self.rows, index+1]
        d0, d1 = slopes[self.rows, index], slopes[self.rows, index+1]
        # Horner form: normalized-segment cubic and its spatial derivative.
        a = 2*y0-2*y1+self.ds*(d0+d1)
        b = -3*y0+3*y1-self.ds*(2*d0+d1)
        c = self.ds*d0
        value = ((a*t+b)*t+c)*t+y0
        derivative = ((3*a*t+2*b)*t+c)/self.ds
        return value, derivative

    def query(self, s: torch.Tensor) -> dict[str, torch.Tensor]:
        """s shape [env] or [env, preview]; endpoint-clamped, metre-based."""
        scalar = s.ndim == 1
        if scalar:
            s = s[:, None]
        value, derivative = self._interpolate(s)
        psi, k, gamma = value[..., 3], value[..., 4], value[..., 5]
        kp, gp = derivative[..., 4], derivative[..., 5]
        tangent = torch.stack([gamma.cos()*psi.cos(), gamma.cos()*psi.sin(), gamma.sin()], -1)
        normal = torch.stack([-psi.sin(), psi.cos(), torch.zeros_like(psi)], -1)
        up = torch.linalg.cross(tangent, normal)
        tangent_prime = (gamma.cos()*k)[..., None]*normal + gp[..., None]*up
        result = dict(position=value[..., :3], tangent=tangent, normal=normal, up=up,
                      heading=psi, kappa=k, gamma=gamma, kappa_prime=kp, gamma_prime=gp,
                      tangent_prime=tangent_prime, curvature=torch.linalg.vector_norm(tangent_prime, dim=-1))
        return {key: x[:, 0] for key, x in result.items()} if scalar else result

    def positions(self, s: torch.Tensor) -> torch.Tensor:
        """Position-only distance preview avoids unused geometry work."""
        return self._interpolate(s, position_only=True)[0]

    def project(self, points: torch.Tensor, previous_s: torch.Tensor, window_m: torch.Tensor) -> torch.Tensor:
        """Same local segment projection, including clipped search-window endpoints.

        Only the segments inside each row's window are gathered, rather than a
        full [env, entire-path] distance matrix. Invalid robot states retain s.
        """
        finite = torch.isfinite(points).all(-1) & torch.isfinite(window_m)
        previous_s = previous_s.to(torch.float64)
        window = torch.where(finite, window_m, torch.zeros_like(window_m)).to(torch.float64).clamp_min(0)
        lo = (previous_s-window).clamp(0, self.length)
        hi = (previous_s+window).clamp(0, self.length)
        first = torch.floor(lo/self.ds).long().clamp(0, self.count-2)
        last = torch.floor(hi/self.ds).long().clamp(0, self.count-2)
        width = int((last-first).max().item())+1
        index = first[:, None]+torch.arange(width, device=self.device)[None, :]
        valid = index <= last[:, None]
        s0 = torch.maximum(index.to(torch.float64)*self.ds, lo[:, None])
        s1 = torch.minimum((index.to(torch.float64)+1)*self.ds, hi[:, None])
        p0, p1 = self.positions(s0), self.positions(s1)
        delta = p1-p0
        alpha = ((points[:, None, :]-p0)*delta).sum(-1)/delta.square().sum(-1).clamp_min(1e-20)
        alpha = alpha.clamp(0, 1)
        distance = (p0+alpha[..., None]*delta-points[:, None, :]).square().sum(-1)
        distance = torch.where(valid, distance, torch.full_like(distance, float('inf')))
        best = distance.argmin(-1, keepdim=True)
        progress = (s0+alpha*(s1-s0)).gather(1, best).squeeze(1)
        return torch.where(finite, progress, previous_s).to(points.dtype)
