"""Generate representative geometry and a standalone figure; no simulator or training."""
import argparse
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flapping_bot.path_tracking.unified_trajectory import Trajectory, TrajectorySpec, clamped_knots, sample_trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    knots = clamped_knots(60., 10.)
    n = len(knots)-4
    base = TrajectorySpec(knots, (0.,)*n, (0.,)*n, 60.)
    reverse = np.r_[np.ones(4), 0., -np.ones(4)]
    recipes = {'straight': base, 'climb': replace(base, gamma_coeff=(.1,)*n),
               'descent': replace(base, gamma_coeff=(-.1,)*n), 'turn': replace(base, kappa_coeff=(.02,)*n),
               'left-right': replace(base, kappa_coeff=tuple(.02*reverse)),
               'climb-descent': replace(base, gamma_coeff=tuple(.12*reverse)),
               'coupled 3D': sample_trajectory(7).spec}
    fig = plt.figure(figsize=(14, 9), layout='constrained')
    ax = fig.add_subplot(2, 2, (1, 3), projection='3d')
    ak, ag = fig.add_subplot(2, 2, 2), fig.add_subplot(2, 2, 4)
    for name, spec in recipes.items():
        path = Trajectory(spec)
        s = np.linspace(0, spec.length_m, 601)
        q = path.query(s)
        line, = ax.plot(*q['position'].T, label=name)
        ak.plot(s, q['kappa'], color=line.get_color(), label=name)
        ag.plot(s, np.rad2deg(q['gamma']), color=line.get_color())
    ax.set(xlabel='x [m]', ylabel='y [m]', zlabel='z [m]', title='Unified cubic B-spline trajectories (z-up)')
    ax.legend(fontsize=8)
    ak.set(xlabel='3D arc length s [m]', ylabel='d heading / ds [rad/m]', title='Heading rate (not Frenet curvature)')
    ag.set(xlabel='3D arc length s [m]', ylabel='slope [deg]', title='Path slope')
    for axis in (ak, ag):
        axis.axvspan(9, 51, color='grey', alpha=.08)
        axis.grid(alpha=.3)
    fig.savefig(args.output/'trajectories.png', dpi=170)
    fig.savefig(args.output/'trajectories.svg')
    (args.output/'tasks.json').write_text(json.dumps({k: v.to_dict() for k, v in recipes.items()}, indent=2))
    print(args.output/'trajectories.png')


if __name__ == '__main__':
    main()
