"""Bounded geometry-only coverage/query benchmark; no simulator or training."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from flapping_bot.path_tracking.unified_trajectory import SamplingConfig, reversal_locations, sample_trajectory
from flapping_bot.path_tracking.unified_trajectory_cache import TrajectoryCache


def coverage(paths, cfg):
    """Nonexclusive measured geometry labels, used only for diagnostics."""
    counts = dict(exact_straight=0, near_straight=0, one_direction=0, sustained=0, reversal=0,
                  double_reversal=0, coupled_3d=0)
    for path in paths:
        s = np.linspace(cfg.initial_straight_m, cfg.length_m, 551)
        q = path.query(s)
        k = q['kappa']/cfg.kappa_amplitude
        g = q['gamma']/cfg.gamma_amplitude
        peaks = np.array([np.max(abs(k)), np.max(abs(g))])
        counts['exact_straight'] += int(np.max(peaks) == 0)
        counts['near_straight'] += int(np.max(peaks) <= .05)
        positive, negative = np.array([k.max(), g.max()]) > .08, np.array([k.min(), g.min()]) < -.08
        counts['one_direction'] += int(np.max(peaks) > .08 and not np.any(positive & negative))
        events = reversal_locations(path, cfg)
        counts['reversal'] += int(any(events.values()))
        counts['double_reversal'] += int(all(events.values()))
        counts['coupled_3d'] += int(np.mean((abs(k) > .08) & (abs(g) > .08)) >= .1)
        sustained = False
        for value, derivative, amplitude in ((k, q['kappa_prime'], cfg.kappa_amplitude),
                                             (g, q['gamma_prime'], cfg.gamma_amplitude)):
            mask = (abs(value) > .08) & (abs(derivative)/amplitude < 1e-6)
            edges = np.diff(np.r_[False, mask, False].astype(int))
            starts, stops = np.where(edges == 1)[0], np.where(edges == -1)[0]-1
            sustained |= any(s[b]-s[a] >= 10 for a, b in zip(starts, stops))
        counts['sustained'] += int(sustained)
    return counts


def timings(fn, repeats):
    fn()
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        fn()
        samples.append(1000*(perf_counter()-start))
    return dict(median_ms=float(np.median(samples)), min_ms=min(samples), repeats=repeats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--before', type=Path, help='Optional saved pre-change benchmark JSON')
    parser.add_argument('--num-envs', type=int, default=256)
    parser.add_argument('--repeats', type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    cfg = SamplingConfig()
    start = perf_counter()
    paths = [sample_trajectory(seed, cfg) for seed in range(args.num_envs)]
    generation_s = perf_counter()-start
    start = perf_counter()
    cache = TrajectoryCache(args.num_envs, cfg.length_m)
    cache.update_rows(torch.arange(args.num_envs), paths)
    cache_s = perf_counter()-start
    progress = torch.linspace(15., 45., args.num_envs, dtype=torch.float64)
    points = torch.tensor(np.stack([p.query(float(s))['position'] for p, s in zip(paths, progress)]))
    points += torch.tensor([0., .2, .1])
    window = torch.full((args.num_envs,), .7, dtype=torch.float64)
    offsets = torch.tensor([2., 5., 10.], dtype=torch.float64)

    def legacy():
        queries, previews = [], []
        for path, point, s in zip(paths, points.numpy(), progress.numpy()):
            projected = path.project(point, s, .7)
            queries.append(path.query(projected))
            previews.append(path.query(projected+offsets.numpy())['position'])
        reference = {key: torch.as_tensor(np.stack([q[key] for q in queries]), dtype=torch.float32)
                     for key in queries[0]}
        reference['preview'] = torch.as_tensor(np.stack(previews), dtype=torch.float32)
        return reference

    def batched():
        projected = cache.project(points, progress, window)
        result = {k: v.float() for k, v in cache.query(projected).items()}
        result['preview'] = cache.positions(projected[:, None]+offsets).float()
        return result

    report = dict(num_envs=args.num_envs, seeds=[0, args.num_envs-1], torch_threads=torch.get_num_threads(),
                  config=asdict(cfg), generation_seconds=generation_s, cache_build_seconds=cache_s,
                  sampling_attempts=dict(mean=float(np.mean([p.sampling_attempts for p in paths])),
                                         maximum=max(p.sampling_attempts for p in paths)),
                  coverage=coverage(paths, cfg), legacy_query=timings(legacy, args.repeats),
                  cached_query=timings(batched, args.repeats))
    old, new = legacy(), batched()
    report['matched_query_max_abs_error'] = {key: float((old[key]-new[key]).abs().max()) for key in old}
    # Independent off-grid analytic oracle, spanning entry, maneuver and endpoint.
    query_s = torch.linspace(.037, cfg.length_m-.073, 257).expand(args.num_envs, -1)
    exact = [path.query(query_s[0].numpy()) for path in paths]
    result = cache.query(query_s)
    report['analytic_max_abs_error'] = {
        key: float(np.max(abs(value.numpy()-np.stack([q[key] for q in exact]))))
        for key, value in result.items()}
    if args.before:
        report['pre_change'] = json.loads(args.before.read_text())
    assert all(value > 0 for value in report['coverage'].values()), report['coverage']
    assert report['analytic_max_abs_error']['position'] < 2e-5
    assert report['analytic_max_abs_error']['tangent'] < 2e-6
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
