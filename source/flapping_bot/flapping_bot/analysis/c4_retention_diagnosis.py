"""Read-only checkpoint/trace analysis; no policy updates or holdout access.

Run with ``python -m flapping_bot.analysis.c4_retention_diagnosis --help``.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from flapping_bot.px4_like.pure_rl_split_actor import PureRLSplitActor


def diagnose(pilot: Path, memory_path: Path, output: Path) -> dict:
    """Compare frozen actors on identical memory and development observations."""
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    inputs = {}

    def record(path: Path) -> None:
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    record(Path(__file__).resolve())

    def load_actor(path: Path):
        record(path)
        state = torch.load(path, map_location='cpu', weights_only=False)['model_state_dict']
        actor = PureRLSplitActor(555, [256, 128], 'elu').eval()
        actor.load_state_dict({k[6:]: v for k, v in state.items() if k.startswith('actor.')}, strict=True)
        return actor, state['log_std'].exp()

    record(memory_path)
    memory = torch.load(memory_path, map_location='cpu', weights_only=True)
    assert memory['role'] == 'training_only'
    report = dict(schema='c4_retention_diagnosis_v1', arms={}, fixed_states={}, traces={},
                  holdout_opened=False, policy_updated=False,
                  channel_order=['frequency', 'rudder', 'left_elevon', 'right_elevon'])
    for arm in 'ABCD':
        training = pilot / arm / 'training'
        path = training / 'update_diagnostics.jsonl'
        record(path)
        diagnostics = [json.loads(line) for line in path.read_text().splitlines()]
        events = EventAccumulator(str(training)); events.Reload()
        for event_file in training.glob('events.out.tfevents.*'):
            record(event_file)
        arm_report = dict(windows={}, memory={})
        report['arms'][arm] = arm_report
        for start, stop in ((25, 50), (51, 75), (76, 100)):
            rows = [r for r in diagnostics if start <= r['iteration'] <= stop]
            keys = sorted(set().union(*(r.keys() for r in rows)))
            means = {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys if k != 'iteration'}
            conflict = {k: float(np.mean([r[k] < 0 for r in rows if k in r]))
                        for k in keys if k.startswith(('cos_', 'tail_cos_'))}
            scalar = {}
            for tag in events.Tags()['scalars']:
                if tag.startswith(('Loss/', 'C4Reward/c4/', 'C4Train/c4/', 'Policy/')):
                    values = [e.value for e in events.Scalars(tag) if start <= e.step <= stop]
                    if values:
                        scalar[tag] = float(np.mean(values))
            arm_report['windows'][f'{start}-{stop}'] = dict(probe_count=len(rows), means=means,
                                                         negative_cosine_fraction=conflict, scalars=scalar)
        for index in (50, 75, 100):
            actor, std = load_actor(training / f'model_{index}.pt')
            with torch.no_grad():
                prediction = actor(memory['observations'])
            arm_report['memory'][str(index)] = dict(std=std.tolist(), families={
                str(f): (prediction[memory['families'] == f] - memory['actions'][memory['families'] == f])
                .square().mean(0).sqrt().tolist() for f in range(6)})

    actors = {i: load_actor(pilot / 'D/training' / f'model_{i}.pt')[0] for i in (50, 75)}
    for index in (50, 75):
        errors, deltas, replay_errors = [], [], []
        for path in sorted((pilot / 'D' / f'evaluation_{index}/c4/traces').glob('*.npz')):
            record(path)
            with np.load(path) as trace:
                observations = torch.from_numpy(trace['policy_input']).float()
                with torch.no_grad():
                    a, b = actors[50](observations), actors[75](observations)
                recorded = torch.from_numpy(trace['raw_action']).float()
                own = a if index == 50 else b
                replay_errors.append(float((own - recorded).abs().max()))
                deltas.append((b - a).numpy())
                errors.append({k: float(trace[k].mean()) for k in (
                    'cross_track_error_m', 'height_error_m', 'along_track_velocity_mps',
                    'actual_flap_frequency_hz', 'actual_rudder_rad',
                    'actual_left_elevon_rad', 'actual_right_elevon_rad')})
        assert len(errors) == 96, 'Incomplete development traces'
        assert max(replay_errors) < 1e-5, 'Actor loading or observation/action alignment failed'
        delta = np.concatenate(deltas)
        report['fixed_states'][str(index)] = dict(cases=len(errors), transitions=len(delta),
            max_recorded_action_error=max(replay_errors), mean_delta=delta.mean(0).tolist(),
            rms_delta=np.sqrt(np.mean(delta ** 2, axis=0)).tolist(),
            p95_absolute_delta=np.quantile(np.abs(delta), .95, axis=0).tolist())
        report['traces'][str(index)] = {k: float(np.mean([e[k] for e in errors])) for k in errors[0]}
    report['input_sha256'] = inputs
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--memory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.pilot.resolve(), args.memory.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
