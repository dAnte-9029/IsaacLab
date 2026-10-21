"""Frozen development/holdout sets and explicit C4 plus legacy retention workflow."""

from dataclasses import replace
import json
import math
from pathlib import Path
import subprocess

import torch

from flapping_bot.path_tracking.continuous_curve import (
    sample_accepted_continuous_curve_specs, save_continuous_curves,
)


def freeze_c4_sets(output: Path, count: int = 96) -> dict:
    """Create once; fixed independent seeds, same candidate distribution, no tuning."""
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'schema': 'c4_random_eval_sets_v1', 'count': count, 'sets': {}}
    for name, seed in (('development', 130363), ('holdout', 196613)):
        generator = torch.Generator().manual_seed(seed)
        specs = sample_accepted_continuous_curve_specs(count, generator=generator,
                                                       dtype=torch.float64)
        phases = torch.rand(count, generator=generator).tolist()
        specs = tuple(replace(s, curve_id=f'{name}_{i}', initial_flap_phase_rad=phases[i]*2*math.pi)
                      for i, s in enumerate(specs))
        save_continuous_curves(output/f'{name}.json', specs, provenance={'role': name, 'seed': seed})
        manifest['sets'][name] = {'seed': seed, 'file': f'{name}.json'}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def c4_tracking_gate(summary: dict) -> bool:
    """Development tracking gate; passing is not full promotion evidence."""
    means = summary.get('case_balanced_means', {})
    upper = {'mean_horizontal_error_m': .5, 'mean_vertical_error_m': .5,
             'p95_horizontal_error_m': 1.5, 'p95_vertical_error_m': 1.5,
             'reverse_motion_fraction': .01, 'p95_roll_deg': 25.}
    return bool(summary.get('evaluation_complete') and not summary.get('smoke_only')
                and summary.get('survival_rate', 0) >= .95 and summary.get('completion_rate', 0) >= .95
                and summary.get('success_rate', 0) >= .90
                and summary.get('roll_limit_termination_count', -1) == 0
                and all(means.get(k) is not None and math.isfinite(means[k]) and means[k] <= v
                        for k,v in upper.items())
                and bool(summary.get('slices'))
                and all(s.get('success_rate', 0) >= .90 for groups in summary['slices'].values()
                        for s in groups.values()))


def evaluation_commands(root: Path, checkpoint: Path, curves: Path, output: Path) -> list[list[str]]:
    """Every suite uses fresh processes; old suites retain their old observations."""
    return [
        [str(root/'isaaclab.sh'), '-p', str(root/'scripts/flapping_px4/evaluate_continuous_curve.py'),
         '--checkpoint', str(checkpoint), '--curves', str(curves), '--preview-time-scale', '2',
         '--output', str(output/'c4')],
        [str(root/'isaaclab.sh'), '-p', str(root/'scripts/flapping_rl/evaluate_pure_rl_c3a_checkpoint.py'),
         '--checkpoint', str(checkpoint), '--spatial-stage', 'c3c', '--output-dir', str(output/'retention'),
         '--pure-rl-split-frequency-actor', '--pure-rl-heading-canonical-observation'],
    ]


def evaluate_with_retention(root: Path, checkpoint: Path, curves: Path, output: Path) -> dict:
    """Always run old suites, even after a C4 gate failure, to diagnose forgetting."""
    output.mkdir(parents=True, exist_ok=False)
    commands = evaluation_commands(root, checkpoint, curves, output)
    (output/'commands.json').write_text(json.dumps(commands, indent=2))
    for index, command in enumerate(commands):
        with (output/f'worker_{index}.log').open('w') as stream:
            # Legacy evaluator exits 2 for gate failure; JSON is authoritative.
            subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=False)
    current = json.loads((output/'c4/summary.json').read_text())
    retained = json.loads((output/'retention'/checkpoint.stem/'checkpoint_summary.json').read_text())
    result = {'schema': 'c4_training_evaluation_v1', 'checkpoint': str(checkpoint),
              'curves': str(curves), 'c4_passed': c4_tracking_gate(current),
              'retention_passed': bool(retained['all_hard_gates_passed']),
              'promotion_eligible': False,
              'note': 'Single-checkpoint evidence; adjacent passing checkpoints and independent holdout still required'}
    result['all_suites_passed'] = result['c4_passed'] and result['retention_passed']
    (output/'summary.json').write_text(json.dumps(result, indent=2))
    return result
