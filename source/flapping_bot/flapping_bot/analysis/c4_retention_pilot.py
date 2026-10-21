"""Bounded sequential A--D runner with durable, fail-closed scientific results."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


def write_json(path: Path, value: dict) -> None:
    """Atomically publish status without exposing a half-written result."""
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def candidate_pairs(evaluations: dict[str, dict], *, transitions_per_iteration: int = 256 * 48) -> list[list[int]]:
    """Adjacent all-suite evidence at the frozen minimum and sample cadence."""
    pairs = []
    for left, right in ((50, 75), (75, 100)):
        if ((left + 1) * transitions_per_iteration < 614400
                or (right - left) * transitions_per_iteration < 307200):
            continue
        if all(evaluations.get(str(index), {}).get('all_suites_passed') is True for index in (left, right)):
            pairs.append([left, right])
    return pairs


def source_manifest(root: Path) -> dict[str, str]:
    """Hash project and launcher sources including untracked implementation files."""
    files = []
    for folder in ('source/flapping_bot', 'scripts/flapping_px4', 'scripts/flapping_rl'):
        files.extend(p for p in (root / folder).rglob('*') if p.suffix in ('.py', '.cpp', '.toml'))
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files)}


def run_pilot(root: Path, output: Path, weights: Path, memory: Path, development: Path) -> dict:
    """Run 101 updates/arm and all six suites at 50/75/100; never promote or open holdout."""
    from flapping_bot.analysis.c4_training_evaluation import evaluate_with_retention

    for path in (weights, memory, development):
        if not path.is_file():
            raise FileNotFoundError(path)
    # Fail closed on a wrong split even if the user passes an arbitrary filename.
    curves = json.loads(development.read_text())
    if curves.get('provenance', {}).get('role') != 'development':
        raise ValueError('Pilot requires an explicitly marked development recipe set')
    output.mkdir(parents=True, exist_ok=False)
    frozen_sources = source_manifest(root)
    write_json(output / 'source_manifest.json', frozen_sources)
    for relative_path in frozen_sources:
        destination = output / 'source_snapshot' / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative_path, destination)
    contract = dict(schema='c4_retention_pilot_v1', variants=['A', 'B', 'C', 'D'], seed=0,
                    iterations=101, envs=256, steps=48, minibatches=16, memory_coefficient=1.0,
                    evaluation_indices=[50, 75, 100], weights=str(weights), memory=str(memory),
                    development=str(development),
                    input_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (weights, memory, development)},
                    git_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                    timestamp=datetime.now(timezone.utc).isoformat())
    write_json(output / 'contract.json', contract)
    (output / 'working_tree.patch').write_bytes(subprocess.check_output(['git', 'diff', '--binary'], cwd=root))
    report = dict(schema='c4_retention_pilot_results_v1', variants={}, promotion_eligible=False,
                  holdout_opened=False, expanded_envelope=False)
    started = time.monotonic()
    def check_sources():
        if source_manifest(root) != frozen_sources:
            raise RuntimeError('Project sources changed during the experiment; refusing mixed-code evidence')
    try:
        for variant in contract['variants']:
            check_sources()
            training = output / variant / 'training'
            training.parent.mkdir()
            write_json(output / 'status.json', dict(status='running', variant=variant, phase='training'))
            command = [sys.executable, str(root / 'scripts/flapping_px4/train_continuous_curve.py'),
                       '--weights', str(weights), '--retention-variant', variant, '--iterations', '101',
                       '--num-envs', '256', '--steps', '48', '--minibatches', '16', '--seed', '0',
                       '--save-interval', '25', '--output', str(training)]
            if variant in ('C', 'D'):
                command += ['--memory', str(memory), '--memory-coefficient', '1.0']
            write_json(output / variant / 'command.json', dict(command=command))
            with (output / variant / 'training.log').open('w') as stream:
                completed = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=False)
            status = json.loads((training / 'status.json').read_text())
            summary = json.loads((training / 'summary.json').read_text())
            if completed.returncode or status.get('status') != 'completed' or summary.get('completed_iterations') != 101:
                raise RuntimeError(f'Incomplete training for {variant}')
            arm = dict(training=summary, evaluations={}, development_candidate_pairs=[])
            report['variants'][variant] = arm
            write_json(output / 'summary.json', report)
            for index in contract['evaluation_indices']:
                check_sources()
                write_json(output / 'status.json', dict(status='running', variant=variant, phase='evaluation', checkpoint=index))
                evaluation = evaluate_with_retention(root, training / f'model_{index}.pt', development,
                                                     output / variant / f'evaluation_{index}')
                arm['evaluations'][str(index)] = evaluation
                arm['development_candidate_pairs'] = candidate_pairs(arm['evaluations'])
                write_json(output / 'summary.json', report)
        report['elapsed_seconds'] = time.monotonic() - started
        report['training_transitions'] = 4 * 101 * 256 * 48
        report['next_gate'] = ('review_development_candidates_before_multiseed_and_expansion'
                              if any(v['development_candidate_pairs'] for v in report['variants'].values())
                              else 'diagnose_failed_retention_before_any_expansion')
        write_json(output / 'summary.json', report)
        write_json(output / 'status.json', dict(status='completed', scientific_gate=report['next_gate']))
    except BaseException:
        write_json(output / 'status.json', dict(status='failed', error=traceback.format_exc()))
        raise
    return report
