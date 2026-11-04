"""Prepare and execute the single seed, two-arm D-50 continuation experiment."""

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import time

import torch
import yaml

from flapping_bot.analysis.c4_retention_pilot import write_json
from flapping_bot.px4_like.behavioral_anchoring import ANCHOR_SCHEMA, EPSILON, TailAnchors
from flapping_bot.px4_like.c4_retention import BehaviorMemory
from flapping_bot.px4_like.pure_rl_split_actor import PureRLSplitActor


def assert_state_equal(left: object, right: object) -> None:
    """Check complete nested optimizer/curriculum state, including Adam counters."""
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_state_equal(a, b)
    else:
        assert left == right


def prepare(root: Path, output: Path) -> dict:
    """Freeze reference using only the preexisting training-only memory inputs."""
    pilot = root / 'logs/continual_c4/20260922_pilot'
    source = pilot / 'D/training/model_50.pt'
    memory_path = root / 'logs/continual_c4/20260922_memory/memory.pt'
    checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    state = checkpoint['infos']['c4_training']
    assert state['completed_iterations'] == 51 and state['learning_rate'] == 5e-5
    assert len(checkpoint['optimizer_state_dict']['state']) == 19
    cfg = yaml.safe_load((source.parent / 'params/agent.yaml').read_text())
    assert cfg['policy']['class_name'] == 'PureRLSplitActorCritic'
    assert not cfg['empirical_normalization'] and not cfg['policy']['actor_obs_normalization']
    actor = PureRLSplitActor(555, cfg['policy']['actor_hidden_dims'], cfg['policy']['activation']).eval()
    actor.load_state_dict({k.removeprefix('actor.'): v for k, v in checkpoint['model_state_dict'].items()
                           if k.startswith('actor.')}, strict=True)
    actor.requires_grad_(False)
    payload = torch.load(memory_path, map_location='cpu', weights_only=True)
    BehaviorMemory(payload, seed=0)
    payload = copy.deepcopy(payload)
    with torch.no_grad():
        payload['actions'] = actor(payload['observations']).clone()
    payload.update(anchor_schema=ANCHOR_SCHEMA, source_checkpoint=str(source),
                   reference_actor=copy.deepcopy(actor.state_dict()), epsilon=[EPSILON] * 6,
                   reference_note='D-50 deterministic unclipped means; legacy targets replaced')
    anchors = TailAnchors(payload)
    assert torch.equal(anchors.distances(actor.tail_actor), torch.zeros(6))
    output.mkdir(parents=True, exist_ok=False)
    torch.save(payload, output / 'anchors.pt')
    contract = dict(source=str(source), legacy_memory=str(memory_path), anchor_path=str(output / 'anchors.pt'),
                    development=str(root / 'logs/continuous_curve/20260910_c4_frozen_sets/development.json'),
                    baseline_evaluation=str(pilot / 'D/evaluation_50'), modes=['S', 'H'], seed=0,
                    iterations_per_arm=25, transitions_per_arm=307200, num_envs=256, steps=48,
                    minibatches=16, epochs=4, learning_rate=5e-5, schedule='fixed',
                    lambda_soft=1., family_weights=[1/6]*6, epsilon=[EPSILON]*6, max_backtracks=10,
                    anchor_samples=[len(i) for i in anchors.indices], evaluation_updates=[12,25],
                    optimizer_restore='full D-50 Adam, guard schedule, curriculum and torch RNG',
                    simulation_restore='fresh episodes; no bitwise trajectory continuation',
                    legacy_teacher_loss=False, holdout_opened=False, promotion_eligible=False)
    write_json(output / 'contract.json', contract)
    return contract


def run(root: Path, output: Path) -> None:
    """Two fresh workers, equal new-transition budgets, frozen six-suite evaluation."""
    contract = json.loads((output / 'contract.json').read_text())
    summary_path = output / 'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else dict(arms={}, contract=contract)
    started = time.perf_counter()
    try:
        for mode in contract['modes']:
            folder = output / mode
            folder.mkdir(exist_ok=True)
            training = folder / 'training'
            command = [str(root / 'isaaclab.sh'), '-p', str(root / 'scripts/flapping_px4/train_continuous_curve.py'),
                       '--resume', contract['source'], '--output', str(training), '--retention-variant', 'D',
                       '--memory', contract['legacy_memory'], '--anchoring-mode', mode, '--anchors', contract['anchor_path'],
                       '--iterations', '25', '--seed', '0', '--num-envs', '256', '--steps', '48',
                       '--minibatches', '16', '--save-interval', '100000']
            if not training.exists():
                write_json(folder / 'command.json', dict(command=command))
                write_json(output / 'status.json', dict(status='running', mode=mode, phase='training'))
                tick = time.perf_counter()
                with (folder / 'training.log').open('x') as log:
                    subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
                arm = dict(training=json.loads((training / 'summary.json').read_text()), evaluations={},
                           training_wall_seconds=time.perf_counter() - tick)
            else:
                if json.loads((training / 'status.json').read_text())['status'] != 'completed':
                    raise RuntimeError('Existing incomplete training requires explicit checkpoint recovery; never restart it')
                arm = summary['arms'][mode]
            assert arm['training']['completed_iterations'] - arm['training']['start_iteration'] == 25
            rows = [json.loads(line) for line in (training / 'anchoring.jsonl').read_text().splitlines()]
            assert len(rows) == 25 and [r['update'] for r in rows] == list(range(1, 26))
            if mode == 'H':
                assert not any(r['constraint_violated'] for r in rows)
            arm['anchoring'] = dict(accepted=sum(r['accepted_tail_steps'] for r in rows),
                                    backtracked=sum(r['backtracked_tail_steps'] for r in rows),
                                    rejected=sum(r['rejected_tail_steps'] for r in rows),
                                    alpha_sum=sum(r['alpha_sum'] for r in rows),
                                    final_families=rows[-1]['families'])
            summary['arms'][mode] = arm
            write_json(output / 'summary.json', summary)
            for update in contract['evaluation_updates']:
                write_json(output / 'status.json', dict(status='running', mode=mode, phase='evaluation', update=update))
                # Frozen legacy evaluator requires model_<absolute zero-based iteration>.pt.
                checkpoint = training / f'model_{50 + update}.pt'
                if not checkpoint.exists():
                    os.link(training / f'update_{update}.pt', checkpoint)
                arm['evaluations'][str(update)] = evaluate_saved_checkpoint(
                    root, checkpoint, Path(contract['development']), folder / f'evaluation_{update}')
                write_json(output / 'summary.json', summary)
        summary['wall_seconds'] = time.perf_counter() - started
        write_json(output / 'summary.json', summary)
        write_json(output / 'status.json', dict(status='completed'))
    except BaseException as error:
        write_json(output / 'status.json', dict(status='failed', error=repr(error)))
        raise


def evaluate_saved_checkpoint(root: Path, checkpoint: Path, curves: Path, output: Path) -> dict:
    """Reuse completed frozen suites after an orchestration interruption, never train."""
    from flapping_bot.analysis.c4_training_evaluation import evaluation_commands, c4_tracking_gate
    output.mkdir(parents=True, exist_ok=True)
    commands = evaluation_commands(root, checkpoint, curves, output)
    write_json(output / 'canonical_commands.json', dict(commands=commands))
    result_paths = [output / 'c4/summary.json',
                    output / 'retention' / checkpoint.stem / 'checkpoint_summary.json']
    for index, (command, result_path) in enumerate(zip(commands, result_paths)):
        if result_path.exists():
            continue
        with (output / f'canonical_worker_{index}.log').open('x') as log:
            completed = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=False)
        if completed.returncode not in (0, 2) or not result_path.exists():
            raise RuntimeError(f'Frozen evaluation failed: {command}')
    current, retained = [json.loads(p.read_text()) for p in result_paths]
    result = dict(checkpoint=str(checkpoint), curves=str(curves), c4_passed=c4_tracking_gate(current),
                  retention_passed=bool(retained['all_hard_gates_passed']), promotion_eligible=False)
    result['all_suites_passed'] = result['c4_passed'] and result['retention_passed']
    write_json(output / 'summary.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('prepare', 'run'))
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.operation == 'prepare':
        prepare(args.root.resolve(), args.output.resolve())
    else:
        run(args.root.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
