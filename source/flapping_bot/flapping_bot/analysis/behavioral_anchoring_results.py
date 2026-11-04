"""Offline summary of the fixed two-arm experiment; never runs or tunes a policy."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from flapping_bot.analysis.behavioral_anchoring_experiment import assert_state_equal
from flapping_bot.analysis.c4_retention_pilot import write_json
from flapping_bot.px4_like.behavioral_anchoring import TailAnchors
from flapping_bot.px4_like.pure_rl_split_actor import PureRLSplitActor


def evaluation(folder: Path, stem: str) -> dict:
    """Read unchanged suite gates and principal trajectory errors."""
    c4 = json.loads((folder / 'c4/summary.json').read_text())
    retained = json.loads((folder / 'retention' / stem / 'checkpoint_summary.json').read_text())
    return dict(c4_success_rate=c4['success_rate'], c4_survival_rate=c4['survival_rate'],
                c4_completion_rate=c4['completion_rate'], c4_means=c4['case_balanced_means'],
                legacy_gates=retained['suite_passed'],
                c3b_horizontal_error=retained['suites']['c3b']['mean_abs_horizontal_error_m'],
                c3c_success_rate=retained['suites']['c3c']['overall_success_rate'],
                c1_tail_limit_fraction=retained['suites']['c1']['tail_limit_fraction'],
                c2c_climb_success=retained['suites']['c2c']['climb_success_rate'])


def summarize(root: Path) -> dict:
    """Verify paired initial conditions, endpoint constraints and actual update counts."""
    assert json.loads((root / 'status.json').read_text())['status'] == 'completed'
    experiment = json.loads((root / 'summary.json').read_text())
    contract = experiment['contract']
    source = torch.load(contract['source'], weights_only=True, map_location='cpu')
    anchors_payload = torch.load(contract['anchor_path'], weights_only=True, map_location='cpu')
    initial = [json.loads((root / mode / 'training/initialization.json').read_text()) for mode in ('S','H')]
    assert initial[0] == initial[1]
    assert not initial[0]['legacy_memory_loss_enabled']
    report = dict(initialization_equal=True, transitions_per_arm=307200,
                  baseline=evaluation(Path(contract['baseline_evaluation']), 'model_50'), arms={})
    traces = {}
    for mode in ('S','H'):
        folder = root / mode
        rows = [json.loads(line) for line in (folder / 'training/anchoring.jsonl').read_text().splitlines()]
        assert len(rows) == 25 and sum(r['optimizer_steps'] for r in rows) == 1600
        traces[mode] = rows
        checkpoint = torch.load(folder / 'training/update_25.pt', weights_only=True, map_location='cpu')
        state = checkpoint['infos']['c4_training']
        assert state['completed_iterations'] == 76 and state['anchoring']['updates'] == 25
        assert_state_equal(state['anchoring']['payload'], anchors_payload)
        policy = checkpoint['model_state_dict']
        actor = PureRLSplitActor(555, [256,128], 'elu').eval()
        actor.load_state_dict({k.removeprefix('actor.'): v for k,v in policy.items() if k.startswith('actor.')})
        anchors = TailAnchors(state['anchoring']['payload'])
        endpoint_feasible = anchors.feasible(actor.tail_actor)
        if mode == 'H':
            assert endpoint_feasible and not any(r['constraint_violated'] for r in rows)
        parameter_delta = {}
        for block in ('actor.tail_actor.', 'actor.frequency_actor.', 'critic.', 'log_std'):
            parameter_delta[block] = float(sum((v-source['model_state_dict'][k]).square().sum()
                                              for k,v in policy.items() if k.startswith(block)).sqrt())
        names = list(source['model_state_dict'])
        assert len(names) == 19
        accepted = sum(r['accepted_tail_steps'] for r in rows)
        increment = {names[k]: int(v['step']-source['optimizer_state_dict']['state'][k]['step'])
                     for k,v in checkpoint['optimizer_state_dict']['state'].items()}
        assert all(v == (accepted if 'tail_actor' in name else 1600) for name,v in increment.items())
        events = EventAccumulator(str(folder / 'training'), size_guidance={'scalars':0})
        events.Reload()
        scalars = {}
        for tag in events.Tags()['scalars']:
            if tag.startswith(('C4Train/c4/', 'C4Reward/c4/', 'Perf/', 'Loss/')):
                values = [e.value for e in events.Scalars(tag)]
                scalars[tag] = dict(first5=float(np.mean(values[:5])), last5=float(np.mean(values[-5:])),
                                    mean=float(np.mean(values)), sum=float(np.sum(values)))
        report['arms'][mode] = dict(
            evaluations={str(i):evaluation(folder / f'evaluation_{i}', f'model_{50+i}') for i in (12,25)},
            endpoint_feasible=endpoint_feasible, final_anchor=anchors.metrics(actor),
            max_group_rms=max(v['tail_rms'] for r in rows for v in r['families'].values()),
            accepted=sum(r['accepted_tail_steps'] for r in rows),
            backtracked=sum(r['backtracked_tail_steps'] for r in rows),
            rejected=sum(r['rejected_tail_steps'] for r in rows),
            alpha_sum=sum(r['alpha_sum'] for r in rows),
            alpha_fraction=sum(r['alpha_sum'] for r in rows)/1600,
            candidate_tail_path_length=sum(s['candidate_tail_norm'] for r in rows for s in r['step_details']),
            accepted_tail_path_length=sum(s['effective_tail_norm'] for r in rows for s in r['step_details']),
            update_seconds=sum(r['update_seconds'] for r in rows),
            anchor_seconds=sum(r['anchor_seconds'] for r in rows),
            constraint_overhead_seconds=sum(r['constraint_overhead_seconds'] for r in rows),
            training_wall_seconds=experiment['arms'][mode]['training_wall_seconds'],
            parameter_delta=parameter_delta, adam_step_increments=increment,
            scalars=scalars, action_std=policy['log_std'].exp().tolist(),
            curriculum_updates=state['curriculum']['updates'],
            curriculum_probabilities=state['curriculum']['probabilities'].tolist())
    write_json(root / 'analysis.json', report)
    plot(root, report, traces)
    return report


def plot(root: Path, report: dict, traces: dict) -> None:
    """Standalone figures separate functional drift, update amount and closed-loop evidence."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2,2,figsize=(10,7),layout='constrained')
    colors = {'S':'#b44d22','H':'#176dab'}
    for mode in ('S','H'):
        rows = traces[mode]
        arm = report['arms'][mode]
        axs[0,0].plot(range(1,26),[max(v['tail_rms'] for v in r['families'].values()) for r in rows],
                      label=mode, color=colors[mode])
        axs[0,1].plot(range(1,26),[r['alpha_sum']/64 for r in rows],label=mode,color=colors[mode])
        axs[1,0].plot([0,12,25],[report['baseline']['c4_success_rate']*100]+
                      [arm['evaluations'][str(i)]['c4_success_rate']*100 for i in (12,25)],
                      marker='o',label=mode,color=colors[mode])
        axs[1,1].plot([0,12,25],[report['baseline']['c3c_success_rate']*100]+
                      [arm['evaluations'][str(i)]['c3c_success_rate']*100 for i in (12,25)],
                      marker='o',label=mode,color=colors[mode])
    axs[0,0].axhline(.02,color='black',linestyle='--',linewidth=1)
    axs[0,0].set_ylabel('Max family tail RMS (normalized action)')
    axs[0,1].set_ylabel('Mean accepted tail step fraction')
    axs[1,0].set_ylabel('C4 development success (%)')
    axs[1,1].set_ylabel('C3c frozen-grid success (%)')
    for ax in axs.flat:
        ax.set_xlabel('Additional PPO iterations from D-50')
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle('D-50 continuation: one seed, 307,200 new transitions per arm')
    fig.savefig(root / 'comparison.png',dpi=180)
    fig.savefig(root / 'comparison.svg')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    summarize(args.output.resolve())
