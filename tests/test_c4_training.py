"""Training configuration and fresh-process evaluation orchestration contracts."""
from pathlib import Path
import json
import pytest

from flapping_bot.analysis.c4_training import prepare_agent_config
from flapping_bot.analysis.c4_training_evaluation import (
    c4_tracking_gate, evaluation_commands, evaluate_with_retention, freeze_c4_sets,
)


def test_agent_configuration_preserves_source_and_excludes_group_from_actor():
    source = {'policy': {'class_name': 'PureRLSplitActorCritic'}, 'algorithm': {}}
    config = prepare_agent_config(source, seed=3, steps=48, minibatches=16, save_interval=25)
    assert source == {'policy': {'class_name': 'PureRLSplitActorCritic'}, 'algorithm': {}}
    assert config['obs_groups'] == {'policy': ['policy'], 'critic': ['policy']}
    assert not config['algorithm']['normalize_advantage_per_mini_batch']
    with pytest.raises(ValueError):
        prepare_agent_config({**source, 'clip_actions': 1}, seed=0, steps=48, minibatches=16, save_interval=25)


def test_sets_are_frozen_reproducible_and_disjoint(tmp_path):
    from flapping_bot.path_tracking.continuous_curve import load_continuous_curves
    for name in ('a', 'b'):
        freeze_c4_sets(tmp_path/name, 4)
    dev = load_continuous_curves(tmp_path/'a/development.json')
    hold = load_continuous_curves(tmp_path/'a/holdout.json')
    assert dev == load_continuous_curves(tmp_path/'b/development.json')
    assert all(a.heading_phase_rad != b.heading_phase_rad for a,b in zip(dev,hold))
    assert len({s.initial_flap_phase_rad for s in dev}) == 4
    with pytest.raises(FileExistsError):
        freeze_c4_sets(tmp_path/'a', 4)


def test_missing_gate_evidence_fails_closed():
    assert not c4_tracking_gate({})
    assert not c4_tracking_gate({'evaluation_complete': True, 'smoke_only': True})


def test_retention_runs_after_c4_failure_and_missing_outputs_raise(tmp_path, monkeypatch):
    checkpoint = tmp_path/'model_25.pt'
    output = tmp_path/'eval'
    commands = evaluation_commands(tmp_path, checkpoint, tmp_path/'curves.json', output)
    assert '--preview-time-scale' in commands[0]
    assert '--spatial-stage' in commands[1] and 'c3c' in commands[1]
    assert '--pure-rl-heading-canonical-observation' in commands[1]
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            (output/'c4').mkdir()
            (output/'c4/summary.json').write_text(json.dumps({'evaluation_complete': True, 'success_rate': 0}))
        else:
            folder = output/'retention/model_25'
            folder.mkdir(parents=True)
            (folder/'checkpoint_summary.json').write_text(json.dumps({'all_hard_gates_passed': True}))
    monkeypatch.setattr('flapping_bot.analysis.c4_training_evaluation.subprocess.run', fake_run)
    result = evaluate_with_retention(tmp_path, checkpoint, tmp_path/'curves.json', output)
    assert len(calls) == 2 and result['retention_passed'] and not result['all_suites_passed']
    assert not result['promotion_eligible']
