"""Curriculum coverage/resume and actor-only successful-memory contracts."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from flapping_bot.path_tracking.c4_curriculum import C4Curriculum, C4CurriculumConfig
from flapping_bot.px4_like.c4_retention import (
    BehaviorMemory, MEMORY_SCHEMA, TrainingEpisodeTracker, attach_behavior_memory,
)


def bank() -> dict:
    return dict(schema=MEMORY_SCHEMA, role='training_only',
                observations=torch.randn(192, 555), actions=torch.randn(192, 4),
                families=torch.arange(6).repeat_interleave(32))


def test_curriculum_waits_for_coverage_and_returns_to_regressing_region():
    config = C4CurriculumConfig(window=8, minimum_episodes=8)
    teacher = C4Curriculum(seed=1, config=config)
    initial = teacher.probabilities.clone()
    teacher.observe(0, 0.)
    assert not teacher.update()['ready']
    assert torch.equal(initial, teacher.probabilities)
    teacher = C4Curriculum(seed=1, config=config)
    for _ in range(8):
        for group in range(4):
            teacher.observe(group, 1.)
    for _ in range(20):
        teacher.update()
    baseline = teacher.probabilities.clone()
    for _ in range(8):
        for group in range(4):
            teacher.observe(group, 0. if group == 2 else 1.)
    previous = teacher.probabilities.clone()
    report = teacher.update()
    assert report['regression'][2] == 1.
    assert teacher.probabilities[2] > baseline[2]
    assert (teacher.probabilities - previous).abs().max() <= .0500000001
    assert teacher.probabilities.min() >= config.coverage_fraction / 4
    assert torch.isclose(teacher.probabilities.sum(), torch.tensor(1., dtype=torch.float64))
    previous = teacher.probabilities.clone()
    teacher.update()
    assert torch.equal(previous, teacher.probabilities)


def test_sampler_private_rng_resume_and_bin_bounds():
    teacher = C4Curriculum(seed=3)
    global_rng = torch.get_rng_state().clone()
    teacher.sample(3)
    saved = teacher.state_dict()
    specs, bins = teacher.sample(12)
    assert torch.equal(global_rng, torch.get_rng_state())
    resumed = C4Curriculum(seed=999)
    resumed.load_state_dict(saved)
    replay, replay_bins = resumed.sample(12)
    assert specs == replay and torch.equal(bins, replay_bins)
    for spec, group in zip(specs, bins):
        assert (spec.heading_rate_amplitude_rad_per_m >= .007) == bool(group // 2)
        assert (spec.slope_amplitude_rad >= torch.pi / 60) == bool(group % 2)
    assert (resumed.probabilities > 0).all()


def test_memory_is_balanced_private_and_resume_reproducible():
    payload = bank()
    memory = BehaviorMemory(payload, seed=17)
    rng = torch.get_rng_state().clone()
    state = memory.state_dict()
    observations, actions = memory.draw(60)
    assert torch.equal(rng, torch.get_rng_state())
    restored = BehaviorMemory(payload, seed=0)
    restored.load_state_dict(state)
    replay, replay_actions = restored.draw(60)
    assert torch.equal(observations, replay) and torch.equal(actions, replay_actions)
    for family in range(6):
        matches = (observations[:, None, :] == payload['observations'][family * 32:(family + 1) * 32]).all(-1).any(-1)
        assert matches.sum() == 10
    payload['observations'].zero_()
    assert memory.observations.abs().sum() > 0


def test_memory_auxiliary_uses_stored_states_and_teacher_actions_only():
    memory = BehaviorMemory(bank(), seed=4)
    obs = TensorDict({'policy': torch.zeros(12, 555), 'actor_gradient_probe_group': torch.zeros(12, 1)}, [12])
    augmented, _ = memory(obs=obs, actions=None, env=None)
    assert torch.equal(obs['policy'], augmented['policy'][:12])
    assert augmented['policy'][12:].abs().sum() > 0
    expected = memory._targets.clone()
    current = torch.ones(12, 4, requires_grad=True)
    _, targets = memory(obs=None, actions=current, env=None)
    assert torch.equal(targets[12:], expected) and not targets.requires_grad
    assert memory._targets is None
    with pytest.raises(RuntimeError):
        memory(obs=None, actions=current, env=None)
    algorithm = SimpleNamespace(symmetry=None)
    attach_behavior_memory(algorithm, memory, 1.)
    assert not algorithm.symmetry['use_data_augmentation']


def test_invalid_memory_fails_closed():
    payload = bank()
    for key, value in [('role', 'development'), ('observations', torch.zeros(192, 554)),
                       ('actions', torch.full((192, 4), float('nan'))),
                       ('families', torch.zeros(192, dtype=torch.long))]:
        with pytest.raises(ValueError):
            BehaviorMemory({**payload, key: value}, seed=1)


def test_tracker_terminal_error_and_failed_episodes_are_not_erased_by_reset():
    teacher = C4Curriculum(seed=0, config=C4CurriculumConfig(window=2, minimum_episodes=1))
    tracker = TrainingEpisodeTracker(2, 4)
    env = SimpleNamespace(
        _eval_pure_rl_cross_track_error_m=torch.tensor([.1, .2]),
        _eval_pure_rl_height_error_m=torch.zeros(2), reset_terminated=torch.zeros(2, dtype=torch.bool),
        reset_time_outs=torch.zeros(2, dtype=torch.bool),
        _pure_rl_spatial_path=SimpleNamespace(task_family_id=torch.tensor([5, 5]), hard_slice_id=torch.tensor([100, 102])),
        _eval_pure_rl_along_track_progress_m=torch.tensor([141., 141.]))
    assert not tracker.observe(env, teacher)
    env.reset_terminated[0] = True
    env._eval_pure_rl_cross_track_error_m[0] = 3.5
    row = tracker.observe(env, teacher)[0]
    assert not row['success'] and row['mean_error'][0] > 1.
    assert teacher.history[0] == [0.] and teacher.counts[2] == 0
    assert tracker.counts.tolist() == [0, 2]


def test_spatial_sampler_labels_do_not_touch_legacy_rows():
    path = Path(__file__).resolve().parents[1] / 'source/flapping_bot/flapping_bot/direct/flapping_bot/pure_rl_spatial_path.py'
    spec = importlib.util.spec_from_file_location('c4_retention_spatial_test', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    sample_spatial_path_batch = module.sample_spatial_path_batch
    teacher = C4Curriculum(seed=9)
    paths = sample_spatial_path_batch(num_paths=64, stage='c4joint', device='cpu', dtype=torch.float32,
                                      generator=torch.Generator().manual_seed(7),
                                      continuous_curve_sampler=teacher.sample)
    curves = paths.task_family_id == 5
    assert curves.any() and (~curves).any()
    assert ((paths.hard_slice_id[curves] >= 100) & (paths.hard_slice_id[curves] <= 103)).all()
    assert torch.isfinite(paths.points_world_m).all()


def test_pilot_requires_complete_adjacent_sample_equivalent_evidence():
    from flapping_bot.analysis.c4_retention_pilot import candidate_pairs
    passed = {'all_suites_passed': True}
    assert candidate_pairs({'50': passed, '100': passed}) == []
    assert candidate_pairs({'50': passed, '75': passed}) == [[50, 75]]
    assert candidate_pairs({'50': passed, '75': passed}, transitions_per_iteration=96 * 16) == []
    assert candidate_pairs({'50': passed, '75': {'all_suites_passed': False}}) == []
