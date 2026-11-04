"""Functional bounds, real Adam state, PPO parity and fixed-reference resume."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO

from flapping_bot.analysis.behavioral_anchoring_experiment import assert_state_equal
from flapping_bot.px4_like.behavioral_anchoring import (
    ANCHOR_SCHEMA, AnchoredPPO, ConstrainedTailStep, TailAnchors,
)
from flapping_bot.px4_like.c4_retention import MEMORY_SCHEMA
from flapping_bot.px4_like.pure_rl_split_actor import PureRLSplitActorCritic

torch.set_num_threads(1)


def make_payload(tail, epsilon=0.0004):
    obs = torch.zeros(192, 555)
    obs[:, 0] = 1.
    with torch.no_grad():
        actions = torch.cat((torch.zeros(192, 1), tail(obs)), -1)
    return dict(schema=MEMORY_SCHEMA, anchor_schema=ANCHOR_SCHEMA, role='training_only',
                observations=obs, actions=actions, families=torch.arange(6).repeat_interleave(32),
                epsilon=[epsilon]*6, reference_actor=copy.deepcopy(tail.state_dict()))


@pytest.mark.parametrize('epsilon,max_backtracks,kind', [(10.,10,'accept'), (.0004,10,'backtrack'), (1e-16,0,'reject')])
def test_real_adam_accept_backtrack_reject(epsilon, max_backtracks, kind):
    tail = torch.nn.Linear(555, 3)
    extra = torch.nn.Parameter(torch.tensor(1.))  # stands for independent critic/std/frequency
    opt = torch.optim.Adam([*tail.parameters(), extra], lr=.1)
    anchors = TailAnchors(make_payload(tail, epsilon))
    before = copy.deepcopy(tail.state_dict())
    # Prime real momentum without moving the anchor reference.
    for p in tail.parameters():
        p.grad = torch.ones_like(p)
    extra.grad = torch.ones_like(extra)
    opt.step()
    tail.load_state_dict(before)
    before_opt = copy.deepcopy(opt.state_dict())
    clone = copy.deepcopy(tail)
    clone_opt = torch.optim.Adam([*clone.parameters(), torch.nn.Parameter(extra.detach().clone())], lr=.1)
    clone_opt.load_state_dict(before_opt)
    for p in tail.parameters():
        p.grad = torch.ones_like(p)
    result = ConstrainedTailStep(opt, tail, anchors, max_backtracks=max_backtracks).step()
    alpha = result['alpha']
    assert anchors.feasible(tail)
    assert (alpha == 1 if kind == 'accept' else 0 < alpha < 1 if kind == 'backtrack' else alpha == 0)
    # A partial step is numerically a real Adam step at reduced LR, with identical moments.
    if alpha:
        clone_opt.param_groups[0]['lr'] *= alpha
        for p in clone.parameters():
            p.grad = torch.ones_like(p)
        clone_opt.step()
        for p, q in zip(tail.parameters(), clone.parameters()):
            torch.testing.assert_close(p, q, atol=2e-8, rtol=1e-6)
            assert_state_equal(opt.state[p], clone_opt.state[q])
    else:
        assert_state_equal(tail.state_dict(), before)
        for i, p in enumerate(tail.parameters()):
            assert_state_equal(opt.state[p], before_opt['state'][i])
    assert opt.state[extra]['step'] == 2  # independent parameters always progress


def test_groups_are_not_hidden_and_soft_loss_is_tail_only():
    tail = torch.nn.Linear(555, 3, bias=False)
    payload = make_payload(tail)
    payload['actions'][:32, 1:] += .04
    anchors = TailAnchors(payload)
    assert anchors.distances(tail).mean() < .0004
    assert not anchors.feasible(tail)
    frequency = torch.nn.Linear(555, 1)
    anchors.distances(tail).mean().backward()
    assert any(p.grad is not None for p in tail.parameters())
    assert all(p.grad is None for p in frequency.parameters())
    payload['role'] = 'development'
    with pytest.raises(ValueError):
        TailAnchors(payload)


def test_unconstrained_update_matches_installed_ppo():
    torch.manual_seed(7)
    obs = TensorDict({'policy': torch.randn(12,555)}, [12])
    policy = PureRLSplitActorCritic(obs, {'policy':['policy'], 'critic':['policy']}, 4,
                                   actor_hidden_dims=[8], critic_hidden_dims=[8], noise_std_type='log')
    base = PPO(copy.deepcopy(policy), num_learning_epochs=1, num_mini_batches=1, schedule='fixed')
    hard = AnchoredPPO(copy.deepcopy(policy), num_learning_epochs=1, num_mini_batches=1, schedule='fixed')
    hard.configure_anchors(make_payload(policy.actor.tail_actor, epsilon=100.), 'H')
    with torch.no_grad():
        actions = policy.act(obs)
        batch = (obs, actions, policy.evaluate(obs), torch.randn(12,1), torch.randn(12,1),
                 policy.get_actions_log_prob(actions).unsqueeze(-1), policy.action_mean, policy.action_std,
                 (None,None), None)
    for algorithm in (base,hard):
        algorithm.storage = SimpleNamespace(observations=obs,
            mini_batch_generator=lambda *args: iter([batch]), clear=lambda: None)
    rng = torch.get_rng_state()
    expected = base.update()
    torch.set_rng_state(rng)
    actual = hard.update()
    assert_state_equal(base.policy.state_dict(), hard.policy.state_dict())
    assert_state_equal(base.optimizer.state_dict(), hard.optimizer.state_dict())
    for key in expected:
        assert expected[key] == pytest.approx(actual[key])


def test_checkpoint_resume_preserves_reference_and_next_step(tmp_path):
    tail = torch.nn.Linear(555,3)
    payload = make_payload(tail)
    anchors = TailAnchors(payload)
    opt = torch.optim.Adam(tail.parameters(), lr=.1)
    for p in tail.parameters():
        p.grad = torch.ones_like(p)
    ConstrainedTailStep(opt, tail, anchors).step()
    file = tmp_path/'resume.pt'
    torch.save(dict(model=tail.state_dict(), optimizer=opt.state_dict(),
                    anchoring=dict(payload=anchors.payload, mode='H', updates=1)), file)
    saved = torch.load(file, weights_only=True)
    clone = copy.deepcopy(tail)
    new_opt = torch.optim.Adam(clone.parameters(), lr=.1)
    new_opt.load_state_dict(saved['optimizer'])
    restored = TailAnchors(saved['anchoring']['payload'])
    assert_state_equal(anchors.payload, restored.payload)
    for module, optimizer, bank in ((tail,opt,anchors),(clone,new_opt,restored)):
        for p in module.parameters():
            p.grad = torch.ones_like(p)
        ConstrainedTailStep(optimizer,module,bank).step()
    assert_state_equal(tail.state_dict(), clone.state_dict())
    assert_state_equal(opt.state_dict(), new_opt.state_dict())


def test_evaluation_restart_reuses_completed_c4_without_extra_training(tmp_path, monkeypatch):
    from flapping_bot.analysis.behavioral_anchoring_experiment import evaluate_saved_checkpoint
    output = tmp_path / 'evaluation'
    (output / 'c4').mkdir(parents=True)
    (output / 'c4/summary.json').write_text(json.dumps({'evaluation_complete': True}))
    checkpoint = tmp_path / 'model_62.pt'
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        target = output / 'retention/model_62'
        target.mkdir(parents=True)
        (target / 'checkpoint_summary.json').write_text(json.dumps({'all_hard_gates_passed': False}))
        return SimpleNamespace(returncode=2)
    monkeypatch.setattr('flapping_bot.analysis.behavioral_anchoring_experiment.subprocess.run', fake_run)
    first = evaluate_saved_checkpoint(tmp_path, checkpoint, tmp_path / 'development.json', output)
    second = evaluate_saved_checkpoint(tmp_path, checkpoint, tmp_path / 'development.json', output)
    assert first == second and len(calls) == 1
    assert 'evaluate_pure_rl_c3a_checkpoint.py' in calls[0][2]
