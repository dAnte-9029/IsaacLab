"""Opt-in fixed-rollout diagnostics; never minibatch-internal PPO statistics.

Supported authority: CPU non-recurrent Unified 39D MLP with constant 4D std,
identity observation normalizers, no dropout/batchnorm. No distribution cache
mutation, action sampling, optimizer hooks, backward, or storage mutation.
"""
from contextlib import contextmanager
from time import perf_counter
import random
import numpy as np
import torch
from torch import nn

SCHEMA = 'unified_ppo_fixed_rollout_diagnostics_v1'


def analytic_kl(old_mean, old_std, new_mean, new_std):
    """KL(old || new), summed over independent action dimensions, per state."""
    return (torch.log(new_std / old_std)
            + (old_std.square() + (old_mean-new_mean).square()) / (2*new_std.square()) - .5).sum(-1)


def explained_variance(values, returns):
    target_var = returns.double().var(unbiased=False)
    if target_var <= 1e-12:
        return None
    return float(1-(returns.double()-values.double()).var(unbiased=False)/target_var)


def distribution_stats(x):
    x = x.detach().double().reshape(-1)
    return dict(mean=float(x.mean()), std=float(x.std(unbiased=False)),
                quantiles=dict(zip(['p01','p10','p50','p90','p99'],
                                   torch.quantile(x, x.new_tensor([.01,.1,.5,.9,.99])).tolist())))


def _tensors(obj):
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values(): yield from _tensors(value)
    elif isinstance(obj, (tuple, list)):
        for value in obj: yield from _tensors(value)


@contextmanager
def read_only_guard(alg):
    """Check CPU RNG and tensor mutation counters without copying Adam every call.

    Offline tests additionally compare exact values. This module never uses .data
    writes (which can bypass PyTorch mutation counters).
    """
    rng = (torch.get_rng_state().clone(), np.random.get_state(), random.getstate())
    tensors = list(alg.policy.parameters()) + list(alg.policy.buffers())
    tensors += list(_tensors(alg.optimizer.state_dict()))
    for value in vars(alg.storage).values():
        if hasattr(value, 'items') and not isinstance(value, dict):
            tensors.extend(v for _, v in value.items() if torch.is_tensor(v))
        else: tensors.extend(_tensors(value))
    # compute_returns rebinds advantages inside inference_mode in the real runner.
    # Such tensors have no version counter: compare a small exact value copy.
    versions = [None if x.is_inference() else x._version for x in tensors]
    inference_copies = {i:x.detach().clone() for i,x in enumerate(tensors) if x.is_inference()}
    modes = [m.training for m in alg.policy.modules()]
    step = alg.storage.step
    distribution = alg.policy.distribution
    groups = [{k:v for k,v in g.items() if k!='params'} for g in alg.optimizer.param_groups]
    yield
    now = np.random.get_state()
    assert torch.equal(rng[0], torch.get_rng_state()), 'diagnostics changed Torch RNG'
    assert rng[1][0] == now[0] and np.array_equal(rng[1][1], now[1]) and rng[1][2:] == now[2:]
    assert rng[2] == random.getstate(), 'diagnostics changed Python RNG'
    for i,(x,version) in enumerate(zip(tensors,versions)):
        assert (torch.equal(x,inference_copies[i]) if version is None else x._version==version), 'diagnostics mutated a training tensor'
    assert step == alg.storage.step and modes == [m.training for m in alg.policy.modules()]
    assert alg.policy.distribution is distribution
    assert groups == [{k:v for k,v in g.items() if k!='params'} for g in alg.optimizer.param_groups]


def _supported(alg):
    p = alg.policy
    if (p.is_recurrent or p.state_dependent_std or not hasattr(p, 'std')
            or p.std.shape != (4,) or next(p.parameters()).device.type != 'cpu'
            or not isinstance(p.actor_obs_normalizer, nn.Identity)
            or not isinstance(p.critic_obs_normalizer, nn.Identity)
            or any(not isinstance(m, (nn.Sequential, nn.Linear, nn.ELU)) for m in p.actor.modules())
            or any(not isinstance(m, (nn.Sequential, nn.Linear, nn.ELU)) for m in p.critic.modules())):
        raise ValueError('Diagnostics require the CPU Unified constant-std ELU MLP contract')


@torch.no_grad()
def capture_before_update(alg):
    start = perf_counter()
    _supported(alg)
    with read_only_guard(alg):
        s = alg.storage
        if s.step != s.num_transitions_per_env:
            raise ValueError('Diagnostics require a complete rollout storage')
        batch = dict(obs=s.observations.flatten(0, 1).clone())
        for key in ('actions','actions_log_prob','mu','sigma','values','returns','advantages'):
            batch[key] = getattr(s, key).flatten(0, 1).detach().clone()
        batch['pre_mean'] = alg.policy.act_inference(batch['obs']).clone()
        batch['pre_std'] = alg.policy.std.detach().clone()
        batch['pre_value'] = alg.policy.evaluate(batch['obs']).clone()
        # Fixed-policy rollout, before any PPO update. Fail if storage is stale.
        torch.testing.assert_close(batch['pre_mean'], batch['mu'], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(batch['pre_std'].expand_as(batch['sigma']), batch['sigma'])
        batch['returns_stats'] = distribution_stats(batch['returns'])
        batch['advantages_stats'] = distribution_stats(batch['advantages'])
    batch['capture_s'] = perf_counter()-start
    return batch


@torch.no_grad()
def diagnose_after_update(alg, batch, *, update, rollout_policy_version, updated_policy_version):
    start = perf_counter()
    with read_only_guard(alg):
        mean = alg.policy.act_inference(batch['obs'])
        std = alg.policy.std.detach()
        value = alg.policy.evaluate(batch['obs'])
        kl = analytic_kl(batch['pre_mean'].double(), batch['pre_std'].double(), mean.double(), std.double())
        log_prob = torch.distributions.Normal(mean, std).log_prob(batch['actions']).sum(-1)
        log_ratio = log_prob - batch['actions_log_prob'].squeeze(-1)
        # Compare in log space, avoiding exponent overflow. Raw storage actions.
        clip = ((log_ratio < np.log1p(-alg.clip_param)) | (log_ratio > np.log1p(alg.clip_param))).float()
        delta = mean-batch['pre_mean']
        result = dict(schema=SCHEMA, update=int(update), rollout_policy_version=rollout_policy_version,
                      updated_policy_version=updated_policy_version, samples=int(mean.shape[0]),
                      scope='pre_post_update_fixed_rollout_batch_not_internal_minibatches',
                      analytic_kl_old_to_new=distribution_stats(kl),
                      raw_action_ratio_clip_fraction=float(clip.mean()),
                      log_ratio=distribution_stats(log_ratio), actor_std_before=batch['pre_std'].tolist(),
                      actor_std_after=std.tolist(), actor_mean_delta_mean=delta.mean(0).tolist(),
                      actor_mean_delta_rms=delta.square().mean(0).sqrt().tolist(),
                      critic_explained_variance_before=explained_variance(batch['pre_value'],batch['returns']),
                      critic_explained_variance_after=explained_variance(value,batch['returns']),
                      returns_before=batch['returns_stats'], advantages_before=batch['advantages_stats'],
                      gradient_norm=None, gradient_norm_status='unavailable_without_optimizer_update_instrumentation')
    result['diagnostics_wall_s'] = batch['capture_s']+perf_counter()-start
    return result
