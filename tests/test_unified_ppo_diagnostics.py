"""CPU synthetic storage only: no simulator, environment rollout, or formal training."""
from copy import deepcopy
import random
import numpy as np
import pytest
import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from flapping_bot.analysis.unified_ppo_contract import POLICY, PPO_CONFIG
from flapping_bot.analysis.unified_ppo_diagnostics import (
    analytic_kl, explained_variance, capture_before_update, diagnose_after_update)

def equal(a,b):
    if torch.is_tensor(a): assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b

def fixture(n=8,t=8):
    torch.set_num_threads(1);torch.manual_seed(19)
    obs=TensorDict({'policy':torch.randn(n,39)},batch_size=[n])
    policy=ActorCritic(obs,{'policy':['policy'],'critic':['policy']},4,**POLICY)
    alg=PPO(policy,device='cpu',**PPO_CONFIG);alg.init_storage('rl',n,t,obs,[4]);s=alg.storage
    with torch.no_grad():
        policy.std.copy_(torch.tensor([.15,.2,.25,.3]))
        s.observations['policy'].copy_(torch.randn(t,n,39))
        flat=s.observations.flatten(0,1);mu=policy.act_inference(flat).reshape(t,n,4)
        s.mu.copy_(mu);s.sigma.copy_(policy.std.expand_as(mu))
        s.actions.copy_(mu+torch.randn_like(mu)*s.sigma)
        s.actions[0,0,:]=torch.tensor([2.,-2.,1.5,-1.5])
        s.actions_log_prob.copy_(torch.distributions.Normal(s.mu,s.sigma).log_prob(s.actions).sum(-1,keepdim=True))
        s.values.copy_(policy.evaluate(flat).reshape(t,n,1))
        s.returns.copy_(s.values+torch.randn_like(s.values));s.advantages.copy_((s.returns-s.values))
        s.advantages.sub_(s.advantages.mean()).div_(s.advantages.std()+1e-8)
    s.step=t
    with torch.inference_mode():
        s.rewards.copy_(torch.randn_like(s.rewards))
        alg.compute_returns(obs)
    assert s.advantages.is_inference()
    return alg

def snapshot(alg):
    s=alg.storage
    return deepcopy(dict(policy=alg.policy.state_dict(),optimizer=alg.optimizer.state_dict(),
        storage={k:v.clone() for k,v in vars(s).items() if torch.is_tensor(v)},obs=s.observations['policy'].clone(),step=s.step,
        gradients=[p.grad.clone() if p.grad is not None else None for p in alg.policy.parameters()]))

def test_formula_and_explained_variance():
    m=torch.tensor([[1.,2.,3.,4.],[.1,.2,.3,.4]],dtype=torch.float64)
    s=torch.tensor([.1,.2,.3,.4],dtype=torch.float64);n=m+.13;t=s*1.2
    actual=analytic_kl(m,s,n,t)
    independent=torch.distributions.kl_divergence(torch.distributions.Normal(m,s),torch.distributions.Normal(n,t)).sum(-1)
    torch.testing.assert_close(actual,independent,atol=1e-14,rtol=1e-14)
    torch.testing.assert_close(analytic_kl(m,s,m,s),torch.zeros(2,dtype=torch.float64),atol=1e-14,rtol=0)
    assert explained_variance(torch.ones(8),torch.ones(8)) is None
    assert explained_variance(torch.zeros(8),torch.ones(8)+torch.arange(8)*1e-8) is None
    x=torch.arange(8.)
    assert explained_variance(x,x)==1.
    assert explained_variance(torch.zeros(8),x)==0.

def test_read_only_rng_and_same_policy():
    a=fixture();before=snapshot(a);rng=torch.get_rng_state().clone();npstate=np.random.get_state();pystate=random.getstate()
    b=capture_before_update(a);d=diagnose_after_update(a,b,update=1,rollout_policy_version='v0',updated_policy_version='v0')
    equal(before,snapshot(a));assert torch.equal(rng,torch.get_rng_state());assert np.array_equal(npstate[1],np.random.get_state()[1]);assert pystate==random.getstate()
    assert d['analytic_kl_old_to_new']['mean']==pytest.approx(0,abs=1e-12)
    assert d['raw_action_ratio_clip_fraction']==0 and d['samples']==64
    assert d['actor_std_before']==pytest.approx([.15,.2,.25,.3])
    assert d['gradient_norm'] is None and 'minibatches' in d['scope']

def test_raw_actions_not_clipped():
    a=fixture();b=capture_before_update(a)
    with torch.no_grad():a.policy.actor[-1].bias.add_(.15)
    d=diagnose_after_update(a,b,update=1,rollout_policy_version='v0',updated_policy_version='v1')
    with torch.no_grad():
        mean=a.policy.act_inference(b['obs']);dist=torch.distributions.Normal(mean,a.policy.std)
        ratio=(dist.log_prob(b['actions']).sum(-1)-b['actions_log_prob'].squeeze(-1)).exp()
        expected=((ratio<.8)|(ratio>1.2)).float().mean()
        clipped=(dist.log_prob(b['actions'].clamp(-1,1)).sum(-1)-b['actions_log_prob'].squeeze(-1)).exp()
    assert d['raw_action_ratio_clip_fraction']==float(expected)
    assert not torch.equal(ratio,clipped)
    assert torch.equal(b['actions'],a.storage.actions.flatten(0,1))

def test_identical_real_ppo_updates_with_and_without_diagnostics():
    """Same installed PPO.update on synthetic samples, exact Adam/RNG/loss parity."""
    a=fixture();b=fixture();state=torch.get_rng_state().clone()
    loss_a=a.update();after_rng=torch.get_rng_state().clone();after=snapshot(a)
    torch.set_rng_state(state)
    captured=capture_before_update(b);loss_b=b.update()
    result=diagnose_after_update(b,captured,update=1,rollout_policy_version='v0',updated_policy_version='v1')
    assert loss_a==loss_b;equal(after,snapshot(b));assert torch.equal(after_rng,torch.get_rng_state())
    assert result['analytic_kl_old_to_new']['mean']>=0 and result['samples']==64

def test_partial_storage_and_unsupported_normalizer_fail_closed():
    a=fixture();a.storage.step=2
    with pytest.raises(ValueError,match='complete'):capture_before_update(a)
    a=fixture();a.policy.actor_obs_normalizer=torch.nn.BatchNorm1d(39)
    with pytest.raises(ValueError,match='contract'):capture_before_update(a)

def test_default_off_runner_path_is_original_update():
    """Static wiring gate complements dynamic exact PPO parity; no Sim startup."""
    import ast
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    source=(root/'scripts/flapping_px4/train_unified_trajectory.py').read_text()
    assert "'--ppo-diagnostics',action='store_true'" in source
    tree=ast.parse((root/'source/flapping_bot/flapping_bot/analysis/unified_ppo_training.py').read_text())
    captures=[n for n in ast.walk(tree) if isinstance(n,ast.IfExp) and isinstance(n.body,ast.Call)
              and isinstance(n.body.func,ast.Name) and n.body.func.id=='capture_before_update']
    assert len(captures)==1 and ast.unparse(captures[0].test)=='diagnostics_enabled'
    assert isinstance(captures[0].orelse,ast.Constant) and captures[0].orelse.value is None
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
           and ast.unparse(n.func)=='alg.update']
    assert len(calls)==1
