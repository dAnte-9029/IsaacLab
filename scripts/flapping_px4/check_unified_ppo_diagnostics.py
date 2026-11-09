"""CPU-only diagnostics overhead and saved-policy repeatability, explicit results path."""
import argparse
import inspect
import json
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from flapping_bot.analysis.unified_ppo_contract import POLICY,PPO_CONFIG
from flapping_bot.analysis.unified_ppo_diagnostics import capture_before_update,diagnose_after_update
from flapping_bot.analysis.unified_retrospective import actor


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    args=parser.parse_args();out=args.results
    torch.set_num_threads(1);torch.manual_seed(19)
    obs=TensorDict({'policy':torch.randn(256,39)},batch_size=[256])
    policy=ActorCritic(obs,{'policy':['policy'],'critic':['policy']},4,**POLICY)
    alg=PPO(policy,device='cpu',**PPO_CONFIG);alg.init_storage('rl',256,48,obs,[4]);s=alg.storage
    with torch.no_grad():
        policy.std.copy_(torch.tensor([.15,.2,.25,.3]))
        s.observations['policy'].copy_(torch.randn(48,256,39));flat=s.observations.flatten(0,1)
        s.mu.copy_(policy.act_inference(flat).reshape(48,256,4));s.sigma.copy_(policy.std.expand_as(s.mu))
        s.actions.copy_(s.mu+torch.randn_like(s.mu)*s.sigma)
        s.actions_log_prob.copy_(torch.distributions.Normal(s.mu,s.sigma).log_prob(s.actions).sum(-1,keepdim=True))
        s.values.copy_(policy.evaluate(flat).reshape(48,256,1));s.returns.copy_(s.values+torch.randn_like(s.values))
        s.advantages.copy_(s.returns-s.values)
    s.step=48
    with torch.inference_mode():
        s.rewards.copy_(torch.randn_like(s.rewards))
        alg.compute_returns(obs)
    assert s.advantages.is_inference()
    timings=[]
    for _ in range(6):
        batch=capture_before_update(alg)
        result=diagnose_after_update(alg,batch,update=0,rollout_policy_version='synthetic',updated_policy_version='synthetic')
        timings.append(result['diagnostics_wall_s'])
    x=np.load(out/'fixed_observations.npz')['observations'];saved=np.load(out/'fixed_actor_means.npz')
    audit=json.loads((out/'audit.json').read_text());errors={}
    for u in [25,50,75,100,125,150]:
        path=next(r['path'] for r in audit['checkpoints'] if Path(r['path']).name==f'model_{u:06d}.pt')
        ck=torch.load(path,map_location='cpu',weights_only=False)
        means=actor(x,ck['policy']).numpy();errors[str(u)]=float(abs(means-saved[f'G{u}']).max());assert errors[str(u)]==0
    result=dict(synthetic_batch_size=12288,diagnostics_wall_s=timings,median_warm_s=float(np.median(timings[1:])),
                max_saved_actor_reproduction_error=errors,installed_ppo_path=inspect.getfile(PPO),
                gradient_norm='unavailable',physx_rollouts=0,formal_training_updates=0,optimizer_updates_in_benchmark=0)
    (out/'offline_acceptance.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
