"""Verify fixed budgets, all frozen batches, policy identity and paired source/configuration."""
import os,json,hashlib
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[2];base=Path(os.environ.get('TRIM_OUTPUT_ROOT',root/'logs/unified_trim_centered_pilot_v1'))
a=root.parent/'unified-rl-performance-phase4/logs/unified_pilot_l3d_fixed_seed0_25u'
def read(p):return json.loads(p.read_text())
manifest=[]
for group in 'AMB':
 train=a if group=='A' else base/group
 for u in [0,5,10,15,20,25]:
  if group=='A' and u in [0,10]:ev=Path(str(a)+f'_update{u}')
  elif u==25:ev=train/'evaluation_000025'
  elif group=='A' and u==5:ev=base/'evaluation_A_000005_original_entry'
  else:ev=base/f'evaluation_{group}_{u:06d}'
  s=read(ev/'summary.json')
  assert all(s[k] is True for k in ['passed','batch_validated','terminal_masks_match','policy_unchanged'])
  assert s['eval_step_budget']==7200 and s['eval_suite']=='development' and len(s['tasks'])==5
  ck=Path(str(a)+'_update0/evaluation_smoke_only.pt') if group=='A' and u==0 else train/f'model_{u:06d}.pt'
  c=torch.load(ck,weights_only=False)
  assert s['version']==c['version']
  # state_dict starts with std followed by Actor/Critic, matching policy.parameters() here.
  digest=hashlib.sha256(b''.join(v.numpy().tobytes() for v in c['policy'].values())).hexdigest()
  assert s['policy_digest_before']==s['policy_digest_after']==digest
  manifest.append(dict(group=group,update=u,checkpoint=str(ck),evaluation=str(ev),version=c['version'],policy_digest=digest,initialization_contract=c.get('initialization_contract'),strict_replay=True))
for group in ['M','B']:
 s=read(base/group/'summary.json');assert s['state']['updates']==25 and s['local_train_transitions']==307200
 assert s['optimizer_step_after']==500
 assert s['state']['eval_env_steps']>0 and len(s['state']['evaluations'])==1
 assert len((base/group/'training.jsonl').read_text().splitlines())==25
assert (base/'M/source_snapshot/analysis/unified_ppo_training.py').read_bytes()==(base/'B/source_snapshot/analysis/unified_ppo_training.py').read_bytes()
x=read(base/'M/contract.json')['plant'];y=read(a/'contract.json')['plant'];diff=[]
def compare(x,y,p=''):
 if isinstance(x,dict):
  assert x.keys()==y.keys()
  for k in x:compare(x[k],y[k],p+'/'+k)
 elif x!=y:diff.append(p)
compare(x,y)
assert set(diff)=={'/configuration/scene/robot/spawn/asset_path','/configuration/scene/robot/spawn/usd_dir','/configuration/robot/spawn/asset_path','/identity/urdf'}
assert read(base/'M/contract.json')['plant']==read(base/'B/contract.json')['plant']
(base/'verified_evaluations.json').write_text(json.dumps(dict(passed=True,evaluations=manifest,physical_path_only_differences=diff,training_transitions=614400,reproduction_transitions=12288),indent=2))
print('ALL_18_FROZEN_BATCHES_VERIFIED')
