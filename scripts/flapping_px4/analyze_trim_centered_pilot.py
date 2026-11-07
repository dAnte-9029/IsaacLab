import os
import json, sqlite3, math, re, xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2]
BASE=Path(os.environ.get('TRIM_OUTPUT_ROOT',ROOT/'logs/unified_trim_centered_pilot_v1'))
ORIGINAL=ROOT.parent/'unified-rl-performance-phase4/logs/unified_pilot_l3d_fixed_seed0_25u'
import sys
GROUP=sys.argv[1]
OUT=ORIGINAL if GROUP=='A' else BASE/GROUP
torch.set_num_threads(1)
def read(p): return json.loads(p.read_text())
def stats(x):
 a=np.asarray(x,dtype=float)
 return dict(n=len(a),mean=a.mean(axis=0).tolist(),median=np.median(a,axis=0).tolist(),p10=np.quantile(a,.1,axis=0).tolist(),p90=np.quantile(a,.9,axis=0).tolist(),std=a.std(axis=0).tolist()) if len(a) else None
contract=read(OUT/'contract.json'); cfg=contract['plant']['configuration']; init=contract['initial']
urdf=ET.parse(ROOT/'source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf')
limits={}
for j in urdf.findall('joint'):
 l=j.find('limit')
 if l is not None:
  lo,hi=float(l.attrib['lower']),float(l.attrib['upper']); margin=(1-cfg['joint_limit_softness'])*(hi-lo)/2; limits[j.attrib['name']]=(lo+margin,hi-margin)
angles=np.deg2rad([init['rudder_deg'],init['symmetric_deg']+init['differential_deg'],init['symmetric_deg']-init['differential_deg']])
trim=[2*(init['frequency_hz']-cfg['min_flap_hz'])/(cfg['max_flap_hz']-cfg['min_flap_hz'])-1]
for n,a in zip(['rudder','left_tail','right_tail'],angles):
 lo,hi=limits[n];trim.append(2*(a-lo)/(hi-lo)-1)
trim=np.array(trim)
checkpoints={0:torch.load(Path(str(OUT)+'_update0/evaluation_smoke_only.pt') if GROUP=='A' else OUT/'model_000000.pt',map_location='cpu',weights_only=False)}
for u in [5,10,15,20,25]: checkpoints[u]=torch.load(OUT/f'model_{u:06d}.pt',map_location='cpu',weights_only=False)
def act(obs,u):
 x=torch.tensor(np.array(obs),dtype=torch.float32); sd=checkpoints[u]['policy']
 for i in [0,2,4,6]:
  x=F.linear(x,sd[f'actor.{i}.weight'],sd[f'actor.{i}.bias'])
  if i<6: x=F.elu(x)
 return x.detach().numpy()
def obs(c,length,ids):
 r=np.array(c['root_state'],np.float32);q=r[3:7]
 def rotate(v):
  v=np.array(v,np.float32);t=np.cross(q[1:],v)*np.float32(2);return v-q[0]*t+np.cross(q[1:],t)
 ref=c['reference']; phase=np.float32(c['phase'])
 x=np.concatenate([rotate(r[7:10])/10,rotate(r[10:13])/10,rotate([0,0,-1]),rotate(ref['error'])/cfg['tracking_scale_m'],rotate(ref['tangent']),*[(rotate(np.array(p,np.float32)-r[:3])/d) for p,d in zip(ref['preview'],cfg['preview_distances_m'])],np.array([ref[k] for k in ['kappa','gamma','kappa_prime','gamma_prime']],np.float32),[np.sin(phase),np.cos(phase)],c['command'],[np.float32(c['actuator_state']['_freq'])/np.float32(5)],np.array(c['joint_pos'])[ids],[1-np.float32(c['projection']['progress'])/np.float32(length)]])
 return np.clip(x,-5,5).astype(np.float32)
def analyze(path,training=False,u=None):
 db=sqlite3.connect((path/'evidence/rollout_feedback.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)
 integrity=db.execute('pragma integrity_check').fetchone()[0]
 meta={i:json.loads(p) for i,p in db.execute('select id,metadata from episodes')}
 terminals={i:json.loads(p) for i,p in db.execute('select id,payload from terminals')}
 m=next(iter(meta.values())); ids=[int(np.argmin(abs(np.array(m['initial']['joint_pos'])-a))) for a in angles]
 assert len(set(ids))==3
 np.testing.assert_allclose(m['initial']['action'],trim,atol=2e-7)
 groups=defaultdict(lambda:defaultdict(list)); perupdate=defaultdict(lambda:defaultdict(list)); reasons=Counter(); examples={}; eps=[]; obs_samples=defaultdict(list); raw_samples=defaultdict(list); reward_sums=defaultdict(lambda:np.zeros(5)); counts=Counter(); last=None; previous=None; first_violation={}; near_runs=np.zeros(4,dtype=int); sustained=Counter(); run_groups=[[] for _ in range(4)]
 for eid,step,payload in db.execute('select episode,step,payload from steps order by episode,step'):
  s=json.loads(payload); c=s['control'];r=meta[eid];tc=r['contract']
  version=int(s['policy_version'].rsplit('update',1)[1]) if training else u
  update=version+1 if training else u;g=(update-1)//5 if training else u
  d=groups[g]; counts[update]+=1
  for name in ['raw_command','command','applied']: d[name].append(c[name])
  d['joint_deg'].append(np.rad2deg(np.array(c['joint_pos'])[ids])); ast=c['actuator_state']
  d['freq'].append([ast['_requested_frequency_hz'],ast['_applied_frequency_hz'],ast['_freq']]);d['path_error'].append(s['path_error_m'])
  perupdate[update]['path_error'].append(s['path_error_m'])
  for field in ['raw_command','command','applied']: perupdate[update][field].append(c[field])
  assert np.array_equal(np.clip(c['raw_command'],-1,1),c['command'])
  assert np.array_equal(c['applied'][1:],c['command'][1:])
  if eid!=last:
   previous=None
   for j in range(4):
    if near_runs[j]>=6:
     for gg in run_groups[j]: sustained[(gg,j)]+=1
    near_runs[j]=0;run_groups[j]=[]
  for j,a in enumerate(c['command']):
   if abs(a)>=.95: near_runs[j]+=1;run_groups[j].append(g)
   else:
    if near_runs[j]>=6:
     for gg in run_groups[j]: sustained[(gg,j)]+=1
    near_runs[j]=0;run_groups[j]=[]
  if previous is not None and version in checkpoints:
   obs_samples[version].append(obs(previous,r['trajectory']['length_m'],ids));raw_samples[version].append(c['raw_command'])
  prevhigh=previous['projection']['high_water'] if previous else 0.
  prevact=previous['command'] if previous else r['initial']['action']
  tracking=math.exp(-sum(x*x for x in c['path_error_xyz'])/cfg['tracking_scale_m']**2)
  terminal=terminals.get(eid,{})
  ending=terminal.get('outcome') if step==terminal.get('steps') else None
  parts=np.array([cfg['progress_weight']*max(0,c['projection']['high_water']-prevhigh)*tracking,cfg['tracking_weight']*(tracking-1)*tc['evaluation_dt_s'],-cfg['action_change_weight']*np.mean((np.array(c['command'])-prevact)**2),cfg['completion_bonus']*(ending=='completed'),-cfg['failure_penalty']*(ending in ['failed','invalid_data'])])
  reward_sums[update]+=parts
  triggers=[]
  for flag,yes in [('height_low',s['height_m']<tc['min_height_m']),('height_high',s['height_m']>tc['max_height_m']),('path_error',s['path_error_m']>tc['max_path_error_m']),('tilt',s['tilt_cos']<tc['min_tilt_cos']),('body_rate',s['body_rate_rad_s']>tc['max_body_rate_rad_s']),('nonfinite',not s['finite'])]:
   if yes: triggers.append(flag)
  if triggers and eid not in first_violation: first_violation[eid]=dict(step=step,triggers=triggers)
  if ending:
   e=dict(id=eid,outcome=ending,steps=step,seconds=step/60,distance=terminal['effective_distance_m'],speed=terminal['effective_distance_m']/(step/60),h=terminal['horizontal_rmse_m'],v=terminal['vertical_rmse_m'],rmse3d=math.hypot(terminal['horizontal_rmse_m'],terminal['vertical_rmse_m']),end_update=update,end_progress=s['progress_m'],triggers=triggers,height=s['height_m'],tilt=s['tilt_cos'],rate=s['body_rate_rad_s'],error=s['path_error_m'])
   if ending in ['failed','invalid_data']:
    assert first_violation[eid]['step']==step
    key='+'.join(triggers);reasons[key]+=1;examples.setdefault(key,dict(episode=eid,step=step,values={k:s[k] for k in ['height_m','tilt_cos','body_rate_rad_s','path_error_m']},root_state=c['root_state']))
   eps.append(e)
  last=eid;previous=c
 for j in range(4):
  if near_runs[j]>=6:
   for gg in run_groups[j]: sustained[(gg,j)]+=1
 result=dict(integrity=integrity,steps=sum(counts.values()),episodes=len(meta),outcomes=dict(Counter(r['outcome'] for r in terminals.values())),failure_reasons=dict(reasons),failure_examples=examples,terminal_rows=eps,groups={},per_update={},joint_indices=ids)
 for g,d in groups.items():
  o={k:stats(v) for k,v in d.items()}; raw=np.array(d['raw_command']);cmd=np.array(d['command']);ap=np.array(d['applied']);fr=np.array(d['freq'])
  o['sustained_near_limit_fraction_6steps']=[sustained[(g,j)]/len(raw) for j in range(4)]
  o.update(clip_fraction=(abs(raw)>1).mean(0).tolist(),near_limit_command=(abs(cmd)>=.95).mean(0).tolist(),near_limit_applied=(abs(ap)>=.95).mean(0).tolist(),raw_trim_distance=stats(np.linalg.norm(raw-trim,axis=1)),command_trim_distance=stats(np.linalg.norm(cmd-trim,axis=1)),applied_trim_distance=stats(np.linalg.norm(ap-trim,axis=1)),governor_limited_fraction=float((abs(fr[:,0]-fr[:,1])>1e-5).mean()))
  ee=[e for e in eps if e['outcome']!='manual_reset' and ((e['end_update']-1)//5 if training else u)==g]
  o['terminated_episode_stats']={k:stats([e[k] for e in ee]) for k in ['steps','seconds','distance','speed','h','v','rmse3d']};o['terminated_count']=len(ee);o['failure_reasons']=dict(Counter('+'.join(e['triggers']) for e in ee if e['outcome']=='failed'))
  result['groups'][str(g)]=o
 for update,d in perupdate.items():
  ee=[e for e in eps if e['end_update']==update and e['outcome']!='manual_reset']
  result['per_update'][str(update)]=dict(path_error=stats(d['path_error']),terminated=len(ee),survival=stats([e['seconds'] for e in ee]),distance=stats([e['distance'] for e in ee]),reward_parts_mean=(reward_sums[update]/counts[update]).tolist(),reward_recomputed=float(reward_sums[update].sum()/counts[update]))
 for update,d in perupdate.items():
  raw=np.array(d['raw_command']); clipped=np.array(d['command']); applied=np.array(d['applied'])
  result['per_update'][str(update)]['actions']=dict(raw=stats(raw),clipped=stats(clipped),applied=stats(applied),clip_fraction=(abs(raw)>1).mean(0).tolist(),raw_trim_distance=float(np.linalg.norm(raw-trim,axis=1).mean()),applied_trim_distance=float(np.linalg.norm(applied-trim,axis=1).mean()))
 result['actor_means']={}
 for version,x in obs_samples.items():
  means=act(x,version); raws=np.array(raw_samples[version]);sd=checkpoints[version]['policy'];std=sd.get('std')
  result['actor_means'][str(version)]=dict(n=len(x),mean_stats=stats(means),distance_trim=stats(np.linalg.norm(means-trim,axis=1)),std=std.tolist() if std is not None else None,residual_stats=stats(raws-means),frozen_reconstruction_max_error=float(abs(raws-means).max()) if not training else None)
 db.close();return result
def evaluation_path(u):
 if GROUP=='A' and u==5:return BASE/'evaluation_A_000005_original_entry'
 if u==25:return OUT/'evaluation_000025'
 if GROUP=='A' and u in (0,10):return Path(str(OUT)+f'_update{u}')
 return BASE/f'evaluation_{GROUP}_{u:06d}'
result=dict(group=GROUP,trim=trim.tolist(),configuration=cfg,training=analyze(OUT,True),frozen={},details={},same_state_drift={})
for u in [0,5,10,15,20,25]:
 path=evaluation_path(u)
 result['frozen'][str(u)]=analyze(path,u=u)
 db=sqlite3.connect((path/'evidence/rollout_feedback.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)
 rows=[];states=[]
 for eid,m in db.execute('select id,metadata from episodes'):
  m=json.loads(m);trace=[json.loads(p) for p, in db.execute('select payload from steps where episode=? order by step',(eid,))]
  last=trace[-1];c=last['control'];e=np.array(c['path_error_xyz']);ref=c['reference']
  ids=[int(np.argmin(abs(np.array(m['initial']['joint_pos'])-a))) for a in angles]
  end=next((s for s in trace if s['progress_m']>=m['trajectory']['length_m']-.001),None)
  rows.append(dict(episode=eid,slot=m['initial']['scene']['env_index'],errors=[s['control']['path_error_xyz'] for s in trace[:125]],terminal_step=last['step'],first_end_step=end['step'] if end else None,first_end_error=end['path_error_m'] if end else None,terminal_along_error=float(e@ref['tangent']),terminal_cross_error=float(e@ref['normal']),overshoot=bool(end and e@ref['tangent']<0)))
  if u==10:states.extend(obs(s['control'],m['trajectory']['length_m'],ids) for s in trace[:-1])
 result['details'][str(u)]=rows
 if states:
  torch.save(torch.tensor(np.array(states)),BASE/f'{GROUP}_update10_visited_observations.pt')
  for late in [15,20,25]:
   delta=act(states,late)-act(states,10)
   result['same_state_drift'][str(late)]=dict(n=len(states),mean=delta.mean(0).tolist(),rms=np.sqrt((delta**2).mean(0)).tolist())
 db.close()
result['checkpoint_std']={str(u):ck['policy']['std'].tolist() for u,ck in checkpoints.items()}
result['training_reports']=[json.loads(l) for l in (OUT/'training.jsonl').read_text().splitlines()]
result['reward_reconstruction_max_error']=max(abs(r['reward_mean']-result['training']['per_update'][str(r['update'])]['reward_recomputed']) for r in result['training_reports'])
(BASE/f'{GROUP}_diagnostics.json').write_text(json.dumps(result,indent=2,allow_nan=False))
print(GROUP,'ANALYSIS_COMPLETE',result['reward_reconstruction_max_error'])
