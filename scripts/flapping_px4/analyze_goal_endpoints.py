"""Independent raw-trajectory endpoint metrics for G/R and read-only historical B."""
import os,json,sqlite3,math,hashlib
from pathlib import Path
import numpy as np
import torch
from flapping_bot.path_tracking.unified_trajectory import Trajectory,TrajectorySpec
from flapping_bot.path_tracking.trajectory_feedback import FeedbackStore
from flapping_bot.analysis.unified_ppo_contract import validate_evaluation_batch,evaluation_tasks
root=Path(__file__).resolve().parents[2]
base=Path(os.environ.get('GOAL_OUTPUT_ROOT',root/'logs/unified_goal_consistent_reward_v1'))
old=root.parent/'unified-trim-centered-pilot-v1/logs/unified_trim_centered_pilot_v1'
results=[];errors={};manifest=[];traces={}
def rmse(e):
 e=np.asarray(e);return [float(np.sqrt((e[:,:2]**2).sum(1).mean())),float(np.sqrt((e[:,2]**2).mean())),float(np.sqrt((e**2).sum(1).mean()))]
for group in ['G','R','B']:
 folder=old if group=='B' else base
 for u in [0,5,10,15,20,25]:
  ev=folder/group/'evaluation_000025' if u==25 else folder/f'evaluation_{group}_{u:06d}'
  summary=json.loads((ev/'summary.json').read_text())
  ck=torch.load(folder/group/f'model_{u:06d}.pt',map_location='cpu',weights_only=False)
  digest=hashlib.sha256(b''.join(v.numpy().tobytes() for v in ck['policy'].values())).hexdigest()
  assert summary['policy_digest_before']==summary['policy_digest_after']==digest
  assert summary['version']==ck['version'] and summary['policy_unchanged'] and summary['batch_validated']
  store=FeedbackStore(ev/'evidence',readonly=True)
  records=validate_evaluation_batch(store,evaluation_tasks(),ck['plant'],ck['evaluation_contract'],ck['evaluation_initial'],ck['version'])
  manifest.append(dict(group=group,update=u,evaluation=str(ev),checkpoint=str(folder/group/f'model_{u:06d}.pt'),strict_sql_replay=True))
  for r in records:
   i=r['initial']['scene']['env_index'];trace=list(store.trace(r['instance_id']));path=Trajectory(TrajectorySpec(**r['trajectory']))
   end=path.query(path.spec.length_m);goal=end['position'];tangent=end['tangent']
   pos=np.array([s['control']['root_state'][:3] for s in trace]);d=np.linalg.norm(pos-goal,axis=1)
   err=np.array([s['control']['path_error_xyz'] for s in trace]);errors[group,u,i]=err
   traces[f'{group}_{u}_{i}']=np.c_[np.arange(1,len(trace)+1)/60,pos,err,d,[s['progress_m'] for s in trace]]
   prog=np.array([s['progress_m'] for s in trace]);dt=r['contract']['evaluation_dt_s']
   near={}
   for radius in [5.,2.,1.,.5]:
    ids=np.flatnonzero(d<=radius)
    j=int(ids[0]) if len(ids) else None
    near[str(radius)]=None if j is None else dict(time_s=trace[j]['step']*dt,step=trace[j]['step'],distance_m=float(d[j]),h_m=float(np.linalg.norm(err[j,:2])),v_m=float(abs(err[j,2])),progress_m=float(prog[j]))
   saturated=np.flatnonzero(prog>=path.spec.length_m-1e-5)
   beyond=(pos-goal)@tangent>0
   # Plane crossing is a descriptive geometric observation, not a new success rule.
   last=trace[-1];c=r['contract'];why=[]
   for name,flag in [('path_error',last['path_error_m']>c['max_path_error_m']),('height_low',last['height_m']<c['min_height_m']),('height_high',last['height_m']>c['max_height_m']),('tilt',last['tilt_cos']<c['min_tilt_cos']),('body_rate',last['body_rate_rad_s']>c['max_body_rate_rad_s']),('nonfinite',not last['finite'])]:
    if flag:why.append(name)
   strict=(prog[-1]>=path.spec.length_m-.5 and last['path_error_m']<=.5 and d[-1]<=.5 and not why)
   if group!='B':assert (r['outcome']=='completed')==strict
   minj=int(np.argmin(d));nearids=np.flatnonzero(d<=5)
   start=np.asarray(r['initial']['root_state'][:3]);chord=goal-start
   alpha=np.clip((pos-start)@chord/(chord@chord),0,1)
   chord_errors=pos-(start+alpha[:,None]*chord)
   ds=np.r_[np.linalg.norm(start-goal),d];psi=-ds/(ds+5)
   if r['outcome'] in ('completed','failed','invalid_data'):psi[-1]=0.
   shaped=.99*psi[1:]-psi[:-1]
   telescope=float(np.sum(.99**np.arange(len(shaped))*shaped)+psi[0]-.99**len(shaped)*psi[-1])
   assert abs(telescope)<1e-12
   results.append(dict(group=group,update=u,slot=i,name=evaluation_tasks()[i]['name'],steps=len(trace),time_s=len(trace)*dt,
    potential_telescoping_residual=telescope,
    travelled_distance_m=float(np.linalg.norm(np.diff(np.vstack([start,pos]),axis=0),axis=1).sum()),
    endpoint_chord_m=float(np.linalg.norm(chord)),chord_rmse_3d_m=float(np.sqrt(np.mean(np.sum(chord_errors**2,axis=1)))),
    max_path_error_m=float(np.linalg.norm(err,axis=1).max()),
    outcome=r['outcome'],failure_reasons=why,strict_completed=bool(strict),max_progress_m=float(prog.max()),
    min_goal_distance_m=float(d[minj]),final_goal_distance_m=float(d[-1]),first_entry=near,
    error_at_closest_goal_h_v_m=[float(np.linalg.norm(err[minj,:2])),float(abs(err[minj,2]))],
    near_goal_5m_rmse_h_v_3d=rmse(err[nearids]) if len(nearids) else None,
    projection_saturated=bool(len(saturated)),first_projection_saturation_s=float((saturated[0]+1)*dt) if len(saturated) else None,
    crossed_endpoint_plane=bool(beyond.any()),rmse_h_v_3d=rmse(err),prefix125_rmse_h_v_3d=rmse(err[:125]),
    strict_endpoint_entries=int(near['0.5'] is not None),
    unobserved_after_termination=True))
  store.close()
pairs=[]
for u in [0,5,10,15,20,25]:
 for i in range(5):
  n=min(len(errors[g,u,i]) for g in ['G','R']);nb=min(len(errors[g,u,i]) for g in ['G','R','B'])
  pairs.append(dict(update=u,slot=i,common_steps=n,common_with_B_steps=nb,
   **{g:dict(prefix125=rmse(errors[g,u,i][:125]),common=rmse(errors[g,u,i][:n]),common_with_B=rmse(errors[g,u,i][:nb])) for g in ['G','R','B']}))
for g in ['G','R']:
 s=json.loads((base/g/'summary.json').read_text());assert s['state']['updates']==25 and s['local_train_transitions']==307200 and s['optimizer_step_after']==500
 assert len(s['state']['evaluations'])==1 and s['state']['pending_evaluation']==False
 assert len((base/g/'training.jsonl').read_text().splitlines())==25
result=dict(passed=True,evaluations=manifest,results=results,paired_windows=pairs,final_rollouts=0,final_metrics_accessed=False,predeclared_final_geometry_metadata="inherited contract only; no simulation or tuning",
 endpoint_method='Trajectory(saved raw spec).query(L), float64; raw world root state; no cached/logged endpoint distance',
 scope='18 read-only replay batches: 12 new G/R and 6 historical B; only new G/R share terminal contract')
np.savez_compressed(base/'frozen_traces.npz',**traces)
(base/'endpoint_comparison.json').write_text(json.dumps(result,indent=2))
print('ALL_12_NEW_FROZEN_BATCHES_VERIFIED')
