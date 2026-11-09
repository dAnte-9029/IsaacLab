"""Explicit-path, closed-archive-only G25--G150 analysis; no simulator imports."""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from flapping_bot.analysis.unified_retrospective import (
    immutable_store,observation,actor,rotate_inverse,first_event,trend_event,failure_reasons,geometry_profile)
from flapping_bot.analysis.unified_ppo_contract import validate_evaluation_batch,evaluation_tasks
from flapping_bot.analysis.unified_ppo_diagnostics import analytic_kl
from flapping_bot.path_tracking.unified_trajectory import Trajectory,TrajectorySpec

UPDATES=[25,50,75,100,125,150]
FEATURES=['straight','kappa_sustained','gamma_sustained','kappa_reversal','gamma_reversal','coupled','climb','descent','kappa_sustained40','flat_sustained40','negative_turn','positive_turn']
PRIMARY=['coupled','reverse','sustained_turn','sustained_vertical','straight_or_short']
BINS=[0,5,10,20,30,40,50,60.0001]

def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def digest(sd):return hashlib.sha256(b''.join(x.numpy().tobytes() for x in sd.values())).hexdigest()
def rms(a):return np.sqrt(np.mean(np.square(a),axis=0)).tolist()
def csv_write(path,rows):
    with Path(path).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def audit(args):
    old=args.g25;run=args.longrun;out=args.output
    files={'g25':str(old),'longrun':str(run),'comparison':str(args.comparison),'checkpoints':[], 'missing':[]}
    for p in [old/'training.jsonl',run/'training.jsonl',run/'evaluations.jsonl',args.comparison,old/'source_snapshot',run/'source_snapshot']:
        if not p.exists():files['missing'].append(str(p))
    if files['missing']:write(out/'audit.json',files);raise ValueError(files['missing'])
    reports=[readline for folder in [old,run] for readline in map(json.loads,(folder/'training.jsonl').read_text().splitlines())]
    assert [r['update'] for r in reports]==list(range(1,151))
    assert all(r['train_transitions']==12288*r['update'] for r in reports)
    final=torch.load(run/'model_000150_final.pt',map_location='cpu',weights_only=False)
    first=torch.load(old/'model_000025_final.pt',map_location='cpu',weights_only=False)
    assert final['state']['train_transitions']==1843200 and final['state']['updates']==150 and not final['state']['pending_evaluation']
    baseline=first['plant'];contract=first['contract'];ckpts={}
    for folder in [old,run]:
        for p in sorted(folder.glob('model_*.pt')):
            ck=torch.load(p,map_location='cpu',weights_only=False);u=ck['state']['updates']
            assert ck['plant']==baseline and ck['contract']==contract and ck['evaluation_contract']==first['evaluation_contract']
            assert ck['schema']=='unified_li_ppo_v2' and ck['contract']['reward_variant']=='strict_goal'
            assert ck['evaluation_contract']['kind']=='unified_terminal_v2'
            assert ck['initialization_contract']['mode']=='trim_local'
            assert ck['policy_digest']==digest(ck['policy'])
            steps=sorted(set(float(v['step']) for v in ck['optimizer']['state'].values()))
            assert steps==[20.*u] if u else not steps
            assert ck['state']['train_transitions']==12288*u
            files['checkpoints'].append(dict(path=str(p),update=u,version=ck['version'],adam_steps=steps,digest=ck['policy_digest'],saved_code_commit=ck.get('source_provenance',{}).get('code_commit'),saved_workspace_dirty=ck.get('source_provenance',{}).get('workspace_dirty')))
            if p.name==f'model_{u:06d}.pt':ckpts[u]=ck
    assert all(u in ckpts for u in range(25,151,5))
    assert read(run/'contract.json')['initial_policy_digest']==first['policy_digest']
    assert read(run/'summary.json')['optimizer_step_before']==500 and read(run/'summary.json')['optimizer_step_after']==3000
    files.update(final_updates=150,transitions=1843200,adam_before=500,adam_after=3000,checkpoint_contracts_equal=True,initial_digest_equal=True)
    files['snapshots']={}
    for folder in [old,run]:
        snap=folder/'source_snapshot';checked=0
        for name,h in baseline['identity']['physical_sources'].items():
            p=snap/'physical'/name
            assert p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==h
            checked+=1
        for name,h in contract['sampling_sources'].items():
            assert hashlib.sha256((snap/'path_tracking'/name).read_bytes()).hexdigest()==h
        assert hashlib.sha256((snap/'direct/flapping_bot/unified_trajectory_env.py').read_bytes()).hexdigest()==contract['task_source']
        files['snapshots'][str(folder)]=dict(physical_files_verified=checked)
    stores=[immutable_store(folder/'evidence') for folder in [old,run]]
    files['databases']=[]
    for s in stores:
        assert s.db.execute('pragma integrity_check').fetchone()[0]=='ok'
        counts={t:s.db.execute('select count(*) from '+t).fetchone()[0] for t in ['episodes','steps','terminals']}
        orphans=s.db.execute('select count(*) from steps s left join episodes e on s.episode=e.id where e.id is null').fetchone()[0]
        nonterminal=s.db.execute('select count(*) from episodes e left join terminals t on e.id=t.id where t.id is null').fetchone()[0]
        assert orphans==0
        files['databases'].append(dict(path=str(s.path),integrity='ok',counts=counts,orphan_steps=orphans,episodes_without_terminal=nonterminal))
    cutoff=first['evidence_sequence'];q='select * from terminals where seq<=? order by seq'
    assert stores[0].db.execute(q,(cutoff,)).fetchall()==stores[1].db.execute(q,(cutoff,)).fetchall()
    # Prefix equality includes raw metadata and EVERY stored historical control step.
    hashes=[]
    for s in stores:
        h=hashlib.sha256()
        for row in s.db.execute('select e.id,e.metadata from episodes e join terminals t on e.id=t.id where t.seq<=? order by e.id',(cutoff,)):
            h.update(json.dumps(row).encode())
        for row in s.db.execute('select s.episode,s.step,s.payload from steps s join terminals t on s.episode=t.id where t.seq<=? order by s.episode,s.step',(cutoff,)):
            h.update(json.dumps(row).encode())
        hashes.append(h.hexdigest())
    assert hashes[0]==hashes[1]
    files['historical_prefix']=dict(terminal_cutoff=cutoff,metadata_steps_terminals_exact=True,counted_once='analyze longrun SQLite only; contains historical prefix')
    for s in stores:s.close()
    write(out/'audit.json',files)
    print('AUDIT_PASS',flush=True)
    return ckpts,reports,baseline,files


def frozen(args,ckpts,plant):
    cfg=plant['configuration'];out=args.output
    import xml.etree.ElementTree as ET
    urdf=Path(plant['identity']['urdf']);assert urdf.exists()
    soft_limits={}
    for joint in ET.parse(urdf).findall('joint'):
        limit=joint.find('limit')
        if limit is not None:
            lo,hi=float(limit.attrib['lower']),float(limit.attrib['upper']);margin=(1-cfg['joint_limit_softness'])*(hi-lo)/2
            soft_limits[joint.attrib['name']]=(lo+margin,hi-margin)
    rows=[];traces={};obs_rows=[];obs_meta=[];validation=[]
    for u in UPDATES:
        folder=args.g25 if u==25 else args.longrun;ev=folder/f'evaluation_{u:06d}';s=read(ev/'summary.json');ck=ckpts[u]
        assert s['policy_digest_before']==s['policy_digest_after']==ck['policy_digest'] and s['optimizer_updates']==0
        assert s['eval_suite']=='development' and s['version']==ck['version']
        store=immutable_store(ev/'evidence');assert store.db.execute('pragma integrity_check').fetchone()[0]=='ok'
        records=validate_evaluation_batch(store,evaluation_tasks(),ck['plant'],ck['evaluation_contract'],ck['evaluation_initial'],ck['version'])
        maxerr=0.;nr=0
        for r in records:
            slot=r['initial']['scene']['env_index'];name=evaluation_tasks()[slot]['name'];tr=list(store.trace(r['instance_id']));c=[x['control'] for x in tr]
            path=Trajectory(TrajectorySpec(**r['trajectory']));progress=np.array([x['progress_m'] for x in tr]);q=path.query(progress);gp=geometry_profile(path)
            root=np.array([x['root_state'] for x in c]);t=np.arange(1,len(tr)+1)*r['contract']['evaluation_dt_s']
            e=root[:,:3]-q['position'];basis=np.stack([q['tangent'],q['normal'],q['up']],axis=1)
            local=np.einsum('nij,nj->ni',basis,e);vel=np.einsum('nij,nj->ni',basis,root[:,7:10]);end=path.query(path.spec.length_m)
            dist=np.linalg.norm(root[:,:3]-end['position'],axis=1);plane=(root[:,:3]-end['position'])@end['tangent']
            init=args.initial;angles=np.deg2rad([init['rudder_deg'],init['symmetric_deg']+init['differential_deg'],init['symmetric_deg']-init['differential_deg']])
            ids=[int(np.argmin(abs(np.asarray(r['initial']['joint_pos'])-a))) for a in angles];assert len(set(ids))==3
            joints=np.rad2deg(np.array([x['joint_pos'] for x in c])[:,ids]);raw=np.array([x['raw_command'] for x in c]);cmd=np.array([x['command'] for x in c])
            command_deg=np.column_stack([np.rad2deg(soft_limits[name][0]+(cmd[:,j+1]+1)*.5*(soft_limits[name][1]-soft_limits[name][0])) for j,name in enumerate(['rudder','left_tail','right_tail'])])
            freq=np.array([[x['actuator_state'][k] for k in ['_requested_frequency_hz','_applied_frequency_hz','_freq']] for x in c])
            euler=Rotation.from_quat(root[:,[4,5,6,3]]).as_euler('xyz',degrees=True);rates=np.rad2deg(rotate_inverse(root[:,3:7],root[:,10:13]))
            observations=np.array([observation(x,cfg,path.spec.length_m,ids) for x in c[:-1]])
            predicted=actor(observations,ck['policy']).numpy();error=float(np.max(abs(predicted-raw[1:])))
            maxerr=max(maxerr,error);nr+=len(observations)
            assert error<2e-5,('observation reconstruction',u,name,error)
            # Fixed deterministic selection: union of all source policies, all five tasks.
            # Rows always identify the PRE-action state at SQLite step k, tested against action k+1.
            selections=defaultdict(set)
            def add(label,idx):
                for j in np.atleast_1d(idx):
                    if 0<=j<len(observations):selections[int(j)].add(label)
            add('early',np.arange(0,min(60,len(observations)),10))
            for label,mask in [('lateral_onset',abs(local[:,1])>1),('vertical_change',abs(local[:,2])>.5),('near_goal',dist<10)]:
                ix=np.flatnonzero(mask)
                if len(ix):add(label,np.arange(max(0,ix[0]-5),min(len(observations),ix[0]+6)))
            add('pre_failure',np.arange(max(0,len(observations)-12),len(observations)))
            add('progress_grid',np.arange(0,len(observations),30))
            if name=='sustained':add('sustained',np.flatnonzero(abs(q['kappa'])>.002)[::15])
            if 'reverse' in name:
                gi=np.minimum(np.searchsorted(gp['grid'],progress),len(gp['grid'])-1);add('reverse',np.flatnonzero(gp['kappa_reversal'][gi])[::10])
            if name=='coupled_3d':add('coupled',np.flatnonzero((abs(q['kappa'])>.002)&(abs(q['gamma'])>.02))[::15])
            for j,labels in sorted(selections.items()):
                obs_rows.append(observations[j]);obs_meta.append(dict(source_update=u,task=name,episode=r['instance_id'],state_step=j+1,action_step=j+2,time_s=float(t[j]),progress_m=float(progress[j]),labels=sorted(labels)))
            events={}
            for axis,col in [('lateral',1),('vertical',2)]:
                for threshold in [1,2,4]:events[f'{axis}_abs_gt_{threshold}m_s']=first_event(abs(local[:,col])>threshold,t)
            for threshold in [2,4,6]:events[f'path_error_gt_{threshold}m_s']=first_event(np.linalg.norm(e,axis=1)>threshold,t)
            events.update(path_norm_outward=trend_event(np.linalg.norm(e,axis=1),t,1),path_norm_recovery=trend_event(np.linalg.norm(e,axis=1),t,-1),outward=trend_event(local[:,1],t,1),recovery=trend_event(local[:,1],t,-1),projection_60m_s=first_event(progress>=60-1e-5,t),cross_endpoint_plane_s=first_event(plane>0,t))
            j=int(dist.argmin());events['closest_goal_s']=float(t[j]);first4=np.flatnonzero(np.linalg.norm(e,axis=1)>4)
            stage=('midpath_divergence' if len(first4) and progress[first4[0]]<50 else 'late_path_or_endpoint')
            rows.append(dict(update=u,task=name,steps=len(tr),survival_s=float(t[-1]),min_goal_m=float(dist[j]),effective_distance_m=r['effective_distance_m'],rmse_path_t_y_z=rms(local),events=events,outcome=r['outcome'],failure_reasons=failure_reasons(tr[-1],r['contract']),terminal_root_state=root[-1].tolist(),terminal_local_error=local[-1].tolist(),terminal_local_velocity=vel[-1].tolist(),terminal_rpy_deg=euler[-1].tolist(),terminal_body_rates_deg_s=rates[-1].tolist(),failure_stage=stage,first_4m_progress=float(progress[first4[0]]) if len(first4) else None,max_same_sign_turn_m=gp['kappa_max_same_sign_m']))
            array=np.column_stack([t,progress,local,vel,dist,plane,euler,rates,joints,cmd,raw,freq,q['kappa'],q['gamma'],np.gradient(local[:,1],t),.5*(joints[:,1]+joints[:,2]),.5*(joints[:,1]-joints[:,2]),.5*(cmd[:,2]+cmd[:,3]),.5*(cmd[:,2]-cmd[:,3])])
            body_vel=rotate_inverse(root[:,3:7],root[:,7:10])
            course=np.rad2deg(np.arctan2(root[:,8],root[:,7]));heading=np.rad2deg(q['heading'])
            wrap=lambda x:(x+180)%360-180
            extra=np.column_stack([command_deg,.5*(command_deg[:,1]+command_deg[:,2]),.5*(command_deg[:,1]-command_deg[:,2]),cfg['min_flap_hz']+(raw[:,0]+1)*.5*(cfg['max_flap_hz']-cfg['min_flap_hz']),q['curvature'],heading,course,wrap(course-heading),wrap(euler[:,2]-heading),np.rad2deg(np.arctan2(body_vel[:,1],body_vel[:,0]))])
            array=np.column_stack([array,extra])
            names=['time_s','progress_m','e_parallel_m','e_y_m','e_z_m','v_parallel_mps','v_y_mps','v_z_mps','goal_distance_m','endpoint_plane_m','roll_deg','pitch_deg','yaw_deg','p_deg_s','q_deg_s','r_deg_s','rudder_joint_deg','left_joint_deg','right_joint_deg','freq_cmd_norm','rudder_cmd_norm','left_cmd_norm','right_cmd_norm','freq_raw_norm','rudder_raw_norm','left_raw_norm','right_raw_norm','requested_hz','governor_hz','actual_hz','kappa_rad_per_m','gamma_rad','de_y_dt_mps','symmetric_joint_deg','differential_joint_deg','symmetric_cmd_norm','differential_cmd_norm']
            names+=['rudder_command_deg','left_command_deg','right_command_deg','symmetric_command_deg','differential_command_deg','raw_frequency_hz','curvature_3d_per_m','path_heading_deg','course_deg','course_error_deg','yaw_error_deg','body_velocity_angle_deg']
            np.savetxt(out/f'trace_G{u}_{name}.csv',array,delimiter=',',header=','.join(names),comments='',fmt='%.9g');traces[u,name]=(array,names)
        validation.append(dict(update=u,rows_verified=nr,max_abs_raw_action_error=maxerr,strict_replay=True,checkpoint=str(folder/f'model_{u:06d}.pt')));store.close()
    historical=read(args.comparison)
    for r in rows:
        old=next(x for x in historical['tasks'] if x['update']==r['update'] and x['name']==r['task'])
        assert abs(old['min_goal_m']-r['min_goal_m'])<1e-8
    write(out/'frozen_events.json',rows);write(out/'observation_validation.json',validation)
    np.savez_compressed(out/'fixed_observations.npz',observations=np.array(obs_rows));write(out/'fixed_state_index.json',obs_meta)
    x=np.array(obs_rows);means={u:actor(x,ck['policy']) for u,ck in ckpts.items() if u>=25};drift=[]
    pairs=list(zip(UPDATES[:-1],UPDATES[1:]))+[(25,150)]+list(zip(range(25,150,5),range(30,151,5)))
    for a,b in dict.fromkeys(pairs):
        delta=means[b]-means[a];kl=analytic_kl(means[a].double(),ckpts[a]['policy']['std'].double(),means[b].double(),ckpts[b]['policy']['std'].double())
        masks={'all':np.ones(len(x),bool)}
        masks.update({name:np.array([m['task']==name for m in obs_meta]) for name in [t['name'] for t in evaluation_tasks()]})
        masks.update({'phase:'+label:np.array([label in m['labels'] for m in obs_meta]) for label in sorted({l for m in obs_meta for l in m['labels']})})
        for label,mask in masks.items():
            if not mask.any():continue
            d=delta[mask];k=kl[mask];derived=torch.stack([d[:,1],(d[:,2]+d[:,3])/2,(d[:,2]-d[:,3])/2],-1)
            drift.append(dict(old=a,new=b,states=label,n=int(mask.sum()),mean_delta=d.mean(0).tolist(),rms_delta=d.square().mean(0).sqrt().tolist(),p90_abs_delta=torch.quantile(abs(d),.9,dim=0).tolist(),rudder_symmetric_differential_rms=derived.square().mean(0).sqrt().tolist(),kl_old_to_new_mean=float(k.mean()),kl_p90=float(torch.quantile(k,.9))))
    write(out/'fixed_state_drift.json',dict(scope='offline_saved_checkpoint_KL_not_PPO_minibatch_KL',pairs=drift))
    np.savez_compressed(out/'fixed_actor_means.npz',**{f'G{u}':m.numpy() for u,m in means.items()})
    print('FROZEN_AND_OBSERVATION_PASS',validation,flush=True)
    return rows,traces


def training(args,reports,plant):
    out=args.output;cfg=plant['configuration'];store=immutable_store(args.longrun/'evidence')
    progress_gain=np.zeros(151);counts=np.zeros(151,np.int64);progress_counts=np.zeros((151,len(BINS)-1),np.int64)
    labels=np.zeros((151,len(FEATURES),len(BINS)-1),np.int64);primary=np.zeros((151,len(PRIMARY),len(BINS)-1),np.int64)
    reward=np.zeros((151,5));clipping=np.zeros((151,4),np.int64);gov=np.zeros(151,np.int64);fail=defaultdict(Counter);terminal_rows=[]
    mixed=0;manual=0;versions_seen=set();episode_count=0
    for eid,meta in store.db.execute('select id,metadata from episodes order by id'):
        m=json.loads(meta)
        assert m['plant']==plant and m['contract']['kind']=='unified_terminal_v2'
        tr=[json.loads(p) for p, in store.db.execute('select payload from steps where episode=? order by step',(eid,))]
        row=store.db.execute('select payload from terminals where id=?',(eid,)).fetchone();terminal=json.loads(row[0]) if row else None
        if not tr:
            assert terminal and terminal['outcome']=='manual_reset' and terminal['steps']==0
            manual+=1
            terminal_rows.append(dict(episode=eid,last_rollout_update=None,outcome='manual_reset',mixed_versions=False,versions=[],steps=0,effective_distance_m=0.,failure_reasons=[],right_censored=True))
            continue
        assert [s['step'] for s in tr]==list(range(1,len(tr)+1))
        assert all(s['execution_mode']=='training' for s in tr)
        versions=[s['policy_version'] for s in tr];versions_seen.update(versions);mixed+=len(set(versions))>1
        u=np.array([int(v.rsplit('update',1)[1])+1 for v in versions]);assert u.min()>=1 and u.max()<=150
        c=[s['control'] for s in tr];path=Trajectory(TrajectorySpec(**m['trajectory']));gp=geometry_profile(path)
        progress=np.array([s['progress_m'] for s in tr]);preprogress=np.r_[0,progress[:-1]]
        bins=np.minimum(np.searchsorted(BINS,preprogress,side='right')-1,len(BINS)-2)
        gi=np.minimum(np.searchsorted(gp['grid'],preprogress),len(gp['grid'])-1)
        np.add.at(counts,u,1);np.add.at(progress_counts,(u,bins),1);np.add.at(primary,(u,PRIMARY.index(gp['primary']),bins),1)
        for j,key in enumerate(FEATURES):
            mask=gp[key][gi];np.add.at(labels,(u[mask],j,bins[mask]),1)
        high=np.array([x['projection']['high_water'] for x in c]);dh=np.maximum(0,np.diff(np.r_[0,high]))
        np.add.at(progress_gain,u,dh)
        err=np.array([x['path_error_xyz'] for x in c]);tracking=np.exp(-np.sum(err**2,axis=1)/cfg['tracking_scale_m']**2)
        command=np.array([x['command'] for x in c]);raw=np.array([x['raw_command'] for x in c]);prev=np.vstack([m['initial']['action'],command[:-1]])
        parts=np.column_stack([cfg['progress_weight']*dh*tracking,cfg['tracking_weight']*(tracking-1)*m['contract']['evaluation_dt_s'],-cfg['action_change_weight']*np.mean((command-prev)**2,axis=1),np.zeros((len(tr),2))])
        if terminal:
            assert terminal['steps']==len(tr) and set(terminal['policy_versions'])==set(versions)
            outcome=terminal['outcome'];manual+=outcome=='manual_reset'
            if outcome=='completed':parts[-1,3]=cfg['completion_bonus']
            if outcome in ['failed','invalid_data']:parts[-1,4]=-cfg['failure_penalty']
            reasons=failure_reasons(tr[-1],m['contract'])
            if outcome in ['failed','invalid_data']:fail[int(u[-1])].update(reasons)
            terminal_rows.append(dict(episode=eid,last_rollout_update=int(u[-1]),outcome=outcome,mixed_versions=len(set(versions))>1,versions=sorted(set(versions)),steps=len(tr),effective_distance_m=terminal['effective_distance_m'],failure_reasons=reasons,right_censored=outcome=='manual_reset',primary=gp['primary'],max_same_sign_turn_m=gp['kappa_max_same_sign_m'],pre_action_progress_bins=np.bincount(bins,minlength=len(BINS)-1).tolist()))
        np.add.at(reward,u,parts);np.add.at(clipping,u,abs(raw)>1)
        freq=np.array([[x['actuator_state'][k] for k in ['_requested_frequency_hz','_applied_frequency_hz']] for x in c]);np.add.at(gov,u,abs(freq[:,0]-freq[:,1])>1e-5)
        episode_count+=1
        if episode_count%1000==0:print('TRAIN_ARCHIVE_EPISODES',episode_count,'STEPS',int(counts.sum()),flush=True)
    assert np.all(counts[1:]==12288),(counts.tolist())
    assert counts.sum()==1843200 and np.array_equal(primary.sum(1),progress_counts)
    per=[]
    for r in reports:
        u=r['update'];parts=reward[u]/counts[u];delta=abs(float(parts.sum())-r['reward_mean']);assert delta<1e-6,(u,delta)
        ends=[x for x in terminal_rows if x['last_rollout_update']==u]
        per.append(dict(update=u,rollout_policy_update=u-1,transitions=int(counts[u]),new_high_water_distance_sum_m=float(progress_gain[u]),reward_reconstructed=float(parts.sum()),reward_logged=r['reward_mean'],reward_abs_error=delta,reward_progress=float(parts[0]),reward_tracking=float(parts[1]),reward_action_change=float(parts[2]),reward_success=float(parts[3]),reward_failure=float(parts[4]),progress_bins=progress_counts[u].tolist(),primary_counts={k:primary[u,j].tolist() for j,k in enumerate(PRIMARY)},feature_counts={k:labels[u,j].tolist() for j,k in enumerate(FEATURES)},raw_clip_fraction=(clipping[u]/counts[u]).tolist(),governor_limited_fraction=float(gov[u]/counts[u]),policy_std_after=r['policy_std'],parameter_delta_l2=r['parameter_delta_l2'],losses=r['losses'],failure_reasons=dict(fail[u]),terminal_outcomes=dict(Counter(x['outcome'] for x in ends)),mean_effective_distance_terminal_noncensored=float(np.mean([x['effective_distance_m'] for x in ends if not x['right_censored']])) if any(not x['right_censored'] for x in ends) else None))
    result=dict(counted_database=str(store.path),deduplication='longrun only, inherited G1-25 prefix counted once',total_steps=int(counts.sum()),mixed_version_episodes=mixed,manual_reset_right_censored=manual,episodes_with_steps=episode_count,progress_bins_m=BINS,local_feature_thresholds=dict(kappa_rad_per_m=.002,gamma_rad=.02,sustained_same_sign_m=10,reversal_accepted_run_m=2,reversal_window_m=5),progress_attribution='pre-action progress; first interval starts at zero',primary_precedence=PRIMARY,per_update=per)
    write(out/'training_reconstruction.json',result);write(out/'training_terminals.json',terminal_rows)
    csv_write(out/'reward_per_update.csv',[{k:v for k,v in x.items() if not isinstance(v,(dict,list))} for x in per])
    store.close();print('TRAIN_RECONSTRUCTION_PASS',counts.sum(),'reward max',max(x['reward_abs_error'] for x in per),flush=True)


def plots(args,rows,traces):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=args.output
    for task in [x['name'] for x in evaluation_tasks()]:
        fig,axes=plt.subplots(3,2,figsize=(12,9),constrained_layout=True)
        for u in UPDATES:
            a,n=traces[u,task]
            for j,col in enumerate(['e_y_m','e_z_m','goal_distance_m']):
                for k,x in enumerate(['time_s','progress_m']):axes[j,k].plot(a[:,n.index(x)],a[:,n.index(col)],label=f'G{u}',lw=1.2)
        for j,label in enumerate(['Path lateral error (m)','Path vertical error (m)','True goal distance (m)']):
            for k in range(2):axes[j,k].set_ylabel(label);axes[j,k].grid(alpha=.2);axes[j,k].set_xlabel('time (s)' if k==0 else 'projected progress (m)')
        axes[0,0].legend(ncol=3);fig.suptitle(task+' | x - p in analytic path frame');fig.savefig(out/f'errors_{task}.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(6,2,figsize=(14,17),constrained_layout=True)
    fields=[['roll_deg','pitch_deg','yaw_deg'],['p_deg_s','q_deg_s','r_deg_s'],['rudder_cmd_norm','symmetric_cmd_norm','differential_cmd_norm'],['rudder_joint_deg','symmetric_joint_deg','differential_joint_deg'],['freq_raw_norm'],['requested_hz','governor_hz','actual_hz'],['e_y_m','e_z_m'],['v_y_mps','de_y_dt_mps'],['kappa_rad_per_m'],['progress_m'],['goal_distance_m'],['left_joint_deg','right_joint_deg']]
    for ax,fs in zip(axes.flat,fields):
        for u in [25,50,150]:
            a,n=traces[u,'sustained']
            for j,f in enumerate(fs):ax.plot(a[:,0],a[:,n.index(f)],label=f'G{u} {f}',linestyle=['-','--',':'][j])
        ax.legend(fontsize=6,ncol=2);ax.set_xlabel('time (s)');ax.grid(alpha=.2)
    fig.suptitle('Sustained: commands / actual joints / body / path; observational only');fig.savefig(out/'sustained_joint_diagnostics.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,field in zip(axs.flat,['path_heading_deg','course_error_deg','yaw_error_deg','body_velocity_angle_deg']):
        for u in UPDATES:
            a,n=traces[u,'sustained'];ax.plot(a[:,0],a[:,n.index(field)],label=f'G{u}')
        ax.set_ylabel(field);ax.set_xlabel('time (s)');ax.grid(alpha=.2)
    axs[0,0].legend(ncol=3);fig.savefig(out/'sustained_heading.png',dpi=150);plt.close(fig)
    data=read(out/'training_reconstruction.json')['per_update'];x=[r['update'] for r in data]
    fig,axs=plt.subplots(3,1,figsize=(12,10),constrained_layout=True)
    for f in FEATURES:axs[0].plot(x,[sum(r['feature_counts'][f])/r['transitions'] for r in data],label=f)
    axs[0].legend(ncol=4,fontsize=8);axs[0].set_ylabel('Overlapping feature fraction')
    mat=np.array([r['progress_bins'] for r in data]).T/12288
    im=axs[1].imshow(mat,aspect='auto',origin='lower',extent=[.5,150.5,-.5,6.5],vmin=0);axs[1].set_yticks(range(7),['0-5','5-10','10-20','20-30','30-40','40-50','50-60']);fig.colorbar(im,ax=axs[1]);axs[1].set_ylabel('Pre-action progress (m)')
    for f in ['reward_reconstructed','reward_progress','reward_tracking','reward_action_change','reward_success','reward_failure']:axs[2].plot(x,[r[f] for r in data],label=f)
    axs[2].legend(ncol=3,fontsize=8);axs[2].set_xlabel('PPO update');axs[2].set_ylabel('Mean reward / transition');fig.savefig(out/'training_coverage_reward.png',dpi=150);plt.close(fig)
    drift=read(out/'fixed_state_drift.json')['pairs'];fig,axs=plt.subplots(2,1,figsize=(11,7),constrained_layout=True)
    ds=[d for d in drift if d['states']=='all' and d['new']-d['old']==5]
    axs[0].plot([d['new'] for d in ds],[d['kl_old_to_new_mean'] for d in ds],'-o');axs[0].set_ylabel('Offline KL(old || new)')
    for j,label in enumerate(['frequency','rudder','left','right']):axs[1].plot([d['new'] for d in ds],[d['rms_delta'][j] for d in ds],label=label)
    axs[1].legend();axs[1].set_ylabel('Same-state raw mean RMS delta');axs[1].set_xlabel('Saved checkpoint update (5-update intervals)');fig.savefig(out/'fixed_state_drift.png',dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['g25','longrun','comparison','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--refresh-derived',action='store_true',help='Explicitly regenerate this analysis output only')
    args=p.parse_args()
    args.output=args.output.resolve()
    for source in [args.g25.resolve(),args.longrun.resolve(),args.comparison.resolve().parent]:
        if args.output==source or source in args.output.parents:
            raise ValueError('Analysis output must be outside historical input directories')
    args.output.mkdir(parents=True,exist_ok=args.refresh_derived);torch.set_num_threads(1)
    write(args.output/'analysis_status.json',dict(status='running'))
    args.g25=args.g25.resolve();args.longrun=args.longrun.resolve();args.initial=read(args.g25/'contract.json')['initial']
    start=time.perf_counter();ck,reports,plant,au=audit(args);rows,traces=frozen(args,ck,plant);training(args,reports,plant);plots(args,rows,traces)
    write(args.output/'analysis_status.json',dict(status='completed',wall_s=time.perf_counter()-start,formal_updates=0,physx_rollouts=0,new_evaluations=0))
if __name__=='__main__':main()
