"""Eight bounded independent-process interventions; fail closed on historical replay.

This entry never constructs PPO, calls an optimizer, or publishes evaluation evidence.
Run with the repository wrapper and the configured env_isaaclab environment.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback

SCHEMA = 'sustained_cross_policy_intervention_v1'
ROOT = Path(__file__).resolve().parents[2]
# Declared before any replay; no relative tolerance or post-failure adjustment.
TOLERANCES = dict(action=2e-6, physical=2e-5, observation=2e-5)
GROUPS = dict(A=(25,25), B=(25,150), C=(150,25), D=(150,150))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def differences(a, b, path=''):
    if isinstance(a, dict) and isinstance(b, dict):
        result=[]
        for key in sorted(a.keys() | b.keys()):
            result += differences(a.get(key), b.get(key), path+'.'+key)
        return result
    return [] if a==b else [dict(path=path, actual=a, saved=b)]


def numeric_compare(a, b, maxima, path='', *, exact=False):
    import numpy as np
    if isinstance(a, dict):
        if a.keys()!=b.keys(): raise ValueError(('Different fields',path))
        for k in a: numeric_compare(a[k], b[k], maxima, path+'.'+k, exact=exact)
    else:
        x,y=np.asarray(a),np.asarray(b)
        if x.shape!=y.shape: raise ValueError(('Different shape',path,x.shape,y.shape))
        if x.dtype.kind not in 'biuf':
            if not np.array_equal(x,y): raise ValueError(('Different value',path))
            return
        error=float(np.max(np.abs(x.astype(float)-y.astype(float)))) if x.size else 0.
        if not np.isfinite(error): raise ValueError(('Nonfinite',path))
        maxima[path]=max(maxima.get(path,0.),error)
        tol=0. if exact else TOLERANCES['action' if any(k in path for k in ('raw_command','command','applied')) else 'physical']
        if error>tol: raise ValueError(('Replay mismatch',path,error,tol))


def select_actions(mu25, mu150, group, slot):
    """Only the Sustained slot changes; all other slots retain G150."""
    out=mu150.clone()
    f,t=GROUPS[group]
    out[slot,0]=(mu25 if f==25 else mu150)[slot,0]
    out[slot,1:]=(mu25 if t==25 else mu150)[slot,1:]
    return out


def rng_state():
    import torch
    import numpy as np
    return (torch.get_rng_state().clone(), deepcopy(np.random.get_state()), random.getstate())


def require_rng(before):
    import torch
    import numpy as np
    after=rng_state()
    assert torch.equal(before[0],after[0]) and before[2]==after[2]
    assert before[1][0]==after[1][0] and np.array_equal(before[1][1],after[1][1]) and before[1][2:]==after[1][2:]


def child(args):
    import numpy as np
    import torch
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,device='cpu',kit_args=(
        f'--ext-folder {args.plant_assets}/source/flapping_bot/native_extensions '
        '--enable omni.flapping_bot.holonomic_constraint')).app
    env=None
    try:
        from flapping_bot.analysis.unified_ppo_training import build_env, check_initial, tensor_obs, make_policy, validate_saved_policy, policy_digest
        from flapping_bot.analysis.unified_ppo_contract import evaluation_tasks, require_evaluation_checkpoint
        from flapping_bot.analysis.unified_retrospective import immutable_store, observation
        checkpoints={u:torch.load(getattr(args,f'g{u}'),map_location='cpu',weights_only=False) for u in (25,150)}
        for key in ('plant','evaluation_contract','policy_config'):
            assert checkpoints[25][key]==checkpoints[150][key],key
        store=immutable_store(args.history/'evidence')
        assert store.db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        records=store.records
        byslot={r['initial']['scene']['env_index']:r for r in records}
        history={i:list(store.trace(r['instance_id'])) for i,r in byslot.items()}
        store.close()
        summary=json.loads((args.history/'summary.json').read_text())
        assert summary['policy_digest_before']==summary['policy_digest_after']==checkpoints[150]['policy_digest']
        tasks=evaluation_tasks('development');slot=next(i for i,t in enumerate(tasks) if t['name']=='sustained')
        for u,c in checkpoints.items():
            assert c['schema']=='unified_li_ppo_v2' and c['state']['updates']==u
            require_evaluation_checkpoint(c,backend='L3D',sampling='fixed',suite='development',budget=7200,smoke_evaluation=False)
        torch.set_num_threads(1);torch.manual_seed(0)
        args.backend='L3D';args.sampling='fixed';args.seed=0;args.num_envs=5
        args.eval_suite='development';args.reward_variant='strict_goal';args.smoke_evaluation=False
        env=build_env(args,ROOT,evaluation=True)
        # The production builder initializes a journal; disconnect before reset/steps.
        # It remains an empty bootstrap journal, never a frozen evaluation receipt.
        env.rollout_evaluator.store.close();env.rollout_evaluator=None
        assert env.online_sampler is None
        saved=checkpoints[150]['plant'];actual=env.plant_identity
        diff=differences(actual,saved)
        allowed={'.configuration.scene.robot.spawn.asset_path','.configuration.scene.robot.spawn.usd_dir',
                 '.configuration.robot.spawn.asset_path','.identity.urdf'}
        write(args.output/'physical_comparison.json',dict(differences=diff,allowed_path_fields=sorted(allowed),actual=actual,saved=saved))
        assert all(d['path'] in allowed for d in diff),diff
        assert actual['identity']['physical_sources']==saved['identity']['physical_sources']
        assert env.evaluation_contract==checkpoints[150]['evaluation_contract']
        # Compare every URDF mesh dependency as well as the URDF/CSV/source identity.
        import xml.etree.ElementTree as ET
        current=Path(actual['identity']['urdf']);old=Path(saved['identity']['urdf'])
        meshes={}
        for mesh in ET.parse(current).findall('.//mesh'):
            rel=mesh.attrib['filename'];x=(current.parent/rel).resolve();y=(old.parent/rel).resolve()
            hx=hashlib.sha256(x.read_bytes()).hexdigest();hy=hashlib.sha256(y.read_bytes()).hexdigest()
            assert hx==hy;meshes[rel]=hx
        obs,_=env.reset();check_initial(env,'L3D');obs=tensor_obs(obs,5)
        assert tuple(obs['policy'].shape)==(5,39)
        policies={}
        for u,c in checkpoints.items():
            p=make_policy(obs);p.load_state_dict(c['policy'],strict=True);p.requires_grad_(False);p.eval()
            validate_saved_policy(c,p)
            assert all(torch.equal(v,c['policy'][k]) for k,v in p.state_dict().items())
            policies[u]=p
        ids=[int(env._joint_ids[j]) for j in (env._IDX_RUDDER,env._IDX_LEFT_TAIL,env._IDX_RIGHT_TAIL)]
        initial=[];maxima={}
        for i in range(5):
            d=env._robot.data
            state=dict(root_state=d.root_state_w[i].tolist(),joint_pos=d.joint_pos[i].tolist(),joint_vel=d.joint_vel[i].tolist(),
                phase=float(env._phase[i]),frequency=float(env._freq[i]),action=env._actions[i].tolist(),
                applied_action=env._act_cmd[i].tolist(),wind=env._wind_w[i].tolist(),env_origin=env.scene.env_origins[i].tolist(),
                scene=dict(num_envs=5,env_index=i,env_spacing_m=10.))
            numeric_compare(state,byslot[i]['initial'],maxima,f'reset.slot{i}')
            assert env.task_records[i]==byslot[i]['trajectory']
            initial.append(state)
        write(args.output/'contract.json',dict(schema=SCHEMA,code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            group=args.group,switch_after_step=args.start,window_steps=60,slot=slot,seed=0,dt=env.step_dt,
            checkpoint={u:dict(path=str(getattr(args,f'g{u}')),digest=c['policy_digest'],version=c['version']) for u,c in checkpoints.items()},
            tolerances=TOLERANCES,mesh_sha256=meshes,initial=initial,tasks=env.task_records,joint_ids=ids,
            hidden_physx_state='not directly observable; complete input history and D replay are required',
            evidence_use='counterfactual only; no sampler witness or development score',optimizer_updates=0))
        # Capture terminal physical state before DirectRLEnv can reset it.
        original_dones=env._get_dones;terminal_capture={}
        def capture_dones():
            result=original_dones()
            if bool(result[0][slot] or result[1][slot]):
                terminal_capture['controls']=env._interval_controls()
                terminal_capture['flags']={k:v.tolist() for k,v in env.extras['unified'].items()}
            return result
        env._get_dones=capture_dones
        def visible():
            # Includes all per-env tensor buffers, all multibody states and phase-drive tensors.
            tensors={k:v.tolist() for k,v in vars(env).items() if torch.is_tensor(v) and v.ndim and v.shape[0]==5}
            return dict(tensors=tensors,body_state=env._robot.data.body_state_w.tolist(),
                root_state=env._robot.data.root_state_w.tolist(),joint_pos=env._robot.data.joint_pos.tolist(),
                joint_vel=env._robot.data.joint_vel.tolist(),observation=obs['policy'].tolist())
        baseline=args.baseline
        reference=[]
        if baseline:
            reference=[json.loads(line) for line in (baseline/'trace.jsonl').read_text().splitlines()]
            reference_start=json.loads((baseline/'start_state.json').read_text())
        trace=args.output/'trace.jsonl';terminated=None;obs_error=0.;prefix_maxima={}
        with torch.inference_mode(),trace.open('w') as stream:
            for step in range(1,args.start+61):
                before=rng_state()
                mu150=policies[150].act_inference(obs)
                # G25 inference is deterministic and does not alter G150 or environment.
                mu25=policies[25].act_inference(obs)
                require_rng(before)
                for g,u in [('A',25),('D',150)]:
                    assert torch.equal(select_actions(mu25,mu150,g,slot)[slot],{25:mu25,150:mu150}[u][slot])
                group='D' if step<=args.start else args.group
                actions=select_actions(mu25,mu150,group,slot)
                f,t=GROUPS[group]
                assert torch.equal(actions[slot,:1],{25:mu25,150:mu150}[f][slot,:1])
                assert torch.equal(actions[slot,1:],{25:mu25,150:mu150}[t][slot,1:])
                preobs=obs['policy'].tolist()
                new,_,done,timeout,_=env.step(actions);obs=tensor_obs(new,5)
                controls=terminal_capture.get('controls',env._interval_controls())
                entry=dict(schema=SCHEMA,step=step,time_s=step/60,group=group,source_F=f,source_T=t,
                    observations_before=preobs,mu25=mu25.tolist(),mu150=mu150.tolist(),controls=controls,
                    visible=visible(),targets_rad=torch.stack([env._rudder_cmd,env._left_elevon_cmd,env._right_elevon_cmd],-1).tolist(),
                    done=done.tolist(),timeout=timeout.tolist())
                stream.write(json.dumps(entry,allow_nan=False)+'\n');stream.flush()
                if step<=args.start or args.group=='D':
                    for i in range(5):
                        numeric_compare(controls[i],history[i][step-1]['control'],maxima,f'slot{i}')
                        if step>1:
                            reconstructed=observation(history[i][step-2]['control'],saved['configuration'],60.,ids)
                            err=float(np.max(abs(reconstructed-np.asarray(preobs[i]))));obs_error=max(obs_error,err)
                            assert err<=TOLERANCES['observation'],('observation',i,step,err)
                if baseline and step<=args.start:
                    # Stricter cross-process comparison: exact equality of all recorded visible buffers.
                    numeric_compare(entry,reference[step-1],prefix_maxima,exact=True)
                if step==args.start:
                    start=dict(step=step,controls=controls,visible=entry['visible'],targets_rad=entry['targets_rad'])
                    write(args.output/'start_state.json',start)
                    if baseline:numeric_compare(start,reference_start,prefix_maxima,exact=True)
                if bool(done[slot] or timeout[slot]):
                    terminated=dict(step=step,flags=terminal_capture.get('flags'));break
        for u,p in policies.items():
            assert policy_digest(p)==checkpoints[u]['policy_digest']
            assert all(torch.equal(v,checkpoints[u]['policy'][k]) for k,v in p.state_dict().items())
        write(args.output/'summary.json',dict(schema=SCHEMA,passed=True,group=args.group,start=args.start,
            completed_control_steps=step,terminated=terminated,historical_max_abs=maxima,
            historical_observation_max_abs=obs_error,common_prefix_max_abs=prefix_maxima,
            policies_unchanged=True,rng_unchanged_by_inference=True,optimizer_updates=0))
    finally:
        if env is not None:env.close()
        app.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--g25',type=Path,required=True);p.add_argument('--g150',type=Path,required=True)
    p.add_argument('--history',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plant-assets',type=Path,default=ROOT.parent/'control-oriented-3d-wing-aero')
    p.add_argument('--rules',type=Path,default=ROOT.parent/'control-oriented-3d-wing-aero/logs/unified_trajectory_phase2_v1/screening_rules_v2.json')
    p.add_argument('--group',choices=list(GROUPS));p.add_argument('--start',type=int,choices=[120,144]);p.add_argument('--baseline',type=Path)
    args=p.parse_args()
    for k in ('g25','g150','history','output','plant_assets','rules'):
        setattr(args,k,getattr(args,k).resolve())
    for path in (args.g25,args.g150,args.history/'evidence/rollout_feedback.sqlite3',args.rules):
        if not path.exists():raise FileNotFoundError(path)
    if any(args.output.is_relative_to(x) for x in (args.g25.parent,args.g150.parent,args.history)):
        raise ValueError('Output must be outside historical inputs')
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/'status.json',dict(status='running',schema=SCHEMA))
    try:
        if args.group:
            if args.start is None:raise ValueError('--start required')
            child(args)
        else:
            write(args.output/'protocol.json',dict(schema=SCHEMA,groups=GROUPS,start_steps=[120,144],window_steps=60,
                tolerances=TOLERANCES,order='D120,D144,then ABC120,ABC144; stop on any replay failure'))
            for start,group in [(120,'D'),(144,'D')]+[(s,g) for s in (120,144) for g in 'ABC']:
                output=args.output/f'T{start}_{group}'
                cmd=[str(ROOT/'isaaclab.sh'),'-p',str(Path(__file__).resolve()),'--g25',str(args.g25),'--g150',str(args.g150),
                    '--history',str(args.history),'--output',str(output),'--plant-assets',str(args.plant_assets),
                    '--rules',str(args.rules),'--start',str(start),'--group',group]
                if group!='D':cmd+=['--baseline',str(args.output/f'T{start}_D')]
                with (args.output/f'T{start}_{group}.log').open('w') as log:
                    result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                summary=output/'summary.json'
                if result.returncode or not summary.exists() or not json.loads(summary.read_text())['passed']:
                    raise RuntimeError(f'Replay/branch failed: T{start}_{group}; no further branches executed; see log')
                # Kit close may exit before child main returns; the parent owns completion.
                write(output/'status.json',dict(status='completed',schema=SCHEMA,child_wrapper_exit_code=result.returncode))
        write(args.output/'status.json',dict(status='completed',schema=SCHEMA))
    except BaseException:
        write(args.output/'status.json',dict(status='failed',schema=SCHEMA,error=traceback.format_exc()))
        raise

if __name__=='__main__':main()
