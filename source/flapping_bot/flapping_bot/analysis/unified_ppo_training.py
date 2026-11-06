"""Project-local orchestration of the existing RSL-RL PPO, independent of old curricula."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
from time import perf_counter
import uuid

import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic
from rsl_rl.algorithms import PPO
from .unified_ppo_contract import CANDIDATES,POLICY,PPO_CONFIG,evaluation_tasks,categories,source_identity,physical_source_identity,require_compatible_plant
from flapping_bot.path_tracking.unified_trajectory import TrajectorySpec
from flapping_bot.path_tracking.trajectory_feedback import FeedbackStore

SCHEMA='unified_li_ppo_v1'


def append_json(path, value):
    with Path(path).open('a') as stream: stream.write(json.dumps(value,allow_nan=False)+'\n')


def policy_digest(policy):
    digest=hashlib.sha256()
    for p in policy.parameters(): digest.update(p.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def build_env(args, root, *, evaluation=False):
    from flapping_bot.direct.flapping_bot.unified_li_cfg import UnifiedTrajectoryL3DEnvCfg,UnifiedTrajectoryI3DEnvCfg
    from flapping_bot.direct.flapping_bot.unified_trajectory_env import UnifiedTrajectoryEnv
    from flapping_bot.analysis.pure_rl_fixed_action_reward_validation import _configure_robot_asset
    cfg=(UnifiedTrajectoryL3DEnvCfg if args.backend=='L3D' else UnifiedTrajectoryI3DEnvCfg)()
    cfg.seed=args.seed; cfg.scene.num_envs=len(evaluation_tasks(args.eval_suite)) if evaluation else args.num_envs
    cfg.scene.env_spacing=10.; cfg.sim.device='cpu'
    cfg.online_sampling=not evaluation and args.sampling=='adaptive'; cfg.feedback_enabled=True
    cfg.online_rules_path=str(args.rules.resolve()); cfg.online_feedback_directory=str(args.output/'evidence')
    cfg.online_evaluation_mode='frozen' if evaluation else 'training'
    cfg.evaluation_tasks=evaluation_tasks(args.eval_suite) if evaluation else None
    if evaluation: cfg.sampling=replace(cfg.sampling,length_m=cfg.evaluation_tasks[0]["trajectory"]["length_m"])
    cfg.delaurier_wing_geom_csv=args.plant_assets.resolve()/'outputs_DeLaurier/right_wing_te_fit_poly5_gap50.csv'
    urdf=root/'source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf'
    _configure_robot_asset(cfg,asset_path=urdf,usd_dir=args.output/'assets')
    cfg.online_identity_metadata=physical_source_identity(root,args.plant_assets)
    for name in cfg.online_identity_metadata['physical_sources']:
        path=(args.plant_assets/'source/flapping_bot/native_extensions'/name.removeprefix('external_native/')) if name.startswith('external_native/') else root/name
        target=args.output/'source_snapshot'/'physical'/name
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(path,target)
    cfg.online_identity_metadata.update(urdf=str(urdf),urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),
        candidate_initial=list(CANDIDATES[args.backend]),initial_phase_rad=0.,initial_vertical_speed=0.,
        wing_geometry_sha256=hashlib.sha256(cfg.delaurier_wing_geom_csv.read_bytes()).hexdigest(),
        candidate_source='periodic_trim_lateral_v1/'+('L3D_late' if args.backend=='L3D' else 'I3D_refined')+'/search.json')
    import gymnasium as gym
    return gym.make(f'Isaac-FlappingBot-UnifiedTrajectory-{args.backend}-Direct-v0',cfg=cfg).unwrapped


def check_initial(env, backend):
    v,p,f,sym,rud,diff=CANDIDATES[backend]; d=env._robot.data
    assert env._control_aero_model is not None and env.cfg.control_aero_3d.backend==backend
    assert env.cfg.observation_space==39 and env._actions.shape[1]==4
    np.testing.assert_allclose(d.root_lin_vel_w[:,0].numpy(),v,atol=2e-6)
    np.testing.assert_allclose(d.root_lin_vel_w[:,1:].numpy(),0.,atol=2e-6)
    np.testing.assert_allclose(env._phase.numpy(),0.,atol=1e-6)
    np.testing.assert_allclose(env._freq.numpy(),f,atol=1e-6)
    expected=np.array([np.cos(np.deg2rad(p)/2),0.,-np.sin(np.deg2rad(p)/2),0.])
    np.testing.assert_allclose(d.root_quat_w.numpy(),np.broadcast_to(expected,d.root_quat_w.shape),atol=1e-6)
    ids=[int(env._joint_ids[j]) for j in (env._IDX_RUDDER,env._IDX_LEFT_TAIL,env._IDX_RIGHT_TAIL)]
    expected=np.deg2rad([rud,sym+diff,sym-diff])
    np.testing.assert_allclose(d.joint_pos[:,ids].numpy(),np.broadcast_to(expected,(env.num_envs,3)),atol=1e-6)
    return dict(speed_mps=v,nose_up_pitch_deg=p,frequency_hz=f,symmetric_deg=sym,rudder_deg=rud,differential_deg=diff,phase_rad=0.)


def tensor_obs(obs, n): return TensorDict(obs,batch_size=[n])


def collect_logs(env,output):
    tasks=env.drain_tasks(); records=env.drain_terminals()
    for t in tasks:
        t['geometry_categories']=categories(TrajectorySpec(**t['trajectory']))
        append_json(output/'tasks.jsonl',t)
    for r in records:
        append_json(output/'episodes.jsonl',{k:v for k,v in r.items() if k not in ('plant','contract','initial','identity')})
    return tasks,records


def make_policy(obs):
    return ActorCritic(obs,{'policy':['policy'],'critic':['policy']},4,**POLICY)


def save_checkpoint(env,alg,args,state,contract,version,output,*,tag):
    target=output/f'model_{tag}.pt'; evidence=env.rollout_evaluator.store.path
    env.rollout_evaluator.store.flush()
    evidence_sequence=env.rollout_evaluator.store.db.execute('SELECT COALESCE(MAX(seq),0) FROM terminals').fetchone()[0]
    payload=dict(schema=SCHEMA,policy=alg.policy.state_dict(),optimizer=alg.optimizer.state_dict(),
        sampler=env.online_sampler.state_dict() if env.online_sampler else None,
        reset_counts=env.reset_counts.copy(),rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate()),
        state=deepcopy(state),contract=contract,plant=env.plant_identity,evaluation_contract=env.evaluation_contract,
        evaluation_tasks=evaluation_tasks(),version=version,evidence=str(evidence.resolve()),evidence_sequence=evidence_sequence,
        policy_config=POLICY,ppo_config=PPO_CONFIG)
    temporary=target.with_suffix('.pt.tmp')
    payload['source_provenance']=source_identity(Path(__file__).resolve().parents[4])
    torch.save(payload,temporary)
    os.replace(temporary,target)
    latest=output/'latest_checkpoint.txt.tmp'
    latest.write_text(str(target.resolve())+'\n')
    os.replace(latest,output/'latest_checkpoint.txt')
    return target


def run_evaluation_subprocess(args,root,checkpoint,iteration):
    destination=args.output/f'evaluation_{iteration:06d}'
    cmd=['bash',str(root/'isaaclab.sh'),'-p',str(root/'scripts/flapping_px4/train_unified_trajectory.py'),
         '--evaluate','--backend',args.backend,'--sampling',args.sampling,'--seed',str(args.seed),
         '--checkpoint',str(checkpoint),'--output',str(destination),'--rules',str(args.rules.resolve()),
         '--plant-assets',str(args.plant_assets.resolve()),'--threads',str(args.threads),
         '--eval-step-budget',str(args.eval_step_budget)]
    with (args.output/f'evaluation_{iteration:06d}.log').open('w') as log:
        result=subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT)
    # Repository wrapper can mask exit code; completed status and summary are mandatory.
    if result.returncode or not (destination/'summary.json').exists(): raise RuntimeError('Frozen evaluation failed: '+str(destination))
    summary=json.loads((destination/'summary.json').read_text())
    if not summary.get('passed'): raise RuntimeError('Frozen evaluation did not complete')
    return destination,summary


def train(args,root):
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    loaded=torch.load(args.checkpoint,map_location='cpu',weights_only=False) if args.checkpoint else None
    contract=dict(schema=SCHEMA,backend=args.backend,sampling=args.sampling,seed=args.seed,num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,ppo=PPO_CONFIG,policy=POLICY,eval_interval=args.eval_interval,
        sampling_sources={name:hashlib.sha256((root/'source/flapping_bot/flapping_bot/path_tracking'/name).read_bytes()).hexdigest() for name in ('unified_trajectory.py','online_trajectory.py','trajectory_capability.py')},eval_tasks=evaluation_tasks(),final_tasks=evaluation_tasks('final'),task_source=hashlib.sha256((root/'source/flapping_bot/flapping_bot/direct/flapping_bot/unified_trajectory_env.py').read_bytes()).hexdigest(),threads=args.threads,eval_step_budget=args.eval_step_budget)
    if loaded is not None:
        if loaded.get('schema')!=SCHEMA or loaded['policy_config']!=POLICY: raise ValueError('Requires compatible 39D Unified checkpoint')
        def semantic(c):
            c=deepcopy(c); c['eval_tasks']=[t['trajectory'] for t in c['eval_tasks']]
            return c
        if args.resume and semantic(loaded['contract'])!=semantic(contract):
            raise ValueError('Resume training contract mismatch')
        if args.resume:
            FeedbackStore.restore_prefix(loaded['evidence'],args.output/'evidence',loaded.get('evidence_sequence'))
    env=build_env(args,root)
    run_id=uuid.uuid4().hex[:12]
    state=deepcopy(loaded['state']) if args.resume else dict(updates=0,train_transitions=0,eval_env_steps=0,eval_active_steps=0,evaluations=[])
    version=loaded['version'] if loaded else f'{run_id}:update000000'
    try:
        if loaded is not None:
            require_compatible_plant(env.plant_identity,loaded['plant'])
        if args.resume:
            require_compatible_plant(env.plant_identity,loaded['plant'])
            if env.evaluation_contract!=loaded['evaluation_contract']:
                raise ValueError('Resume physical/evaluation identity mismatch')
            env.reset_counts[:]=loaded['reset_counts']
            if env.online_sampler is not None: env.online_sampler.load_state_dict(loaded['sampler'])
        if args.resume and state.get('pending_evaluation'):
            destination,ev=run_evaluation_subprocess(args,root,args.checkpoint,state['updates'])
            receive_evaluation(env,state,destination,ev,version,args.eval_step_budget)
        env.set_policy_version(version)
        obs,_=env.reset(); initial=check_initial(env,args.backend)
        # Explicit partial reset wiring check only in short smoke; both modes get it.
        if args.smoke_check and env.num_envs>1:
            protected=deepcopy(env.task_records[1]); ident=env.rollout_evaluator.active[1]['record']['instance_id']
            env._reset_idx(torch.tensor([0])); assert env.task_records[1]==protected
            assert env.rollout_evaluator.active[1]['record']['instance_id']==ident
            obs=env._get_observations()
        obs=tensor_obs(obs,env.num_envs)
        torch.manual_seed(args.seed)  # same network initialization independently of environment setup/sampler
        policy=make_policy(obs); alg=PPO(policy,device='cpu',**PPO_CONFIG)
        alg.init_storage('rl',env.num_envs,args.rollout_steps,obs,[4])
        if loaded:
            policy.load_state_dict(loaded['policy'])
            if args.resume:
                alg.optimizer.load_state_dict(loaded['optimizer'])
                torch.set_rng_state(loaded['rng']['torch']); np.random.set_state(loaded['rng']['numpy']); random.setstate(loaded['rng']['python'])
        if args.resume and loaded['state'].get('pending_evaluation'):
            save_checkpoint(env,alg,args,state,contract,version,args.output,tag=f'{state["updates"]:06d}_evaluated')
        initial_digest=policy_digest(policy)
        optimizer_step_before=max([float(x['step']) for x in alg.optimizer.state.values()],default=0.)
        (args.output/'contract.json').write_text(json.dumps(dict(training=contract,plant=env.plant_identity,
            termination=env.evaluation_contract,initial=initial,initial_policy_digest=initial_digest),indent=2))
        collect_logs(env,args.output)
        started=perf_counter(); train_seconds=0.; rollout_seconds=0.; update_seconds=0.
        transitions_at_start=state['train_transitions']; iteration_reports=[]; environment_step_s=0.
        for _ in range(args.iterations):
            env.set_policy_version(version,evaluation_mode='training')
            t=perf_counter(); policy.train(); sources=Counter(); geometry=Counter(); endings=Counter(); rewards=[]
            with torch.inference_mode():
                for step in range(args.rollout_steps):
                    actions=alg.act(obs)
                    step_started=perf_counter()
                    next_obs,reward,done,timeout,_=env.step(actions)
                    environment_step_s+=perf_counter()-step_started
                    obs=tensor_obs(next_obs,env.num_envs)
                    assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
                    alg.process_env_step(obs,reward,done|timeout,{'time_outs':timeout})
                    rewards.append(float(reward.mean()))
                    tasks,records=collect_logs(env,args.output)
                    for task in tasks:
                        sources[task['source']]+=1; geometry.update(k for k,v in task['geometry_categories'].items() if v)
                    endings.update(r['outcome'] for r in records)
                alg.compute_returns(obs)
            rollout_time=perf_counter()-t; t=perf_counter()
            before=[p.detach().clone() for p in policy.parameters()]; losses=alg.update()
            delta=float(torch.sqrt(sum(torch.sum((p-q)**2) for p,q in zip(policy.parameters(),before))).detach())
            if not np.isfinite(list(losses.values())).all() or not np.isfinite(delta) or delta<=0: raise RuntimeError('PPO failed to update finitely')
            update_time=perf_counter()-t
            state['updates']+=1; state['train_transitions']+=env.num_envs*args.rollout_steps
            version=f'{run_id}:update{state["updates"]:06d}'
            train_seconds+=rollout_time+update_time; rollout_seconds+=rollout_time; update_seconds+=update_time
            report=dict(update=state['updates'],version=version,train_transitions=state['train_transitions'],
                rollout_s=rollout_time,update_s=update_time,transitions_per_s=env.num_envs*args.rollout_steps/(rollout_time+update_time),
                parameter_delta_l2=delta,losses=losses,reward_mean=float(np.mean(rewards)),episode_outcomes=dict(endings),
                reset_sources=dict(sources),reset_geometry_counts=dict(geometry),
                effective_completion_rate=endings['completed']/sum(endings[k] for k in ('completed','failed','timeout'))
                    if sum(endings[k] for k in ('completed','failed','timeout')) else None)
            iteration_reports.append(report); append_json(args.output/'training.jsonl',report)
            print('PPO_UPDATE',json.dumps(report),flush=True)
            should_eval=args.eval_interval>0 and state['updates']%args.eval_interval==0
            if should_eval or state['updates']%args.save_interval==0:
                state['pending_evaluation']=bool(should_eval)
                ckpt=save_checkpoint(env,alg,args,state,contract,version,args.output,tag=f'{state["updates"]:06d}')
                if should_eval:
                    complete_pending_evaluation(env,alg,args,root,state,contract,version,ckpt)
        # A clean stop is an explicit manual reset boundary, not a trajectory outcome.
        for i in list(env.rollout_evaluator.active): env.accept_terminal(env.rollout_evaluator.abort(i))
        collect_logs(env,args.output)
        final=save_checkpoint(env,alg,args,state,contract,version,args.output,tag=f'{state["updates"]:06d}_final')
        optimizer_step_after=max(float(x['step']) for x in alg.optimizer.state.values())
        report=dict(passed=True,backend=args.backend,sampling=args.sampling,checkpoint=str(final),initial=initial,
            initial_policy_digest=initial_digest,final_policy_digest=policy_digest(policy),
            optimizer_step_before=optimizer_step_before,optimizer_step_after=optimizer_step_after,
            state=state,local_train_transitions=state['train_transitions']-transitions_at_start,
            train_seconds=train_seconds,rollout_seconds=rollout_seconds,update_seconds=update_seconds,
            measured_training_transitions_per_s=(state['train_transitions']-transitions_at_start)/train_seconds,
            loop_wall_s=perf_counter()-started,resume=args.resume,iterations=iteration_reports,
            environment_step_s=environment_step_s,sqlite_flush_s=env.rollout_evaluator.store.flush_seconds,
            sqlite_flush_calls=env.rollout_evaluator.store.flush_calls,
            sqlite_bytes=env.rollout_evaluator.store.path.stat().st_size,
            physical_resume_semantics='fresh episodes; optimizer/RNG/reset sequence/sampler/evidence restored')
        # Check saved optimizer/model state is loadable before declaring success.
        restored=torch.load(final,map_location='cpu',weights_only=False)
        assert all(torch.equal(v,restored['policy'][k]) for k,v in policy.state_dict().items())
        assert restored['optimizer']['state']
        return report
    finally:
        env.rollout_evaluator.store.close(); env.close()


def evaluate(args,root):
    loaded=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if loaded.get('schema')!=SCHEMA: raise ValueError('Requires a Unified PPO checkpoint')
    if loaded['contract']['backend']!=args.backend: raise ValueError('Checkpoint backend mismatch')
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed)
    env=build_env(args,root,evaluation=True)
    try:
        require_compatible_plant(env.plant_identity,loaded['plant'])
        if env.evaluation_contract!=loaded['evaluation_contract']:
            (args.output/'identity_actual.json').write_text(json.dumps(env.plant_identity,indent=2))
            raise ValueError('Evaluation plant/contract differs from checkpoint')
        env.set_policy_version(loaded['version'],evaluation_mode='frozen')
        obs,_=env.reset(); initial=check_initial(env,args.backend)
        obs=tensor_obs(obs,env.num_envs); policy=make_policy(obs); policy.load_state_dict(loaded['policy']); policy.eval()
        before=policy_digest(policy); records=[]; steps=0; started=perf_counter()
        collect_logs(env,args.output)
        with torch.inference_mode():
            while steps < args.eval_step_budget:
                env.set_policy_version(loaded['version'],evaluation_mode='frozen')
                actions=policy.act_inference(obs)
                new,_,_,_,_=env.step(actions); obs=tensor_obs(new,env.num_envs); steps+=1
                _,terminal=collect_logs(env,args.output); records.extend(terminal)
                if steps>env.max_episode_length and not env._evaluation_finished.all():
                    raise RuntimeError('Evaluation failed to terminate under original contract')
        if not env._evaluation_finished.all():
            raise RuntimeError('Evaluation budget exhausted with incomplete episodes; no frozen capability claim')
        assert policy_digest(policy)==before and len(records)==env.num_envs
        assert all(r['policy_execution']=='single' and r['frozen_checkpoint']==loaded['version'] for r in records)
        outcomes=Counter(r['outcome'] for r in records)
        return dict(passed=True,eval_suite=args.eval_suite,trajectories=evaluation_tasks(args.eval_suite),checkpoint=str(args.checkpoint),version=loaded['version'],backend=args.backend,initial=initial,
            policy_unchanged=True,optimizer_updates=0,environment_steps=steps*env.num_envs,
            active_episode_steps=sum(r['steps'] for r in records),
            padding_steps=steps*env.num_envs-sum(r['steps'] for r in records),eval_step_budget=args.eval_step_budget,wall_s=perf_counter()-started,
            completion_rate=outcomes['completed']/len(records),outcomes=dict(outcomes),
            tasks=[dict(name=next(t['name'] for t in evaluation_tasks(args.eval_suite) if t['trajectory']['seed']==r['trajectory']['seed']),
                        categories=categories(TrajectorySpec(**r['trajectory'])),
                        **{k:r[k] for k in ('outcome','steps','effective_distance_m','horizontal_rmse_m','vertical_rmse_m',
                                           'horizontal_max_m','vertical_max_m','policy_execution','frozen_checkpoint')}) for r in records])
    finally:
        env.rollout_evaluator.store.close(); env.close()


def receive_evaluation(env,state,destination,summary,version,budget):
    if (not summary.get('passed') or summary.get('eval_suite')!='development'
            or summary.get('version')!=version or summary.get('environment_steps')!=5*budget):
        raise ValueError('Invalid development evaluation receipt/budget')
    if not state.get('pending_evaluation'): raise ValueError('No pending evaluation; duplicate batch rejected')
    if summary.get('trajectories')!=evaluation_tasks(): raise ValueError('Development task contract mismatch')
    store=FeedbackStore(destination/'evidence')
    try:
        records=list(store.iter_records())
        if len(records)!=5 or any(r.get('frozen_checkpoint')!=version or r.get('policy_execution')!='single'
                or r['plant']!=env.plant_identity or r['contract']!=env.evaluation_contract for r in records):
            raise ValueError('Invalid frozen evidence batch')
        if env.online_sampler is not None:
            for record in records: env.online_sampler.feedback_evaluation(record,store)
            env.online_sampler._update()
    finally: store.close()
    state['eval_env_steps']+=summary['environment_steps']
    state['eval_active_steps']+=summary['active_episode_steps']
    state['evaluations'].append(str(destination))
    state['pending_evaluation']=False


def complete_pending_evaluation(env,alg,args,root,state,contract,version,checkpoint):
    destination,summary=run_evaluation_subprocess(args,root,checkpoint,state['updates'])
    receive_evaluation(env,state,destination,summary,version,args.eval_step_budget)
    save_checkpoint(env,alg,args,state,contract,version,args.output,tag=f'{state["updates"]:06d}_evaluated')
    append_json(args.output/'evaluations.jsonl',dict(update=state['updates'],directory=str(destination),**summary))
