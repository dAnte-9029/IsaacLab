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
from .unified_ppo_contract import (
    CANDIDATES, POLICY, PPO_CONFIG, evaluation_tasks, categories, source_identity,
    physical_source_identity, require_compatible_plant, EVALUATION_PROTOCOL, EVALUATION_LAYOUT,
    require_evaluation_checkpoint, validate_evaluation_batch, require_initial,
    SMOKE_SCHEMA, require_checkpoint_use, evaluation_origin)
from flapping_bot.path_tracking.unified_trajectory import TrajectorySpec
from flapping_bot.path_tracking.trajectory_feedback import FeedbackStore

SCHEMA='unified_li_ppo_v2'
from .unified_goal_reward import REWARD_CONTRACT


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
    cfg.reward_variant=getattr(args,'reward_variant','strict_goal'); cfg.goal_gamma=PPO_CONFIG['gamma']
    cfg.seed=args.seed; cfg.scene.num_envs=len(evaluation_tasks(args.eval_suite)) if evaluation else args.num_envs
    cfg.scene.env_spacing=10.; cfg.sim.device='cpu'
    cfg.online_sampling=not evaluation and args.sampling=='adaptive'; cfg.feedback_enabled=True
    cfg.online_rules_path=str(args.rules.resolve()); cfg.online_feedback_directory=str(args.output/'evidence')
    cfg.online_evaluation_mode='frozen' if evaluation else 'training'
    cfg.online_smoke_fixture=evaluation and getattr(args, 'smoke_evaluation', False)
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


def initialize_policy(policy, obs, env, args, root):
    """Fresh policy only; preserve all weights and center only the output bias."""
    import xml.etree.ElementTree as ET
    mode=getattr(args,'initialization','original')
    cfg=env.plant_identity['configuration']; initial=check_initial(env,args.backend)
    urdf=ET.parse(root/'source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf')
    limits={}
    for joint in urdf.findall('joint'):
        limit=joint.find('limit')
        if limit is not None:
            lo,hi=float(limit.attrib['lower']),float(limit.attrib['upper'])
            margin=(1-cfg['joint_limit_softness'])*(hi-lo)/2
            limits[joint.attrib['name']]=(lo+margin,hi-margin)
    trim=[2*(initial['frequency_hz']-cfg['min_flap_hz'])/(cfg['max_flap_hz']-cfg['min_flap_hz'])-1]
    angles=np.deg2rad([initial['rudder_deg'],initial['symmetric_deg']+initial['differential_deg'],initial['symmetric_deg']-initial['differential_deg']])
    for name,angle in zip(('rudder','left_tail','right_tail'),angles):
        lo,hi=limits[name]; trim.append(2*(angle-lo)/(hi-lo)-1)
    np.testing.assert_allclose(env._actions.numpy(),np.broadcast_to(trim,(env.num_envs,4)),atol=2e-7)
    before={k:v.clone() for k,v in policy.state_dict().items()}
    before_digest=policy_digest(policy)
    layer=[m for m in policy.actor.modules() if isinstance(m,torch.nn.Linear)][-1]
    bias_name=next(n for n,p in policy.named_parameters() if p is layer.bias)
    with torch.no_grad():
        old=policy.act_inference(obs).clone()
        offset=torch.tensor(trim,dtype=old.dtype)-old.mean(0)
        if mode!='original': layer.bias.add_(offset)
        else: offset.zero_()
        if mode=='trim_local': policy.std.fill_(0.15)
        new=policy.act_inference(obs).clone()
    changed=[k for k,v in policy.state_dict().items() if not torch.equal(v,before[k])]
    assert set(changed)<=({bias_name,'std'} if mode=='trim_local' else {bias_name} if mode=='trim_mean' else set())
    if mode!='original': torch.testing.assert_close(new.mean(0),torch.tensor(trim,dtype=new.dtype),atol=1e-7,rtol=0)
    result=dict(schema='trim_initialization_v1',mode=mode,trim=trim,initial_std=policy.std.detach().tolist(),
        bias_offset=offset.tolist(),seed=args.seed,source=source_identity(root),bias_parameter=bias_name,
        changed_parameters=changed,before_digest=before_digest,initial_policy_digest=policy_digest(policy),
        before_mean=old.mean(0).tolist(),after_mean=new.mean(0).tolist(),before_output_std=old.std(0).tolist(),after_output_std=new.std(0).tolist())
    torch.save(dict(observations=obs['policy'],before=before,after=policy.state_dict(),contract=result),args.output/'initialization.pt')
    (args.output/'initialization_contract.json').write_text(json.dumps(result,indent=2))
    return result


def validate_saved_policy(loaded, policy):
    if loaded.get('policy_digest') is not None and policy_digest(policy)!=loaded['policy_digest']:
        raise ValueError('Checkpoint policy digest mismatch')
    init=loaded.get('initialization_contract')
    if init is not None:
        if init['schema']!='trim_initialization_v1' or init['seed']!=loaded['contract']['seed']:
            raise ValueError('Initialization contract mismatch')
        if loaded['state']['updates']==0 and policy_digest(policy)!=init['initial_policy_digest']:
            raise ValueError('Update zero initialization mismatch')


def save_checkpoint(env,alg,args,state,contract,version,output,*,tag):
    target=output/f'model_{tag}.pt'; evidence=env.rollout_evaluator.store.path
    env.rollout_evaluator.store.flush()
    evidence_sequence=env.rollout_evaluator.store.db.execute('SELECT COALESCE(MAX(seq),0) FROM terminals').fetchone()[0]
    payload=dict(schema=SCHEMA,policy=alg.policy.state_dict(),optimizer=alg.optimizer.state_dict(),
        sampler=env.online_sampler.state_dict() if env.online_sampler else None,
        reset_counts=env.reset_counts.copy(),rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate()),
        state=deepcopy(state),contract=contract,plant=env.plant_identity,evaluation_contract=env.evaluation_contract,
        evaluation_tasks=evaluation_tasks(),evaluation_initial=env.evaluation_initial,version=version,evidence=str(evidence.resolve()),evidence_sequence=evidence_sequence,
        policy_config=POLICY,ppo_config=PPO_CONFIG,initialization_contract=getattr(args,'initialization_contract',None),policy_digest=policy_digest(alg.policy))
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
         '--reward-variant',args.reward_variant,'--checkpoint',str(checkpoint),'--output',str(destination),'--rules',str(args.rules.resolve()),
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
    if loaded is not None: require_checkpoint_use(loaded)
    contract=dict(schema=SCHEMA,reward_variant=args.reward_variant,reward=REWARD_CONTRACT,terminal_contract='unified_terminal_v2',evaluation_protocol=EVALUATION_PROTOCOL,evaluation_layout=EVALUATION_LAYOUT,
        backend=args.backend,sampling=args.sampling,seed=args.seed,num_envs=args.num_envs,
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
    state=deepcopy(loaded['state']) if args.resume else dict(updates=0,train_transitions=0,eval_env_steps=0,eval_active_steps=0,eval_reserved_env_steps=0,eval_padding_steps=0,evaluations=[])
    version=loaded['version'] if loaded else f'{run_id}:update000000'
    try:
        if loaded is not None:
            require_compatible_plant(env.plant_identity,loaded['plant'])
        if args.resume:
            require_compatible_plant(env.plant_identity,loaded['plant'])
            if env.evaluation_contract!=loaded['evaluation_contract']:
                raise ValueError('Resume physical/evaluation identity mismatch')
            env.reset_counts[:]=loaded['reset_counts']
            env.evaluation_initial=deepcopy(loaded['evaluation_initial'])
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
        args.initialization_contract=loaded.get('initialization_contract') if loaded else initialize_policy(policy,obs,env,args,root)
        if loaded:
            policy.load_state_dict(loaded['policy'])
            validate_saved_policy(loaded,policy)
            if args.resume:
                alg.optimizer.load_state_dict(loaded['optimizer'])
                torch.set_rng_state(loaded['rng']['torch']); np.random.set_state(loaded['rng']['numpy']); random.setstate(loaded['rng']['python'])
        if args.resume and loaded['state'].get('pending_evaluation'):
            save_checkpoint(env,alg,args,state,contract,version,args.output,tag=f'{state["updates"]:06d}_evaluated')
        initial_digest=policy_digest(policy)
        if not loaded: save_checkpoint(env,alg,args,state,contract,version,args.output,tag='000000')
        optimizer_step_before=max([float(x['step']) for x in alg.optimizer.state.values()],default=0.)
        (args.output/'contract.json').write_text(json.dumps(dict(training=contract,plant=env.plant_identity,
            termination=env.evaluation_contract,initial=initial,initial_policy_digest=initial_digest),indent=2))
        collect_logs(env,args.output)
        started=perf_counter(); train_seconds=0.; rollout_seconds=0.; update_seconds=0.
        transitions_at_start=state['train_transitions']; iteration_reports=[]; environment_step_s=0.
        for _ in range(args.iterations):
            env.set_policy_version(version,evaluation_mode='training')
            reward_components=[]
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
                    reward_components.append({k:float(v.mean()) for k,v in env.reward_terms.items()})
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
                reward_components={k:float(np.mean([r[k] for r in reward_components])) for k in reward_components[0]},
                policy_std=policy.std.detach().tolist(),parameter_delta_l2=delta,losses=losses,reward_mean=float(np.mean(rewards)),episode_outcomes=dict(endings),
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


def create_smoke_snapshot(env, args, root, obs):
    """Untrained initialization only: no PPO object, optimizer or training rollout."""
    torch.manual_seed(args.seed)
    policy = make_policy(tensor_obs(obs, env.num_envs))
    policy.requires_grad_(False); policy.eval()
    version = env._current_policy_version
    contract = dict(schema=SMOKE_SCHEMA, evaluation_protocol=EVALUATION_PROTOCOL,
        evaluation_layout=EVALUATION_LAYOUT, backend=args.backend, sampling=args.sampling,
        seed=args.seed, eval_step_budget=args.eval_step_budget, eval_tasks=evaluation_tasks(),
        final_tasks=evaluation_tasks('final'), policy=POLICY,
        purpose='evaluation-smoke-only; not trained; never resume or capability feedback')
    payload = dict(schema=SMOKE_SCHEMA, smoke_fixture=True, training_updates=0,
        policy=policy.state_dict(), policy_config=POLICY, contract=contract, version=version,
        plant=env.plant_identity, evaluation_contract=env.evaluation_contract,
        evaluation_initial=env.evaluation_initial, evaluation_tasks=evaluation_tasks(),
        source_provenance=source_identity(root), initialization='ActorCritic default initialization, fixed seed',
        initial_policy_digest=policy_digest(policy))
    args.checkpoint = args.output/'evaluation_smoke_only.pt'
    with args.checkpoint.open('xb') as stream: torch.save(payload, stream)
    (args.output/'snapshot_contract.json').write_text(json.dumps(
        {k:v for k,v in payload.items() if k!='policy'}, indent=2))
    return payload


def inspect_evaluation_geometry(env, tasks, output):
    """Compare real scene data and populated cache with independent analytic paths."""
    actual=env.scene.env_origins.detach().cpu().numpy()
    expected=np.array([evaluation_origin(len(tasks), i) for i in range(len(tasks))])
    layout=dict(actual=actual.tolist(), expected=expected.tolist(),
                cache_length_m=env.path_cache.length, cache_count=env.path_cache.count,
                episode_length_s=env.cfg.episode_length_s, max_steps=env.evaluation_contract['max_steps'])
    (output/'scene_layout.json').write_text(json.dumps(layout, indent=2))
    np.testing.assert_array_equal(actual, expected)
    length=tasks[0]['trajectory']['length_m']
    assert env.path_cache.length==length and env.online_sampler is None
    s=torch.linspace(0,length,17,dtype=torch.float64).expand(env.num_envs,-1)
    cached=env.path_cache.positions(s).cpu().numpy()
    analytic=np.stack([p.query(s[0].numpy())['position'] for p in env.paths])
    error=float(np.max(np.abs(cached-analytic)))
    assert error<2e-5, error
    probe=torch.full((env.num_envs,),length*.5,dtype=torch.float64)
    projected=env.path_cache.project(env.path_cache.query(probe)['position'],probe-.1,torch.full_like(probe,.5))
    projection_error=float((projected-probe).abs().max())
    assert projection_error<1e-3, projection_error
    layout.update(cache_analytic_max_error_m=error, projection_error_m=projection_error)
    (output/'scene_layout.json').write_text(json.dumps(layout, indent=2))
    return layout


def evaluate(args,root):
    smoke=getattr(args,'smoke_evaluation',False)
    create=getattr(args,'create_smoke_snapshot',False)
    if create and not smoke: raise ValueError('Snapshot creation requires explicit smoke evaluation')
    loaded=None if create else torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    tasks=evaluation_tasks(args.eval_suite)
    if loaded is not None:
        require_checkpoint_use(loaded, smoke_evaluation=smoke)
        if loaded.get('schema')!=(SMOKE_SCHEMA if smoke else SCHEMA): raise ValueError('Requires a Unified PPO checkpoint')
        if loaded['contract'].get('reward_variant') != args.reward_variant:
            raise ValueError('Evaluation reward variant mismatch')
        tasks = require_evaluation_checkpoint(loaded, backend=args.backend, sampling=args.sampling,
            suite=args.eval_suite, budget=args.eval_step_budget, smoke_evaluation=smoke)
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed)
    env=build_env(args,root,evaluation=True)
    try:
        if create:
            # No action or physics step: capture the actual reset plant and initial state.
            env.set_policy_version(f'smoke-only:{args.backend}:seed{args.seed}:{uuid.uuid4().hex[:12]}',
                                   evaluation_mode='frozen')
            obs,_=env.reset(); initial=check_initial(env,args.backend)
            layout=inspect_evaluation_geometry(env,tasks,args.output)
            loaded=create_smoke_snapshot(env,args,root,obs)
        require_compatible_plant(env.plant_identity,loaded['plant'])
        if env.evaluation_contract!=loaded['evaluation_contract']:
            (args.output/'identity_actual.json').write_text(json.dumps(env.plant_identity,indent=2))
            raise ValueError('Evaluation plant/contract differs from checkpoint')
        env.set_policy_version(loaded['version'],evaluation_mode='frozen')
        if not create:
            obs,_=env.reset(); initial=check_initial(env,args.backend)
            layout=inspect_evaluation_geometry(env,tasks,args.output)
        require_initial(env.evaluation_initial, loaded['evaluation_initial'])
        obs=tensor_obs(obs,env.num_envs); policy=make_policy(obs); policy.load_state_dict(loaded['policy']); policy.requires_grad_(False); policy.eval()
        validate_saved_policy(loaded,policy)
        before=policy_digest(policy); records=[]; steps=0; started=perf_counter()
        collect_logs(env,args.output)
        with torch.inference_mode():
            while steps < args.eval_step_budget:
                env.set_policy_version(loaded['version'],evaluation_mode='frozen')
                actions=policy.act_inference(obs)
                new,_,done,timeout,_=env.step(actions); obs=tensor_obs(new,env.num_envs); steps+=1
                _,terminal=collect_logs(env,args.output); records.extend(terminal)
                for record in terminal:
                    slot=record['initial']['scene']['env_index']; outcome=record['outcome']
                    assert bool(env.extras['unified'][outcome][slot])
                    assert bool(timeout[slot])==(outcome=='timeout')
                    assert bool(done[slot])==(outcome!='timeout')
                assert env.online_sampler is None
                if env._evaluation_finished.all():
                    break
                if steps>env.max_episode_length and not env._evaluation_finished.all():
                    raise RuntimeError('Evaluation failed to terminate under original contract')
        if not env._evaluation_finished.all():
            raise RuntimeError('Evaluation budget exhausted with incomplete episodes; no frozen capability claim')
        assert policy_digest(policy)==before and len(records)==env.num_envs
        assert all(r['policy_execution']=='single' and r['frozen_checkpoint']==loaded['version'] for r in records)
        records = validate_evaluation_batch(env.rollout_evaluator.store, tasks, env.plant_identity,
            env.evaluation_contract, loaded['evaluation_initial'], loaded['version'], smoke_evaluation=smoke)
        outcomes=Counter(r['outcome'] for r in records)
        return dict(passed=True,smoke_fixture=smoke,training_updates=0 if smoke else None,
            scene_layout=layout,batch_validated=True,terminal_masks_match=True,sampler_created=False,
            policy_digest_before=before,policy_digest_after=policy_digest(policy),
            sampling=args.sampling,eval_suite=args.eval_suite,trajectories=evaluation_tasks(args.eval_suite),checkpoint=str(args.checkpoint),version=loaded['version'],backend=args.backend,initial=initial,
            policy_unchanged=True,optimizer_updates=0,environment_steps=steps*env.num_envs,
            executed_environment_steps=steps*env.num_envs,reserved_step_budget=args.eval_step_budget,
            early_stopped=steps < args.eval_step_budget,
            active_episode_steps=sum(r['steps'] for r in records),
            padding_steps=steps*env.num_envs-sum(r['steps'] for r in records),eval_step_budget=args.eval_step_budget,wall_s=perf_counter()-started,
            completion_rate=outcomes['completed']/len(records),outcomes=dict(outcomes),
            tasks=[dict(name=tasks[r['initial']['scene']['env_index']]['name'],
                        categories=categories(TrajectorySpec(**r['trajectory'])),
                        **{k:r[k] for k in ('outcome','steps','effective_distance_m','horizontal_rmse_m','vertical_rmse_m',
                                           'horizontal_max_m','vertical_max_m','policy_execution','frozen_checkpoint')}) for r in records])
    finally:
        env.rollout_evaluator.store.close(); env.close()


def validate_evaluation_accounting(summary, records, budget):
    """Check actual compute against independently replayed episode lengths."""
    n = len(records)
    executed = summary.get('executed_environment_steps')
    active = sum(r['steps'] for r in records)
    last = max(r['steps'] for r in records)
    if (type(executed) is not int or executed != n*last or not 0 < last <= budget
            or summary.get('environment_steps') != executed
            or summary.get('reserved_step_budget') != budget
            or summary.get('eval_step_budget') != budget
            or summary.get('active_episode_steps') != active
            or summary.get('padding_steps') != executed-active
            or summary.get('early_stopped') is not (last < budget)):
        raise ValueError('Invalid frozen evaluation actual/reserved accounting')


def receive_evaluation(env,state,destination,summary,version,budget):
    if summary.get('smoke_fixture'): raise ValueError('Smoke evaluation cannot enter training feedback')
    if (not summary.get('passed') or summary.get('eval_suite')!='development'
            or summary.get('version')!=version):
        raise ValueError('Invalid development evaluation receipt/budget')
    if not state.get('pending_evaluation'): raise ValueError('No pending evaluation; duplicate batch rejected')
    if summary.get('trajectories')!=evaluation_tasks(): raise ValueError('Development task contract mismatch')
    store=FeedbackStore(destination/'evidence', readonly=True)
    try:
        records = validate_evaluation_batch(store, evaluation_tasks(), env.plant_identity,
            env.evaluation_contract, env.evaluation_initial, version)
        if (summary.get('active_episode_steps') != sum(r['steps'] for r in records)
                or summary.get('eval_step_budget') != budget
                or summary.get('policy_unchanged') is not True or summary.get('optimizer_updates') != 0):
            raise ValueError('Invalid frozen evaluation accounting')
        validate_evaluation_accounting(summary, records, budget)
        if env.online_sampler is not None:
            previous = env.online_sampler.state_dict()
            try:
                for record in records: env.online_sampler.feedback_evaluation(record,store)
                env.online_sampler._update()
            except BaseException:
                env.online_sampler.load_state_dict(previous)
                raise
    finally: store.close()
    state['eval_env_steps']+=summary['executed_environment_steps']
    state['eval_reserved_env_steps']=state.get('eval_reserved_env_steps',0)+len(records)*budget
    state['eval_padding_steps']=state.get('eval_padding_steps',0)+summary['padding_steps']
    state['eval_active_steps']+=summary['active_episode_steps']
    state['evaluations'].append(str(destination))
    state['pending_evaluation']=False


def complete_pending_evaluation(env,alg,args,root,state,contract,version,checkpoint):
    destination,summary=run_evaluation_subprocess(args,root,checkpoint,state['updates'])
    receive_evaluation(env,state,destination,summary,version,args.eval_step_budget)
    save_checkpoint(env,alg,args,state,contract,version,args.output,tag=f'{state["updates"]:06d}_evaluated')
    append_json(args.output/'evaluations.jsonl',dict(update=state['updates'],directory=str(destination),**summary))
