"""48-step deterministic CPU PhysX replay; no PPO, backward, or policy checkpoint."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--backend',choices=['L3D','I3D'],required=True)
p.add_argument('--implementation-root',type=Path,default=ROOT)
p.add_argument('--physical-root',type=Path,default=ROOT)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--reference-root',type=Path,default=ROOT.parent/'unified-rl-performance-phase2')
p.add_argument('--profile',action='store_true',help='Measure detailed instrumentation overhead on identical deterministic replay')
a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(a.implementation_root.resolve()/'source/flapping_bot'))
from isaaclab.app import AppLauncher
assets=ROOT.parent/'control-oriented-3d-wing-aero'
app=AppLauncher(headless=True,device='cpu',kit_args=f'--ext-folder {assets}/source/flapping_bot/native_extensions --enable omni.flapping_bot.holonomic_constraint').app
import torch
import numpy as np
from flapping_bot.analysis.unified_ppo_training import check_initial
import flapping_bot.analysis.unified_ppo_training as runner
import benchmark_unified_performance as harness
harness.BENCH_ARGS=SimpleNamespace(evaluation_root=None)
timers=harness.Timers(a.profile)
if a.profile: harness.install(timers,a.output,detail=True)
args=SimpleNamespace(backend=a.backend,seed=0,num_envs=256,sampling='fixed',eval_suite='development',
    output=a.output,rules=assets/'logs/unified_trajectory_phase2_v1/screening_rules_v2.json',plant_assets=assets)
torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0)
env=runner.build_env(args,a.physical_root.resolve())
try:
    env.set_policy_version('deterministic-phase2-replay');obs,_=env.reset();check_initial(env,a.backend)
    physics=[];intervals=[];action_rows=[]
    # Same-state shadow evaluation is validation only, never performance timing.
    import importlib.util
    import flapping_bot.physics.control_aero_3d as aero
    spec=importlib.util.spec_from_file_location('flapping_bot.physics._phase2_shadow',
        a.reference_root/'source/flapping_bot/flapping_bot/physics/control_aero_3d.py')
    refmod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=refmod;spec.loader.exec_module(refmod)
    refmodel=refmod.ControlAero3D(env._control_aero_model.config)
    stats={}
    def exact(name,x,y):
        row=stats.setdefault(name,dict(calls=0,elements=0,max_abs=0.,max_rel=0.,unequal=0))
        row['calls']+=1;row['elements']+=x.numel()
        delta=(x-y).abs()
        row['max_abs']=max(row['max_abs'],float(delta.max()))
        row['max_rel']=max(row['max_rel'],float((delta/y.abs().clamp_min(torch.finfo(y.dtype).tiny)).max()))
        row['unequal']+=int((x!=y).sum())
        if not torch.equal(x,y): raise AssertionError(f'Exact parity failed: {name}: {row}')
    influence=aero.horseshoe_influence
    def influence_checked(*args,**kwargs):
        actual=influence(*args,**kwargs); expected=refmod.horseshoe_influence(*args,**kwargs)
        for name,x,y in zip(('influence','bound'),actual,expected):exact(name,x,y)
        return actual
    aero.horseshoe_influence=influence_checked
    evaluate=env._control_aero_model.evaluate
    def evaluate_checked(*args,**kwargs):
        actual=evaluate(*args,**kwargs);expected=refmodel.evaluate(*args,**kwargs)
        for name in vars(actual):exact('loads.'+name,getattr(actual,name),getattr(expected,name))
        return actual
    env._control_aero_model.evaluate=evaluate_checked
    from flapping_bot.physics.control_aero_kinematics import LinkKinematics3D
    kin_spec=importlib.util.spec_from_file_location('flapping_bot.physics._phase2_kinematics',
        a.reference_root/'source/flapping_bot/flapping_bot/physics/control_aero_kinematics.py')
    kinmod=importlib.util.module_from_spec(kin_spec);sys.modules[kin_spec.name]=kinmod;kin_spec.loader.exec_module(kinmod)
    sample=env._control_aero_kinematics.sample
    def sample_checked(*args,**kwargs):
        actual=sample(*args,**kwargs)
        expected=kinmod.LinkKinematics3D.sample(env._control_aero_kinematics,*args,**kwargs)
        for group,x,y in zip(('geometry','motion','relative'),actual,expected):
            for name in vars(x):exact(group+'.'+name,getattr(x,name),getattr(y,name))
        return actual
    env._control_aero_kinematics.sample=sample_checked
    apply=env._apply_action
    def capture():
        apply()
        d=env._robot.data
        physics.append(torch.cat([env._debug_last_wing_force_b,env._debug_last_wing_moment_b_about_base_com_nm,
            env._debug_last_force_b,env._debug_last_torque_b,d.joint_pos,d.joint_vel,d.root_state_w],-1).clone().numpy())
    env._apply_action=capture
    step_seconds=0.
    with torch.inference_mode():
        for step in range(48):
            actions=.35*torch.sin(torch.arange(256*4,dtype=torch.float32).reshape(256,4)*.013+step*.17)
            action_rows.append(actions.clone().numpy())
            started=perf_counter()
            obs,reward,done,timeout,_=env.step(actions)
            step_seconds+=perf_counter()-started
            intervals.append(torch.cat([obs['policy'],reward[:,None],done[:,None],timeout[:,None],
                env.progress[:,None],env.reference['error']],-1).clone().numpy())
    (a.output/'internal_parity.json').write_text(json.dumps(dict(passed=True,statistics=stats,error_distribution='All differences exactly zero; all quantiles zero.'),indent=2))
    timers.export(a.output)
    np.savez(a.output/'replay.npz',actions=np.stack(action_rows),physics=np.stack(physics),intervals=np.stack(intervals))
    (a.output/'identity.json').write_text(json.dumps(env.plant_identity,indent=2))
    for i in list(env.rollout_evaluator.active): env.rollout_evaluator.abort(i)
    (a.output/'completion.json').write_text(json.dumps(dict(passed=True,steps=48,physics_steps=len(physics),num_envs=256,ppo_updates=0,env_step_s=step_seconds,profile=a.profile)))
finally:
    env.rollout_evaluator.store.close();env.close();app.close()
