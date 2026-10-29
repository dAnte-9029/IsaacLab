"""Bounded native-PhysX open-loop search and paired lateral response; no policy.

All samples include startup. Sections interpolate unwrapped native drive phase,
not an assumed integer period. Positions are base-link positions (world FLU).
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument('--backend', choices=['L3D', 'I3D'], required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--search-cycles',type=int,default=3)
p.add_argument('--wind-small',type=float,default=.5)
p.add_argument('--tail-small',type=float,default=5.)
p.add_argument('--rounds', type=int, default=6)
p.add_argument('--sections', type=int, default=10)
p.add_argument('--compensations',type=Path,help='Two finite-horizon [rudder,differential] increments in degrees to verify')
p.add_argument('--response-point',type=Path,help='Evaluate only a saved best candidate, with complete integrated startup')
p.add_argument('--refine-from', type=Path, help='Search JSON for six-variable bounded local refinement including constant rudder/differential trim')
p.add_argument('--common-only', action='store_true', help='Matched initial-state response only, no search')
p.add_argument('--onset-cycles', type=float, default=1.)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.device = 'cpu'
if a.response_point: a.common_only=True
root = Path(__file__).resolve().parents[2]
if a.output.exists():
    raise FileExistsError(a.output)
a.output.mkdir(parents=True)
a.kit_args = f'{a.kit_args or ""} --ext-folder {root}/source/flapping_bot/native_extensions --enable omni.flapping_bot.holonomic_constraint'
app = AppLauncher(a).app
try:
    import torch
    from isaaclab.utils.math import quat_apply, quat_apply_inverse
    from flapping_bot.direct.flapping_bot.straight_flight_env import FlappingBotStraightFlightEnv, FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg
    from flapping_bot.physics.control_aero_3d import ControlAero3DConfig
    from flapping_bot.analysis.pure_rl_fixed_action_reward_validation import _configure_robot_asset

    torch.set_num_threads(1)
    N = 19 if a.compensations else 17
    cfg = FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg()
    cfg.seed = 29
    cfg.scene.num_envs = N
    cfg.scene.env_spacing = 30.
    cfg.sim.device = 'cpu'
    cfg.randomize_straight_line_heading = cfg.randomize_flap_phase_at_reset = False
    cfg.teacher_guidance_enabled = False
    cfg.freeze_steps_after_reset = 0
    cfg.wind_enabled = cfg.wind_ou_enabled = False
    cfg.height_cmd = 20.
    cfg.control_aero_3d = ControlAero3DConfig(backend=a.backend, twist_amplitude_deg=35., sections_per_wing=a.sections)
    cfg.delaurier_wing_geom_csv = root / 'outputs_DeLaurier/right_wing_te_fit_poly5_gap50.csv'
    _configure_robot_asset(cfg, asset_path=root / 'source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf', usd_dir=root / 'logs/control_aero_3d/assets')
    env = FlappingBotStraightFlightEnv(cfg)
    env.reset()
    dt = float(env.physics_dt)
    ids = env._joint_ids
    tailids = [env._IDX_RUDDER, env._IDX_LEFT_TAIL, env._IDX_RIGHT_TAIL]
    limits = np.array([[float(env._joint_lower_limits[i]), float(env._joint_upper_limits[i])] for i in tailids])
    protocol = dict(backend=a.backend, seed=29, dt=dt, decimation=cfg.decimation, sections=a.sections,
                    twist_deg=35, variables=['speed_mps','nose_up_pitch_deg','frequency_hz','symmetric_tail_deg'],
                    bounds=[[7,1,3,-20],[10,18,4,15]], initial_vertical_speed=0, initial_omega=[0,0,0],
                    tolerances=dict(displacement_yz_m=.05, velocity_mps=.15, attitude_deg=1, omega_rad_s=.15,
                                    frequency_hz=.01, wing_position_deg=.1, wing_velocity_rad_s=.05, tail_deg=.2),
                    sustained_cycles=8, evaluation_last_cycles=5, tail_limits_rad=limits.tolist(),
                    search=f'{0 if a.common_only else a.rounds} local coordinate batches, 9 candidates per batch, 3 integrated cycles; no aero fit',
                    onset_cycles=a.onset_cycles, wind_small_mps=a.wind_small, tail_small_deg=a.tail_small, joint_names=env._resolved_joint_names,
                    physics='CPU native measured holonomic; gravity on; no teacher, root hold, auto-reset, reward or policy')
    (a.output/'protocol.json').write_text(json.dumps(protocol, indent=2))

    def reset(xs):
        for i,x in enumerate(xs):
            cfg.reset_forward_speed_mps=float(x[0]); cfg.reset_pitch_deg=float(x[1])
            cfg.reset_flap_hz=float(x[2]); cfg.reset_elevon_pitch_deg=float(x[3])
            cfg.reset_rudder_deg=float(x[4]) if len(x)>4 else 0.
            cfg.reset_elevon_roll_deg=float(x[5]) if len(x)>5 else 0.
            env._reset_idx(torch.tensor([i]))
        env.scene.write_data_to_sim()
        env.scene.update(dt=dt)
        return env._actions.clone()

    def sample(t):
        d=env._robot.data
        q=d.root_quat_w
        # Keep raw quaternions; Euler angles use right-hand FLU signs, pitch negative nose-up.
        w,x,y,z=q.unbind(-1)
        euler=torch.stack((torch.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),torch.asin((2*(w*y-z*x)).clamp(-1,1)),torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))),-1)
        air=quat_apply_inverse(q,d.root_lin_vel_w-env._wind_w)
        jp=d.joint_pos[:,ids]; jv=d.joint_vel[:,ids]
        return dict(t=np.full(N,t), phase=env._phase.detach().numpy().copy(), frequency=env._freq.detach().numpy().copy(),
                    position=(d.root_pos_w-env.scene.env_origins).numpy().copy(), velocity=d.root_lin_vel_w.numpy().copy(),
                    body_velocity=d.root_lin_vel_b.numpy().copy(), omega=d.root_ang_vel_b.numpy().copy(),
                    quat=q.numpy().copy(), euler=euler.numpy().copy(), air=air.numpy().copy(),
                    beta=torch.atan2(air[:,1],air[:,0]).numpy().copy(), joint=jp.numpy().copy(), joint_velocity=jv.numpy().copy(),
                    wind=env._wind_w.numpy().copy())

    def run(xs,cycles=3,kind='search',name=None):
        xs=np.asarray(xs); xs=np.concatenate([xs,np.repeat(xs[:1],N-len(xs),axis=0)])
        action=reset(xs)
        # All perturbations start after one full integrated cycle; histories are identical before then.
        cases=['base','wind+.5','wind-.5','wind+2','p+.25','p-.25','r+.25','r-.25','rud+5','rud-5','diff+5','diff-5','wind+.5_diff+5','wind+.5_diff-5','wind+.5_rud+5','wind+.5_rud-5','duplicate']
        if a.compensations: cases+=['wind_comp_cycle','wind_comp_1s']
        onset=a.onset_cycles/xs[0,2]
        changed=False
        rows=[sample(0)]
        duration=cycles/min(xs[:,2])+.01 if kind=='search' or kind=='sustain' else onset+1.02
        for k in range(math.ceil(duration/dt)):
            t=k*dt
            if kind=='response' and not changed and t >= onset:
                changed=True; onset=t
                state=env._robot.data.root_state_w.clone()
                for i,axis,rate in [(4,0,.25),(5,0,-.25),(6,2,.25),(7,2,-.25)]:
                    dw=torch.zeros((1,3)); dw[0,axis]=rate
                    state[i,10:13]+=quat_apply(state[i:i+1,3:7],dw)[0]
                env._robot.write_root_velocity_to_sim(state[:,7:13])
                for i,rud,diff in [(8,5,0),(9,-5,0),(10,0,5),(11,0,-5),(12,0,5),(13,0,-5),(14,5,0),(15,-5,0)]:
                    rud*=a.tail_small/5; diff*=a.tail_small/5
                    for c,deg in enumerate([rud+(xs[i,4] if xs.shape[1]>4 else 0),xs[i,3]+diff+(xs[i,5] if xs.shape[1]>5 else 0),xs[i,3]-diff-(xs[i,5] if xs.shape[1]>5 else 0)]):
                        lo,hi=limits[c]; action[i,c+1]=2*(math.radians(deg)-lo)/(hi-lo)-1
                if a.compensations:
                    for i,(rud,diff) in enumerate(json.loads(a.compensations.read_text()),17):
                        for c,deg in enumerate([xs[i,4]+rud,xs[i,3]+xs[i,5]+diff,xs[i,3]-xs[i,5]-diff]):
                            lo,hi=limits[c]; action[i,c+1]=2*(math.radians(deg)-lo)/(hi-lo)-1
            if k%cfg.decimation==0:
                env._pre_physics_step(action)
            env._wind_w.zero_()
            if kind=='response' and changed:
                for i,v in [(1,.5),(2,-.5),(3,2),(12,.5),(13,.5),(14,.5),(15,.5)]: env._wind_w[i,1]=v*(a.wind_small/.5) if abs(v)==.5 else v
                if a.compensations: env._wind_w[17:,1]=a.wind_small
            env._sim_step_counter+=1
            env._apply_action(); env.scene.write_data_to_sim(); env.sim.step(render=False); env.scene.update(dt=dt)
            row=sample((k+1)*dt)
            row['force']=env._debug_last_force_b.numpy().copy()
            row['moment']=env._debug_last_torque_b.numpy().copy()
            rows.append(row)
        trace={key:np.stack([r[key] for r in rows]) for key in rows[0]}
        trace['force']=np.stack([rows[1]['force']]+[r['force'] for r in rows[1:]])
        trace['moment']=np.stack([rows[1]['moment']]+[r['moment'] for r in rows[1:]])
        trace['phase_unwrapped']=np.unwrap(trace['phase'],axis=0)
        if not all(np.isfinite(v).all() for v in trace.values()): raise RuntimeError('Nonfinite trajectory')
        if name:
            np.savez_compressed(a.output/(name+'.npz'),**trace, xs=xs, onset=onset, cases=np.array(cases),wind_small=a.wind_small,tail_small=a.tail_small)
        return trace,xs,onset

    from analyze_periodic_control_aero_3d import phase_sections as sections

    def metrics(tr,i):
        s=sections(tr,i); d={k:np.diff(v,axis=0) for k,v in s.items()}
        # Last two cycle residuals, plus vertical/horizontal drift. All axes penalized.
        residual=np.concatenate([d['velocity'][-2:].ravel()/.15,d['body_velocity'][-2:].ravel()/.15,
                                 d['euler'][-2:].ravel()/math.radians(1),d['omega'][-2:].ravel()/.15,
                                 d['position'][-2:,1:].ravel()/.05])
        return float(np.mean(residual**2)), {k:v.tolist() for k,v in s.items()}, {k:v.tolist() for k,v in d.items()}

    initial=np.array([8,5.837,3,.732] if a.backend=='L3D' else [8,9.302,3,-5.259])
    lo=np.array(protocol['bounds'][0]); hi=np.array(protocol['bounds'][1]); step=np.array([.5,1.5,.25,2.])
    if a.refine_from:
        prior=json.loads(a.refine_from.read_text())['best']
        initial=np.array(prior if len(prior)==6 else prior+[0.,0.])
        lo=np.r_[lo,-10.,-10.]; hi=np.r_[hi,10.,10.]; step=np.array([.25,.75,.125,1.,2.,2.])
        protocol['bounds']=[lo.tolist(),hi.tolist()]
        protocol['variables']+=['rudder_deg','differential_tail_deg']
        protocol['search']=f'{a.rounds} bounded finite-difference Gauss-Newton batches, 13 trajectories per batch, {a.search_cycles} cycles each'
        (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    best=initial; history=[]; global_score=float('inf'); global_best=best.copy()
    for iteration in range(0 if a.common_only else a.rounds):
        candidates=[best.copy()]
        for j in range(len(best)):
            for sign in [-1,1]:
                x=best.copy(); x[j]+=sign*step[j]; candidates.append(np.clip(x,lo,hi))
        tr,xs,_=run(candidates,cycles=a.search_cycles,name=f'search_{iteration:02d}')
        scores=[metrics(tr,i)[0] for i in range(len(candidates))]
        winner=int(np.argmin(scores))
        if min(scores)<global_score:
            global_score=min(scores);global_best=xs[winner].copy()
        if a.refine_from:
            from scipy.optimize import lsq_linear
            def res(i):
                sec=sections(tr,i); dd={k:np.diff(v,axis=0) for k,v in sec.items()}
                return np.concatenate([dd['velocity'][-2:].ravel()/.15,dd['body_velocity'][-2:].ravel()/.15,dd['euler'][-2:].ravel()/math.radians(1),dd['omega'][-2:].ravel()/.15,dd['position'][-2:,1:].ravel()/.05])
            jac=np.stack([(res(2*j+2)-res(2*j+1))/(xs[2*j+2,j]-xs[2*j+1,j]) for j in range(len(best))],axis=1)
            fit=lsq_linear(jac,-res(0),bounds=(np.maximum(lo-best,-step*2)-1e-9,np.minimum(hi-best,step*2)+1e-9))
            best=np.clip(best+fit.x,lo,hi)
        else:
            best=xs[winner].copy()
        history.append(dict(iteration=iteration, candidates=xs[:len(candidates)].tolist(),scores=scores,winner=winner))
        if winner==0: step*=.5
        print('SEARCH',a.backend,iteration,best.tolist(),min(scores),flush=True)
        (a.output/'search.json').write_text(json.dumps(dict(best=best.tolist(),history=history),indent=2))
    if a.refine_from:
        best=global_best
        (a.output/'search.json').write_text(json.dumps(dict(best=best.tolist(),history=history),indent=2))
    if not a.common_only:
        tr,_,_=run([best],cycles=8,kind='sustain',name='sustain_best')
        score,s,d=metrics(tr,0)
        (a.output/'sustain.json').write_text(json.dumps(dict(best=best.tolist(),score=score,sections=s,cycle_deltas=d),indent=2))
    points=[('common',np.array([8,9.302,3,-5.259]))] if a.common_only else [('own',best),('common',np.array([8,9.302,3,-5.259]))]
    if a.response_point:
        points=[('own',np.array(json.loads(a.response_point.read_text())['best']))]
    for label,x in points:
        tr,_,onset=run([x],kind='response',name='response_'+label)
        print('RESPONSE',a.backend,label,'onset',onset,flush=True)
    env.close()
finally:
    app.close()
