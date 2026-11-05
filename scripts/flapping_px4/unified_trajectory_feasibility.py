"""Offline L/I trajectory study; unchanged native plant, no RL or auto-reset."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--backend',choices=['L3D','I3D'],required=True)
p.add_argument('--cases',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--duration',type=float,default=20.)
p.add_argument('--sections',type=int,default=10)
AppLauncher.add_app_launcher_args(p)
a=p.parse_args(); a.device='cpu'
root=Path(__file__).resolve().parents[2]
cases=json.loads(a.cases.read_text())
if a.output.exists(): raise FileExistsError(a.output)
a.output.mkdir(parents=True)
point=root/'logs/control_aero_3d/periodic_trim_lateral_v1'/('L3D_late' if a.backend=='L3D' else 'I3D_refined')/'search.json'
x=np.array(json.loads(point.read_text())['best'])
a.kit_args=f'{a.kit_args or ""} --ext-folder {root}/source/flapping_bot/native_extensions --enable omni.flapping_bot.holonomic_constraint'
app=AppLauncher(a).app
try:
    import torch
    from isaaclab.utils.math import quat_apply, quat_apply_inverse
    from flapping_bot.direct.flapping_bot.straight_flight_env import FlappingBotStraightFlightEnv, FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg
    from flapping_bot.physics.control_aero_3d import ControlAero3DConfig
    from flapping_bot.analysis.pure_rl_fixed_action_reward_validation import _configure_robot_asset

    torch.set_num_threads(1)
    N = len(cases)
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

    from flapping_bot.path_tracking.unified_trajectory import Trajectory, TrajectorySpec
    paths=[Trajectory(TrajectorySpec(**c['trajectory'])) for c in cases]
    protocol=dict(backend=a.backend, initial=x.tolist(), initial_height_m=20., initial_vertical_speed=0.,
        seed=29, physics_dt=dt, control_dt=dt*cfg.decimation, sections=a.sections, twist_deg=35.,
        plant='CPU-native measured PhysX holonomic',wind='zero',duration_s=a.duration,
        tail_limits_rad=limits.tolist(),frequency_limits_hz=[cfg.min_flap_hz,cfg.max_flap_hz],
        frequency_slew_hz_s=cfg.frequency_governor_maximum_rise_rate_hz_per_s,
        joint_names=env._resolved_joint_names, cases=cases,
        acceptance=dict(horizontal_max_m=2.,vertical_max_m=2.,roll_max_deg=60.,pitch_max_deg=45.,
            omega_max_rad_s=8.,speed_min_mps=4.,speed_max_mps=16.,recovery_horizontal_m=1.,
            recovery_vertical_m=1.,recovery_vertical_speed_mps=.5, completion_fraction=.98),
        recovery='last 20 m of straight exit; final 1 s mean vertical velocity',
        no_speed_tracking=True,reference_point='base_link world z-up',
        status_semantics='finite mission only; failed search is NOT FOUND, never physical exclusion')
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    action=reset(np.repeat(x[None],N,axis=0))
    progress=np.zeros(N); filtered=None; rows=[]
    for k in range(math.ceil(a.duration/dt)+1):
        t=k*dt
        if k%cfg.decimation==0:
            row=sample(t)
            state=np.concatenate([row['euler'],row['omega'],row['velocity']],axis=1)
            filtered=state.copy() if filtered is None else filtered+(1-math.exp(-dt*cfg.decimation/.12))*(state-filtered)
            refs=[]; demand=[]; raw=[]
            for i,(c,path) in enumerate(zip(cases,paths)):
                progress[i]=path.project(row['position'][i],progress[i],3.)
                ref=path.query(progress[i]); refs.append(ref['position'])
                speed=np.linalg.norm(row['velocity'][i]); demand.append(speed**2*float(ref['curvature']))
                u=np.array([x[2],x[4],x[3]+x[5],x[3]-x[5]])
                gains=c.get('gains',{})
                if gains and t>=1.:
                    roll,pitch,yaw,pb,qb,rb,vx,vy,vz=filtered[i]
                    err=row['position'][i]-ref['position']
                    lateral=float(err@ref['normal'])
                    heading=float(ref['heading'])-np.clip(gains.get('cross_track',.12)*lateral,-.3,.3)
                    heading_error=math.atan2(math.sin(heading-(math.atan2(vy,vx) if gains.get('course',False) else yaw)),math.cos(heading-(math.atan2(vy,vx) if gains.get('course',False) else yaw)))
                    # z-up FLU: a positive heading turn needs negative bank.
                    bank=np.clip(-math.atan2(speed**2*float(ref['kappa'])*math.cos(float(ref['gamma'])),9.81)-gains.get('heading_bank',.6)*heading_error,-.65,.65)
                    rud=gains.get('yaw',8.)*heading_error-gains.get('r',3.)*(rb-speed*float(ref['kappa']))
                    diff=gains.get('roll',15.)*(bank-roll)-gains.get('p',4.)*pb
                    climb=float(ref['gamma'])-np.clip(.10*err[2],-.2,.2)
                    pitch_target=-(math.radians(4.31 if a.backend=='L3D' else 6.04)+climb)
                    sym=gains.get('pitch',12.)*(pitch_target-pitch)-gains.get('q',3.)*qb
                    u += [gains.get('height',.2)*(-err[2])+gains.get('vz',.3)*(speed*math.sin(float(ref['gamma']))-vz),rud,sym+diff,sym-diff]
                raw.append(u.copy())
                action[i,0]=float(2*np.clip(u[0],cfg.min_flap_hz,cfg.max_flap_hz)/cfg.max_flap_hz-1)
                for j in range(3):
                    lo,hi=limits[j]; action[i,j+1]=float(2*(np.clip(math.radians(u[j+1]),lo,hi)-lo)/(hi-lo)-1)
            row.update(progress=progress.copy(),reference=np.asarray(refs),normal_accel_demand=np.asarray(demand),
                requested_physical=np.asarray(raw), action=action.numpy().copy(),
                applied_action=env._act_cmd.numpy().copy(),governor_limited=env._frequency_governor_limited.numpy().copy())
            rows.append(row)
            if k%480==0: print('TIME',a.backend,round(t,2),'z',row['position'][:,2].round(2).tolist(),flush=True)
            if not np.isfinite(state).all(): raise RuntimeError('Nonfinite state')
            env._pre_physics_step(action)
        if k==math.ceil(a.duration/dt): break
        env._wind_w.zero_(); env._sim_step_counter+=1
        env._apply_action(); env.scene.write_data_to_sim(); env.sim.step(render=False); env.scene.update(dt=dt)
    np.savez_compressed(a.output/'trace.npz',**{key:np.stack([r[key] for r in rows]) for key in rows[0]})
    (a.output/'completed.json').write_text(json.dumps(dict(completed=True,steps=k,backend=a.backend)))
    env.close()
finally:
    app.close()
