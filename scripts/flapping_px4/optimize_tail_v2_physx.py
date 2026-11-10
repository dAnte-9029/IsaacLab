"""Bounded nested Tail V2 optimization on actual CPU native PhysX; no policy."""
import argparse,json,math,time,hashlib
from pathlib import Path
from dataclasses import asdict
import numpy as np
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--evidence-dir',type=Path)
AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.device='cpu';a.backend='L3D';a.sections=10
root=Path(__file__).resolve().parents[2];evidence=a.evidence_dir.resolve() if a.evidence_dir else root/'docs/results/tail_v2_control_v2'
protocol=json.loads((evidence/'protocol.json').read_text())
if a.output.exists() and not a.resume:raise FileExistsError(a.output)
a.output.mkdir(parents=True,exist_ok=a.resume)
a.kit_args=f'{a.kit_args or ""} --ext-folder {root.parent}/control-oriented-3d-wing-aero/source/flapping_bot/native_extensions --enable omni.flapping_bot.holonomic_constraint'
app=AppLauncher(a).app
try:
    import torch
    from isaaclab.utils.math import quat_apply, quat_apply_inverse
    from flapping_bot.direct.flapping_bot.straight_flight_env import FlappingBotStraightFlightEnv, FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg
    from flapping_bot.physics.control_aero_3d import ControlAero3DConfig
    from flapping_bot.analysis.pure_rl_fixed_action_reward_validation import _configure_robot_asset

    torch.set_num_threads(1)
    N = 13
    cfg = FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg()
    cfg.seed = 0
    cfg.scene.num_envs = N
    cfg.scene.env_spacing = 30.
    cfg.sim.device = 'cpu'
    cfg.randomize_straight_line_heading = cfg.randomize_flap_phase_at_reset = False
    cfg.teacher_guidance_enabled = False
    cfg.freeze_steps_after_reset = 0
    cfg.wind_enabled = cfg.wind_ou_enabled = False
    cfg.height_cmd = 20.
    cfg.control_aero_3d = ControlAero3DConfig(backend=a.backend, twist_amplitude_deg=35., sections_per_wing=a.sections)
    cfg.delaurier_wing_geom_csv = root.parent / 'control-oriented-3d-wing-aero/outputs_DeLaurier/right_wing_te_fit_poly5_gap50.csv'
    _configure_robot_asset(cfg, asset_path=root.parent / 'control-oriented-3d-wing-aero/source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf', usd_dir=root / 'logs/control_aero_3d/assets')
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

    from flapping_bot.physics.tail_v2 import load_tail_config
    from flapping_bot.physics.tail_aero import TailAeroModel
    from tail_v2_search import dump,csvwrite
    from analyze_periodic_control_aero_3d import phase_sections
    assert abs(dt-1/480)<1e-12 and cfg.decimation==8 and cfg.seed==0
    started=time.monotonic();physical_steps=0;run_records=[];prior_wall=0.
    if a.resume and (a.output/'progress.json').exists():
        previous=json.loads((a.output/'progress.json').read_text());physical_steps=previous['steps'];run_records=previous['runs'];prior_wall=previous['wall_seconds']
    identity={}
    paths=list((root/'source/flapping_bot/flapping_bot/physics').glob('*.py'))
    paths += [Path(cfg.delaurier_wing_geom_csv),Path(cfg.robot.spawn.asset_path)] if hasattr(cfg,'robot') else [Path(cfg.delaurier_wing_geom_csv)]
    assetroot=root.parent/'control-oriented-3d-wing-aero'
    paths += list((assetroot/'source/flapping_bot/native_extensions').rglob('*.so'))
    paths += [assetroot/'source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf']
    for path in paths:
        if path.is_file():identity[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    dump(a.output/'identity.json',dict(files=identity,seed=cfg.seed,dt=dt,decimation=cfg.decimation,wing=asdict(cfg.control_aero_3d),limits_rad=limits.tolist(),joint_names=env._resolved_joint_names))
    def use(name):
        path=evidence/'candidates'/f'{name}.json'
        env._tail_model=TailAeroModel(load_tail_config(path),'cpu')
        folder=a.output/name;folder.mkdir(exist_ok=True)
        dump(folder/'effective_parameters.json',dict(source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),actual=asdict(env._tail_model.cfg)))
        return folder
    def simulate(xs,seconds,name,control=None,case='hold',pulse=False):
        global physical_steps
        if a.resume and Path(name).exists():
            return dict(np.load(name))
        cases=case if isinstance(case,list) else [case]*N
        cases=cases+['hold']*(N-len(cases))
        xs=np.asarray(xs);xs=np.concatenate([xs,np.repeat(xs[:1],N-len(xs),axis=0)])
        action=reset(xs);rows=[];filt=None;phi=None;steps=math.ceil(seconds/dt)
        # No auto-reset, root hold or state correction during integration.
        state=env._robot.data.root_state_w.clone()
        from isaaclab.utils.math import quat_mul,quat_from_euler_xyz
        for i,scenario in enumerate(cases):
            if scenario=='roll+':
                angle=torch.tensor([math.radians(3)]);zero=angle*0
                state[i:i+1,3:7]=quat_mul(state[i:i+1,3:7],quat_from_euler_xyz(angle,zero,zero))
            if scenario in ('q+','q-','yaw+'):
                dw=torch.zeros((1,3));dw[0,1 if scenario.startswith('q') else 2]=(-.2 if scenario=='q-' else .2)
                state[i,10:13]+=quat_apply(state[i:i+1,3:7],dw)[0]
        env._robot.write_root_pose_to_sim(state[:,:7]);env._robot.write_root_velocity_to_sim(state[:,7:13])
        for k in range(steps):
            if k%8==0:
                row=sample(k*dt);angles=row['euler'];rates=row['omega']
                if phi is None:phi=angles.copy()
                else:phi+=np.arctan2(np.sin(angles-phi),np.cos(angles-phi))
                if filt is None:filt=np.c_[phi,rates]
                else:filt+=(1-math.exp(-(1/60)/protocol['filter_tau_s']))*(np.c_[phi,rates]-filt)
                target=np.zeros((N,3));target[:,1]=-np.deg2rad(xs[:,1])
                if k*dt>=2:
                    for i,scenario in enumerate(cases):
                        if scenario.startswith('pitch'):target[i,1]+=math.radians(3)*(1 if scenario.endswith('+') else -1)
                        if scenario.startswith('roll_cmd'):target[i,0]+=math.radians(5)*(1 if scenario.endswith('+') else -1)
                        if scenario=='yaw_cmd':target[i,2]+=math.radians(5)
                desired=np.c_[xs[:,3],xs[:,5],xs[:,4]]*math.pi/180
                if control is not None:
                    for i,(wn,zeta,Binv) in enumerate(control):
                        err=np.arctan2(np.sin(target[i]-filt[i,:3]),np.cos(target[i]-filt[i,:3]))
                        acc=wn**2*err-2*zeta*wn*filt[i,3:]
                        desired[i]+=Binv@acc
                if pulse:
                    for j in range(3):
                        desired[1+2*j,j]+=math.radians(2)
                        desired[2+2*j,j]-=math.radians(2)
                cmd=np.c_[desired[:,2],desired[:,0]+desired[:,1],desired[:,0]-desired[:,1]]
                for j in range(3):action[:,j+1]=torch.as_tensor(np.clip(2*(cmd[:,j]-limits[j,0])/(limits[j,1]-limits[j,0])-1,-1,1),dtype=action.dtype)
                env._pre_physics_step(action)
                row.update(filtered=filt.copy(),target=target,requested_joint=cmd.copy())
                rows.append(row)
            env._sim_step_counter+=1
            env._apply_action();env.scene.write_data_to_sim();env.sim.step(render=False);env.scene.update(dt=dt)
            physical_steps+=1
            if k%8==0:
                rows[-1]['tail_moment']=env._debug_last_tail_moment_b_about_base_com_nm.numpy().copy()
                rows[-1]['wing_moment']=env._debug_last_wing_moment_b_about_base_com_nm.numpy().copy()
                dd=env._robot.data;air=dd.root_lin_vel_b-quat_apply_inverse(dd.root_quat_w,env._wind_w)
                sr=env._tail_model.compute_surface_results(root_lin_vel_b=air,root_ang_vel_b=dd.root_ang_vel_b,left_elevon_rad=env._debug_last_actual_left_elevon_rad,right_elevon_rad=env._debug_last_actual_right_elevon_rad,rudder_rad=env._debug_last_actual_rudder_rad,base_com_pos_b=dd.body_com_pos_b[:,int(env._base_body_ids[0])])
                rows[-1]['alpha_eff']=np.stack([r.alpha_rad.numpy() for r in sr],-1)
                rows[-1]['CL']=np.stack([r.lift_coefficient.numpy() for r in sr],-1)
        tr={key:np.stack([r[key] for r in rows]) for key in rows[0]};tr['phase_unwrapped']=np.unwrap(tr['phase'],axis=0)
        finite=all(np.isfinite(v).all() for v in tr.values())
        np.savez_compressed(name,**tr,xs=xs)
        run_records.append(dict(path=str(name),steps=steps,env_steps=steps*N,finite=finite,case=case))
        dump(a.output/'progress.json',dict(steps=physical_steps,env_steps=physical_steps*N,wall_seconds=prior_wall+time.monotonic()-started,runs=run_records))
        return tr
    def trim_metric(tr,i):
        sec=phase_sections(tr,i)
        if len(sec['velocity'])<3:return 1e12
        dd={k:np.diff(v,axis=0) for k,v in sec.items()}
        r=np.r_[dd['velocity'][-3:].ravel()/.3,dd['euler'][-3:].ravel()/math.radians(3),dd['omega'][-3:].ravel()/.3,dd['position'][-3:,1:].ravel()/.15]
        return float(np.mean(r*r)) if np.isfinite(r).all() else 1e12
    def ctrl_metric(tr,i):
        cut=60;err=tr['filtered'][cut:,i,:3]-tr['target'][cut:,i]
        err=np.arctan2(np.sin(err),np.cos(err));rms=np.sqrt(np.mean(err**2,0))*180/math.pi
        speed=np.linalg.norm(tr['air'][:,i],axis=-1);height=tr['position'][-1,i,2]-tr['position'][0,i,2]
        actual=tr['joint'][:,i,2:5];sat=np.mean(np.any((actual<=limits[:,0]+math.radians(.5))|(actual>=limits[:,1]-math.radians(.5)),axis=-1));large=np.mean(np.any(abs(actual)>math.radians(25),axis=-1))
        p=env._tail_model.cfg
        af=math.radians((p.polar_horizontal or {}).get('alpha_f_deg',30))
        extra=np.mean(np.any(abs(tr['alpha_eff'][:,i])>=af,axis=-1))
        score=float(np.mean((rms/np.array([5,5,10]))**2)+(height/3)**2+np.mean(((speed-8)/2)**2)+sat*10+large+extra*10)
        gates=protocol['task_gates'];ok=bool(np.isfinite(score) and max(rms[:2])<=5 and rms[2]<=10 and speed.min()>=6 and speed.max()<=12 and abs(height)<=3 and sat<=.05 and large<=.2 and extra<=.1 and np.max(abs(tr['euler'][:,i,:2]))<=math.radians(45))
        return dict(score=score if np.isfinite(score) else 1e12,accepted=ok,rms_roll_pitch_yaw_deg=rms.tolist(),height_drift_m=float(height),speed_min=float(speed.min()),speed_max=float(speed.max()),saturation_fraction=float(sat),large_tail_fraction=float(large),extrapolation_fraction=float(extra),raw_rate_rms=np.sqrt(np.mean(tr['omega'][:,i]**2,axis=0)).tolist(),filtered_rate_rms=np.sqrt(np.mean(tr['filtered'][:,i,3:]**2,axis=0)).tolist())
    selected=json.loads((evidence/'selected.json').read_text());trim=[]
    lo,hi=np.array(protocol['trim_bounds']);step=np.array([.5,2,.25,3,2,2])
    for cid in selected:
        folder=use(cid)
        if a.resume and (folder/'trim.json').exists():
            trim.append(json.loads((folder/'trim.json').read_text()));continue
        best=np.array([8,7,3.5,0,0,0.]);history=[]
        for iteration in range(protocol['trim_rounds']):
            xs=[best.copy()]
            for j in range(6):
                for sign in (-1,1):
                    x=best.copy();x[j]+=sign*step[j];xs.append(np.clip(x,lo,hi))
            tr=simulate(xs,protocol['trim_cycles']/min(np.array(xs)[:,2])+.03,folder/f'trim_{iteration}.npz')
            scores=[trim_metric(tr,i) for i in range(N)];winner=int(np.argmin(scores));best=xs[winner]
            history.append(dict(xs=np.array(xs).tolist(),scores=scores))
        tr=simulate([best],protocol['confirmation_cycles']/best[2]+.03,folder/'trim_confirm.npz')
        rec=dict(id=cid,best=best.tolist(),score=trim_metric(tr,0),history=history)
        dump(folder/'trim.json',rec);trim.append(rec);dump(a.output/'trim_results.json',trim)
        print('TRIM',cid,rec['score'],flush=True)
    shortlist=sorted([r for r in trim if r['id'] not in ('legacy','original_v2')],key=lambda r:r['score'])[:4]
    shortlist += [r for r in trim if r['id'] in ('legacy','original_v2')]
    controls=[]
    for rec in shortlist:
        cid=rec['id'];folder=use(cid);best=np.array(rec['best'])
        if a.resume and (folder/'controller.json').exists():
            controls.append(json.loads((folder/'controller.json').read_text()));continue
        tr=simulate([best],.5,folder/'control_signs.npz',pulse=True)
        # Fit finite-time actual joint impulse -> body rate difference, including cross coupling.
        names=env._resolved_joint_names;jp=tr['joint'];actual=np.stack([(jp[:,:,3]+jp[:,:,4])/2,(jp[:,:,3]-jp[:,:,4])/2,jp[:,:,2]],-1)
        impulses=[];rates=[]
        for j in range(3):
            plus,minus=1+2*j,2+2*j
            impulses.append(np.trapz(actual[:,plus]-actual[:,minus],dx=1/60,axis=0))
            rates.append(tr['omega'][-1,plus]-tr['omega'][-1,minus])
        B=np.array(rates).T@np.linalg.pinv(np.array(impulses).T,rcond=.05)
        from flapping_bot.analysis.tail_control_allocation import inverse_effectiveness
        Binv=inverse_effectiveness(B)
        gains=[(wn,z,Binv) for wn in (1.5,3.,5.) for z in (.5,1.,1.5)]
        dump(folder/'control_effectiveness.json',dict(B_rad_s2_per_rad=B.tolist(),actual_impulses_rad_s=np.array(impulses).tolist(),condition=float(np.linalg.cond(B)),units='columns symmetric,differential,rudder; rows p,q,r; actual PhysX finite-time response, not isolated aero derivative'))
        tr=simulate([best],4,folder/'controller_search.npz',control=gains,case='q+')
        metrics=[ctrl_metric(tr,i) for i in range(9)];winner=min(range(9),key=lambda i:(not metrics[i]['accepted'],metrics[i]['score']))
        cr=dict(id=cid,best=rec['best'],wn=gains[winner][0],zeta=gains[winner][1],Binv=Binv.tolist(),metrics=metrics,winner=winner,score=metrics[winner]['score'])
        controls.append(cr);dump(folder/'controller.json',cr);dump(a.output/'controller_results.json',controls)
        print('CONTROL',cid,cr['score'],flush=True)
    finalists=sorted([r for r in controls if r['id'] not in ('legacy','original_v2')],key=lambda r:r['score'])[:2]
    finalists += [r for r in controls if r['id']=='legacy']
    dump(a.output/'frozen_finalists.json',finalists)
    finals=[]
    for rec in finalists:
        folder=use(rec['id']);ctrl=[(rec['wn'],rec['zeta'],np.array(rec['Binv']))]*N
        for speed in protocol['final_speeds']:
            best=np.array(rec['best']);best[0]=speed
            cases=protocol['final_cases']
            tr=simulate([best],protocol['final_seconds'],folder/f'final_{speed}_batch.npz',control=ctrl,case=cases)
            for i,case in enumerate(cases):
                m=ctrl_metric(tr,i);finals.append(dict(id=rec['id'],speed=speed,case=case,**m))
                print('FINAL',rec['id'],speed,case,m['accepted'],flush=True)
            dump(a.output/'final_results.json',finals)
    accepted=[r['id'] for r in finalists if r['id']!='legacy' and all(x['accepted'] for x in finals if x['id']==r['id'])]
    dump(a.output/'completion.json',dict(completed=True,status='ACCEPTED' if accepted else 'NOT ACCEPTED',accepted=accepted,physical_steps=physical_steps,env_steps=physical_steps*N,wall_seconds=prior_wall+time.monotonic()-started))
    env.close()
finally:
    app.close()
