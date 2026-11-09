"""Offline load, same-state and multibody diagnostics; never writes simulation state."""
import argparse, csv, json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flapping_bot.physics.tail_aero import TailAeroCfg,TailSurfaceCfg,TailAeroModel
from flapping_bot.analysis.tail_unit_audit import TailAuditSettings,evaluate_sweep

NAMES=['fixed_horizontal','left_elevon','right_elevon','fixed_vertical','rudder']
def rotation(q):
    return Rotation.from_quat(np.asarray(q)[...,[1,2,3,0]].reshape(-1,4)).as_matrix().reshape(*q.shape[:-1],3,3)
def csvwrite(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
def query(model,d,i,angles):
    n=len(angles); t=lambda a:torch.tensor(a,dtype=torch.float64)
    return model.compute_surface_results(root_lin_vel_b=t(d['air'][i]).repeat(n,1),root_ang_vel_b=t(d['omega'][i]).repeat(n,1),
        base_com_pos_b=t(d['base_com_b'][i]).repeat(n,1),left_elevon_rad=t(angles[:,1]),right_elevon_rad=t(angles[:,2]),rudder_rad=t(angles[:,4]))
def my(results): return np.stack([r.moment_b_about_com[:,1].numpy() for r in results],1)
def dynamics(d,p):
    R=rotation(d['body_quat']); B=rotation(d['quat']); m=np.array(p['masses']); c=d['body_com']; v=d['body_velocity']; w=d['body_omega']
    cg=(c*m[None,:,None]).sum(1)/m.sum();vg=(v*m[None,:,None]).sum(1)/m.sum();r=c-cg[:,None]
    # PhysX get_coms quaternion is xyzw, in contrast to Isaac Lab wxyz body quaternions.
    C=Rotation.from_quat(np.array(p['coms'])[:,3:]).as_matrix()
    Il=C@np.array(p['inertias']).reshape(-1,3,3)@C.transpose(0,2,1)
    Iw=R@Il@R.swapaxes(-1,-2)
    H=(np.einsum('nbij,nbj->nbi',Iw,w)+np.cross(r,m[None,:,None]*(v-vg[:,None]))).sum(1)
    Ilock=(Iw+m[None,:,None,None]*(np.sum(r*r,axis=-1)[...,None,None]*np.eye(3)-r[...,None]*r[...,None,:])).sum(1)
    Ibody=B.swapaxes(-1,-2)@Ilock@B
    f=np.einsum('nij,nj->ni',B,d['force_b']);M=np.einsum('nij,nj->ni',B,d['torque_b'])+np.cross(c[:,p['base_id']]-cg,f)
    cg_b=np.einsum('nji,nj->ni',B,cg-c[:,p['base_id']])
    residualH=H-np.einsum('nij,nj->ni',Ilock,w[:,p['base_id']])
    # Per-link wing wrench shifted to base COM and rotated to base FLU.
    ids=p['wing_ids'];fw=np.einsum('nwij,nwj->nwi',R[:,ids],d['wing_force_link_n'])
    mw=np.einsum('nwij,nwj->nwi',R[:,ids],d['wing_moment_link_about_com_nm'])+np.cross(c[:,ids]-c[:,p['base_id'],None],fw)
    mwb=np.einsum('nji,nwj->nwi',B,mw)
    return dict(cg=cg,cg_b=cg_b,Ibody=Ibody,H=H,Mworld=M,Hrelative=residualH,wing_individual=mwb,
                Mcom_b=np.einsum('nji,nj->ni',B,M))
def main():
    a=argparse.ArgumentParser();a.add_argument('--input',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    results={};endpoints=[];onsets=[];samples=[];integrals=[];same=[];swaps=[];checks={};dynstats=[]
    for start in (120,144):
        data={};dyn={}
        for g in ('O','S+','S-'):
            folder=args.input/f'T{start}_{g}';p=json.loads((folder/'parameters.json').read_text());d=dict(np.load(folder/'physics.npz'));data[g]=d;dyn[g]=dynamics(d,p)
            for v in d.values(): assert np.isfinite(v).all()
            old=Path('logs/unified_sustained_tail_authority_v1')/folder.name/'trace.jsonl'
            # Unchanged full replay is the observer non-interference check, including all five environments.
            newrows=[json.loads(s) for s in (folder/'trace.jsonl').read_text().splitlines()];oldrows=[json.loads(s) for s in old.read_text().splitlines()]
            assert newrows==oldrows,folder
            checks[folder.name+'_historical_trace_exact']=True
            root=np.array(newrows[-1]['visible']['root_state'])[1]
            endpoints.append(dict(start=start,group=g,time_s=1.,position=root[:3].tolist(),velocity=root[7:10].tolist(),quat=root[3:7].tolist(),omega_b=(rotation(root[3:7]).T@root[10:13]).tolist()))
            # Archive the necessary raw physics samples; full 5-environment traces remain local.
            import shutil
            shutil.copyfile(folder/'physics.npz',out/f'T{start}_{g}_physics.npz')
            checks[folder.name+'_tail_closure']=float(np.max(abs(d['surface_moment_b_about_com'].sum(1)-d['tail_moment_b_about_base_com_nm'])))
            checks[folder.name+'_wing_reference_closure']=float(np.max(abs(dyn[g]['wing_individual'].sum(1)-d['wing_moment_b_about_base_com_nm'])))
            assert checks[folder.name+'_tail_closure']<2e-6 and checks[folder.name+'_wing_reference_closure']<2e-5
        dt=p['dt'];j0=start*8;time=(np.arange(480)-1)*dt
        # Physical clock zero is the first changed tail load, one physics step after command onset.
        plus=data['S+'];minus=data['S-'];o=data['O'];sl=slice(j0,j0+480)
        for g in ('S+','S-'):
            for k in o: assert np.array_equal(data[g][k][:j0],o[k][:j0]),(start,g,k)
        tail=(plus['tail_moment_b_about_base_com_nm']-minus['tail_moment_b_about_base_com_nm'])[sl,1]
        wing=(plus['wing_moment_b_about_base_com_nm']-minus['wing_moment_b_about_base_com_nm'])[sl,1]
        total=(plus['torque_b']-minus['torque_b'])[sl,1]
        def pitch(d):
            q=d['quat'];w,x,y,z=q.T;return np.arcsin(np.clip(2*(w*y-z*x),-1,1))
        alpha=lambda d:-np.arctan2(d['air'][:,2],d['air'][:,0])
        qdiff=(plus['omega']-minus['omega'])[sl,1];adiff=(alpha(plus)-alpha(minus))[sl]
        pdiff=(pitch(plus)-pitch(minus))[sl];vdiff=(plus['velocity']-minus['velocity'])[sl,2];zdiff=(plus['position']-minus['position'])[sl,2]
        arrays=dict(time_s=time,tail_My_Nm=tail,wing_My_Nm=wing,total_My_Nm=total,body_My_Nm=total-tail-wing,
            q_rad_s=qdiff,alpha_rad=adiff,pitch_rad=pdiff,vz_m_s=vdiff,z_m=zdiff,
            qdot_rad_s2=np.gradient(qdiff,dt,edge_order=2))
        for name,v,threshold in [('wing',wing,.01),('q',qdiff,.01),('alpha',adiff,np.deg2rad(.1)),('tail',tail,.01)]:
            hits=np.where(np.convolve((abs(v)>threshold).astype(int),np.ones(3,dtype=int),mode='valid')==3)[0]
            onsets.append(dict(start=start,quantity=name,threshold=threshold,time_s=float(time[hits[0]]) if len(hits) else None))
        csvwrite(out/f'T{start}_differences.csv',[{k:float(v[i]) for k,v in arrays.items()} for i in range(480)])
        for t in (.05,.1,.2,.5,1.):
            # 1s load uses last applied step (1-dt), motion uses original endpoint trace separately below.
            idx=min(round(t/dt)+1,479);samples.append(dict(start=start,requested_s=t,load_time_s=float(time[idx]),**{k:float(v[idx]) for k,v in arrays.items() if k!='time_s'}))
        for label,v in [('tail',tail),('wing',wing),('body',total-tail-wing),('total',total)]:
            integrals.append(dict(start=start,component=label,signed_Nms=float(v.sum()*dt),absolute_Nms=float(abs(v).sum()*dt),mean_Nm=float(v.mean()),peak_abs_Nm=float(abs(v).max()),positive_Nms=float(v.clip(0).sum()*dt),negative_Nms=float(v.clip(max=0).sum()*dt)))
        fig,axs=plt.subplots(3,1,figsize=(10,8),sharex=True)
        for label,v in [('tail',tail),('wing',wing),('total',total)]:axs[0].plot(time,v,label=label)
        axs[0].set_ylabel('Delta My, base COM [Nm]');axs[0].legend()
        axs[1].plot(time,np.rad2deg(qdiff),label='q deg/s');axs[1].plot(time,np.rad2deg(adiff),label='aero alpha deg');axs[1].plot(time,np.rad2deg(pdiff),label='Euler pitch deg');axs[1].legend()
        axs[2].plot(time,vdiff,label='world vz m/s');axs[2].plot(time,zdiff,label='world z m');axs[2].legend();axs[2].set_xlabel('seconds since first changed aerodynamic load')
        for ax in axs: ax.axvspan(-dt,.3-dt,color='grey',alpha=.12);ax.grid()
        fig.tight_layout();fig.savefig(out/f'T{start}_response.png',dpi=160);plt.close(fig)
        cfgdict=p['tail'].copy()
        for n in NAMES:cfgdict[n]=TailSurfaceCfg(**cfgdict[n])
        cfg=TailAeroCfg(**cfgdict);model=TailAeroModel(cfg,'cpu')
        for t in (0,.05,.1,.2,.3,.5,1):
            idx=min(j0+round(t/dt),len(o['air'])-1)
            angles=np.stack([o['surface_deflection_rad'][idx],plus['surface_deflection_rad'][idx],minus['surface_deflection_rad'][idx]])
            m=my(query(model,o,idx,angles));delta=(angles[1,1:3]-angles[2,1:3]).mean()
            eps=1e-4;central=np.repeat(angles[:1],2,0);central[0,1:3]+=eps;central[1,1:3]-=eps;gain=np.diff(my(query(model,o,idx,central)).sum(1))[0]/(-2*eps)
            target=angles[0].copy();pm=np.repeat(target[None],2,0);pm[0,1:3]+=np.deg2rad(4.070865);pm[1,1:3]-=np.deg2rad(4.070865)
            mp=my(query(model,o,idx,pm))
            same.append(dict(start=start,time_s=t,speed=float(np.linalg.norm(o['air'][idx])),sym_O_deg=float(np.rad2deg(angles[0,1:3].mean())),sym_plus_deg=float(np.rad2deg(angles[1,1:3].mean())),sym_minus_deg=float(np.rad2deg(angles[2,1:3].mean())),My_O=float(m[0].sum()),My_plus=float(m[1].sum()),My_minus=float(m[2].sum()),difference=float(m[1].sum()-m[2].sum()),gain_Nm_rad=float(gain),target_4deg_difference=float(mp[0].sum()-mp[1].sum()),fixed_My=float(m[0,0]),left_My=float(m[0,1]),right_My=float(m[0,2])))
            # Symmetric two-path decomposition: direct + state equals observed difference exactly.
            pp=my(query(model,plus,idx,angles[1:])).sum(1);mm=my(query(model,minus,idx,angles[1:])).sum(1)
            direct=.5*((pp[0]-pp[1])+(mm[0]-mm[1]));state=.5*((pp[0]-mm[0])+(pp[1]-mm[1]))
            swaps.append(dict(start=start,time_s=t,direct_Nm=float(direct),state_feedback_Nm=float(state),total_Nm=float(pp[0]-mm[1])))
        fig,ax=plt.subplots(figsize=(9,5));sweep_rows=[]
        for t in (0,.2,.5,1):
            idx=min(j0+round(t/dt),len(o['air'])-1);angles=np.repeat(o['surface_deflection_rad'][idx][None],163,0)
            s=np.linspace(-40.5,40.5,163);diff=(angles[0,1]-angles[0,2])/2;angles[:,1]=np.deg2rad(s)+diff;angles[:,2]=np.deg2rad(s)-diff
            m=my(query(model,o,idx,angles));ax.plot(s,m.sum(1),label=f'{t}s')
            for i,x in enumerate(s):sweep_rows.append(dict(time_s=t,symmetric_deg=x,**{n:float(m[i,j]) for j,n in enumerate(NAMES)},total=float(m[i].sum())))
        ax.set(xlabel='Actual symmetric tail angle [deg]',ylabel='My about base COM [Nm]');ax.legend();ax.grid();fig.tight_layout();fig.savefig(out/f'T{start}_same_state.png',dpi=160);plt.close(fig);csvwrite(out/f'T{start}_sweep.csv',sweep_rows)
        for g,d in data.items():
            dyno=dyn[g];H=dyno['H'];M=dyno['Mworld'];i=j0;j=len(H)-1
            impulse=M[i:j].sum(0)*dt;change=H[j]-H[i];err=change-impulse
            dynstats.append(dict(start=start,group=g,mass_kg=sum(p['masses']),Iyy_min=float(dyno['Ibody'][sl,1,1].min()),Iyy_max=float(dyno['Ibody'][sl,1,1].max()),cg_b_min=dyno['cg_b'][sl].min(0).tolist(),cg_b_max=dyno['cg_b'][sl].max(0).tolist(),H_change=change.tolist(),aero_impulse=impulse.tolist(),closure_error=err.tolist(),closure_relative=float(np.linalg.norm(err)/max(np.linalg.norm(impulse),1e-10)),relative_H_peak=float(np.linalg.norm(dyno['Hrelative'][sl],axis=1).max()),qdot_peak=float(abs(np.gradient(d['omega'][:,1],dt)[sl]).max()),tail_alpha_max_deg=float(np.rad2deg(abs(d['surface_alpha_raw_rad'][sl])).max()),clip_fraction=np.mean(abs(d['surface_alpha_raw_rad'][sl])>np.deg2rad(25),axis=0).tolist()))
            B=rotation(d['quat']);omega=d['omega'].astype(float);I=dyno['Ibody'];hr=np.einsum('nji,nj->ni',B,dyno['Hrelative'])
            grad=lambda x:np.gradient(x,dt,axis=0,edge_order=2)
            locked_acc=np.einsum('nij,nj->ni',I,grad(omega))
            changing_I=np.einsum('nij,nj->ni',grad(I),omega)
            gyro=np.cross(omega,np.einsum('nij,nj->ni',I,omega))
            relative=grad(hr)+np.cross(omega,hr)
            residual=dyno['Mcom_b']-locked_acc-changing_I-gyro-relative
            terms=dict(locked_acc=locked_acc,changing_I=changing_I,gyro=gyro,relative_motion=relative,balance_residual=residual)
            dynstats[-1]['pitch_terms_rms_Nm']={k:float(np.sqrt(np.mean(v[sl,1]**2))) for k,v in terms.items()}
            dynstats[-1]['Iyy_only_qdot_error_rms']=float(np.sqrt(np.mean((dyno['Mcom_b'][sl,1]/I[sl,1,1]-grad(omega)[sl,1])**2)))
            dynstats[-1]['surface_CL_peak']=abs(d['surface_lift_coefficient'][sl]).max(0).tolist()
            dynstats[-1]['surface_alpha_peak_deg']=np.rad2deg(abs(d['surface_alpha_raw_rad'][sl])).max(0).tolist()
            np.savez_compressed(out/f'T{start}_{g}_derived.npz',**dyno,**terms)
    (out/'endpoints.json').write_text(json.dumps(endpoints,indent=2));csvwrite(out/'onsets.csv',onsets)
    csvwrite(out/'sample_times.csv',samples);csvwrite(out/'impulses.csv',integrals);csvwrite(out/'same_state.csv',same);csvwrite(out/'state_swap.csv',swaps)
    (out/'checks.json').write_text(json.dumps(checks,indent=2));(out/'dynamics.json').write_text(json.dumps(dynstats,indent=2));(out/'parameters.json').write_text(json.dumps(p,indent=2))
    # Reuse production audit sweep with explicitly supplied effective cfg and actual COM.
    settings=TailAuditSettings(base_com_pos_b_m=tuple(o['base_com_b'][j0]));rows=[];scale=[]
    for speed in (6.,8.,10.):
        s=torch.tensor([-4.070865,0,4.070865],dtype=torch.float64);v=torch.tensor([[speed,0,0]],dtype=torch.float64).repeat(3,1);z=torch.zeros(3,dtype=torch.float64)
        rows+=evaluate_sweep(model,cfg,settings,sweep=f'speed_{speed}',independent_name='symmetric_deg',independent_values=s,v_b=v,w_b=v*0,symmetric_rad=torch.deg2rad(s),differential_rad=z,rudder_rad=z)
        m=model.compute_surface_results(root_lin_vel_b=v,root_ang_vel_b=v*0,base_com_pos_b=torch.tensor(settings.base_com_pos_b_m,dtype=torch.float64).repeat(3,1),left_elevon_rad=torch.deg2rad(s),right_elevon_rad=torch.deg2rad(s),rudder_rad=z)
        force=sum(r.force_b for r in m).numpy();moment=my(m).sum(1);q=.5*cfg.air_density*speed**2;area=2*cfg.left_elevon.area;lever=abs(cfg.left_elevon.lever_arm_body[0]-settings.base_com_pos_b_m[0]);dd=np.deg2rad(8.14173)
        scale.append(dict(speed=speed,q_Pa=q,delta_Fz_N=float(force[2,2]-force[0,2]),delta_My_Nm=float(moment[2]-moment[0]),gain_Nm_rad=float((moment[2]-moment[0])/dd),equivalent_CL_delta_per_rad=float((moment[2]-moment[0])/(q*area*lever*dd)),Re_elevon=speed*cfg.left_elevon.mean_chord_m/1.5e-5))
    csvwrite(out/'physical_scale.csv',scale);csvwrite(out/'physical_scale_surfaces.csv',rows)
    print(json.dumps(dict(checks=checks,scale=scale,same_state=same[:3],dynamics=dynstats[:1]),indent=2))
if __name__=='__main__':main()
