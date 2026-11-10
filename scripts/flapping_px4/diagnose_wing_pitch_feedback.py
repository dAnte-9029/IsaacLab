"""Offline L3D partial derivatives on archived actual-link states. No simulator."""
import argparse
import dataclasses
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_tail_moments import rotation, csvwrite
from flapping_bot.physics.control_aero_3d import ControlAero3D, ControlAero3DConfig
from flapping_bot.physics.control_aero_kinematics import LinkKinematics3D

ROOT = Path(__file__).resolve().parents[2]

def tensor(a):
    return torch.as_tensor(np.asarray(a).copy(), dtype=torch.float64)


def prepare(d, p, indices):
    ids = p['wing_ids'][::-1]
    joints = [p['joint_names'].index(s+'_wing') for s in ('right', 'left')]
    dt = p['dt']; tau = .15/4
    # Observer runs after phase update. Aero earlier in the same call uses
    # old phase/frequency, previous-step phase acceleration, and current qdd.
    f = d['frequency'].reshape(-1).astype(float)
    ph = d['phase'].reshape(-1).astype(float)
    oldf = f[indices-1]; oldph = ph[indices-1]
    acc = 2*np.pi*(f[indices]-oldf)/(tau*(1-np.exp(-dt/tau)))
    qdd = float(np.float32(.5233)*np.float32(.98))*(np.cos(oldph)*acc-np.sin(oldph)*(2*np.pi*oldf)**2)
    take = lambda k: d[k][indices]
    B = tensor(rotation(take('quat')))
    ref = tensor(take('body_com')[:,p['base_id']])
    args = dict(position=tensor(take('body_link_pos')[:,ids]),rotation=tensor(rotation(take('body_quat')[:,ids])),
        com=tensor(take('body_com')[:,ids]),velocity=tensor(take('body_velocity')[:,ids]),
        omega=tensor(take('body_omega')[:,ids]),wind=tensor(take('wind')),phase=tensor(oldph),
        phase_rate=tensor(2*np.pi*oldf),phase_accel=tensor(2*np.pi*(oldf-f[indices-2])/(tau*(1-np.exp(-dt/tau)))),joint_rate=tensor(take('joint_velocity')[:,joints]),
        joint_accel=tensor(np.stack((-qdd,qdd),1)))
    return args, B, ref, tensor(take('air'))


def perturb(args, B, ref, air, kind, h):
    out = {k:v.clone() for k,v in args.items()}
    if kind == 'q':
        dw = B[:,:,1]*h
        out['omega'] += dw[:,None]
        out['velocity'] += torch.linalg.cross(dw[:,None].expand_as(out['com']),out['com']-ref[:,None])
    else:
        a = -torch.atan2(air[:,2],air[:,0]); new = air.clone()
        if kind == 'alpha':
            xz = torch.linalg.vector_norm(air[:,[0,2]],dim=1)
            new[:,0] = xz*torch.cos(a+h); new[:,2] = -xz*torch.sin(a+h)
        elif kind == 'V':
            new *= (1+h/torch.linalg.vector_norm(air,dim=1))[:,None]
        else:
            raise ValueError(kind)
        out['wind'] -= torch.einsum('bij,bj->bi',B,new-air)
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=ROOT/'docs/results/tail_moment_attribution_v1')
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    p=json.loads((a.input/'parameters.json').read_text())
    cfg=ControlAero3DConfig(backend='L3D',twist_amplitude_deg=35.,sections_per_wing=10)
    kin=LinkKinematics3D(ROOT/'outputs_DeLaurier/right_wing_te_fit_poly5_gap50.csv',ROOT/'source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf',cfg,dtype=torch.float64)
    model=ControlAero3D(cfg); noam=ControlAero3D(dataclasses.replace(cfg,include_added_mass=False))
    summary={}; checks={}; sample_rows=[]
    for start in (120,144):
        data={g:dict(np.load(a.input/f'T{start}_{g}_physics.npz')) for g in ('O','S+','S-')}
        ix=np.arange(start*8,start*8+480); time=(np.arange(480)-1)*p['dt']
        states={g:prepare(d,p,ix) for g,d in data.items()}
        args,B,ref,air=states['O']
        def evaluate(state, am=True):
            arg,b,r,_=state; geo,m,rel=kin.sample(**arg)
            loads=(model if am else noam).evaluate(geo,m,r,rel)
            return torch.einsum('bij,bi->bj',b,loads.moment.sum(1))[:,1].numpy()
        baseline=evaluate(states['O']); circ={g:evaluate(s,False) for g,s in states.items()}
        for g,s in states.items():
            error=np.max(abs(evaluate(s)-data[g]['wing_moment_b_about_base_com_nm'][ix,1]))
            checks[f'T{start}_{g}_reconstruction_max_Nm']=float(error)
            assert error<5e-5, (start,g,error)
        geo,m,rel=kin.sample(**args)
        qa=perturb(args,B,ref,air,'q',.05); qg,qm,qr=kin.sample(**qa)
        dw=B[:,:,1]*.05
        expected=-torch.linalg.cross(dw[:,None].expand_as(geo.three_quarter_chord_w_m),geo.three_quarter_chord_w_m-ref[:,None])
        checks[f'T{start}_q_station_flow_error_mps']=float((qm.flow_three_quarter_w_mps-m.flow_three_quarter_w_mps-expected).abs().max())
        checks[f'T{start}_relative_motion_error']=max(float((getattr(qr,k)-getattr(rel,k)).abs().max()) for k in vars(rel))
        aa=perturb(args,B,ref,air,'alpha',np.deg2rad(.5))
        anew=air-torch.einsum('bji,bj->bi',B,aa['wind']-args['wind'])
        checks[f'T{start}_alpha_speed_error_mps']=float((anew.norm(dim=1)-air.norm(dim=1)).abs().max())
        checks[f'T{start}_alpha_beta_error']=float((anew[:,1]/anew.norm(dim=1)-air[:,1]/air.norm(dim=1)).abs().max())
        checks[f'T{start}_alpha_increment_error_rad']=float((-torch.atan2(anew[:,2],anew[:,0])+torch.atan2(air[:,2],air[:,0])-np.deg2rad(.5)).abs().max())
        for key in list(checks):
            if key.startswith(f'T{start}_') and 'reconstruction' not in key: assert checks[key]<1e-10
        body_geo=torch.einsum('bij,bni->bnj',B,geo.quarter_chord_w_m-ref[:,None])
        body_flowgeo=torch.einsum('bij,bni->bnj',B,geo.three_quarter_chord_w_m-ref[:,None])
        mechanism={}
        for kind,h in [('alpha',np.deg2rad(.5)),('q',.05)]:
            forces=[]
            for sign in (1,-1):
                gg,mm,rr=kin.sample(**perturb(args,B,ref,air,kind,sign*h))
                ff=model.evaluate(gg,mm,ref,rr).force
                forces.append(torch.einsum('bij,bni->bnj',B,ff))
            df=(forces[0]-forces[1])/(2*h)
            mechanism[kind+'_rz_dFx']=(body_geo[:,:,2]*df[:,:,0]).sum(1).numpy()
            mechanism[kind+'_minus_rx_dFz']=(-body_geo[:,:,0]*df[:,:,2]).sum(1).numpy()
        checks[f'T{start}_force_arm_x_range_m']=[float(body_geo[:,:,0].min()),float(body_geo[:,:,0].max())]
        checks[f'T{start}_flow_arm_x_range_m']=[float(body_flowgeo[:,:,0].min()),float(body_flowgeo[:,:,0].max())]
        deriv={}
        for kind,h,small,label in [('alpha',np.deg2rad(.5),np.deg2rad(.25),'Kalpha'),('q',.05,.025,'Dq'),('V',.05,.025,'KV')]:
            for step,suffix in [(h,''),(small,'_half')]:
                pair=[evaluate((perturb(args,B,ref,air,kind,sign*step),B,ref,air)) for sign in (1,-1)]
                deriv[label+suffix]=(pair[0]-pair[1])/(2*step)
            checks[f'T{start}_{label}_step_relative_L2']=float(np.linalg.norm(deriv[label]-deriv[label+'_half'])/np.linalg.norm(deriv[label+'_half']))
            assert checks[f'T{start}_{label}_step_relative_L2']<.01
        plus,minus=data['S+'],data['S-']
        alpha=lambda d:-np.arctan2(d['air'][ix,2],d['air'][ix,0])
        da=alpha(plus)-alpha(minus); dq=(plus['omega']-minus['omega'])[ix,1]
        dv=np.linalg.norm(plus['air'][ix],axis=1)-np.linalg.norm(minus['air'][ix],axis=1)
        actual=(plus['wing_moment_b_about_base_com_nm']-minus['wing_moment_b_about_base_com_nm'])[ix,1]
        pa=deriv['Kalpha']*da; pq=deriv['Dq']*dq; pv=deriv['KV']*dv
        # Same-body-coordinate nonlinear reconstruction with O relative geometry
        # and wing motion; replace only alpha,q,V with each branch values.
        nonlinear=[]
        for g in ('S+','S-'):
            d=data[g]; arg={k:v.clone() for k,v in args.items()}
            deltaq=tensor(d['omega'][ix,1])-tensor(data['O']['omega'][ix,1])
            dw=B[:,:,1]*deltaq[:,None]
            arg['omega']+=dw[:,None]
            arg['velocity']+=torch.linalg.cross(dw[:,None].expand_as(arg['com']),arg['com']-ref[:,None])
            targetalpha=tensor(alpha(d)); speed=tensor(np.linalg.norm(d['air'][ix],axis=1)); beta=torch.asin(air[:,1]/air.norm(dim=1))
            new=torch.stack((speed*beta.cos()*targetalpha.cos(),speed*beta.sin(),-speed*beta.cos()*targetalpha.sin()),1)
            arg['wind']-=torch.einsum('bij,bj->bi',B,new-air)
            nonlinear.append(evaluate((arg,B,ref,air)))
        nl=nonlinear[0]-nonlinear[1]
        amdiff=actual-(circ['S+']-circ['S-'])
        phase=args['phase'].numpy(); cycle=(phase>=phase[1])&(phase<phase[1]+2*np.pi)
        last=np.where(cycle)[0][-1]
        fraction=(phase[1]+2*np.pi-phase[last])/(phase[last+1]-phase[last])
        weights=np.ones(cycle.sum())*p['dt'];weights[-1]*=fraction
        arrays=dict(time_s=time,phase_rad=phase,phase_mod_rad=phase%(2*np.pi),cycle=cycle.astype(int),**deriv,**mechanism,
            alpha_O_rad=alpha(data['O']),q_O_rad_s=data['O']['omega'][ix,1],V_O_mps=np.linalg.norm(data['O']['air'][ix],axis=1),
            delta_alpha_rad=da,delta_q_rad_s=dq,delta_V_mps=dv,actual_Nm=actual,alpha_term_Nm=pa,q_term_Nm=pq,V_term_Nm=pv,
            prediction_aq_Nm=pa+pq,prediction_aqV_Nm=pa+pq+pv,nonlinear_aqV_Nm=nl,added_mass_difference_Nm=amdiff)
        csvwrite(a.output/f'T{start}_derivatives.csv',[{k:float(v[i]) for k,v in arrays.items()} for i in range(len(ix))])
        info={'cycle_start_s':float(time[cycle][0]),'cycle_end_exclusive_s':float(time[last]+fraction*p['dt']), 'cycle_phase_span_sampled_rad':float(np.ptp(phase[cycle])), 'cycle':{},'windows':{}}
        for name in ('Kalpha','Dq'):
            x=deriv[name][cycle];info['cycle'][name]=dict(mean=float(np.average(x,weights=weights)),min=float(x.min()),max=float(x.max()),positive_fraction=float(np.average(x>0,weights=weights)),negative_fraction=float(np.average(x<0,weights=weights)))
        for lo,hi in [(0,.1),(.1,.2),(.2,.5),(.5,1)]:
            mask=(time>=lo-1e-9)&(time<hi-1e-9); y=actual[mask]
            st={}
            for name,x in [('actual',actual),('alpha',pa),('q',pq),('V',pv),('aq',pa+pq),('aqV',pa+pq+pv),('nonlinear',nl),('AM',amdiff)]:
                st[name]=dict(mean=float(x[mask].mean()),integral=float(x[mask].sum()*p['dt']),rmse=float(np.sqrt(np.mean((x[mask]-y)**2))),sign_agreement=float(np.mean(np.sign(x[mask])==np.sign(y))))
            info['windows'][f'{lo}-{hi}']=st
        for t in (0,.1,.2,.5,time[-1]):
            i=np.argmin(abs(time-t));sample_rows.append(dict(start=start,**{k:float(v[i]) for k,v in arrays.items()}))
        info['mechanism_cycle_means']={k:float(np.average(v[cycle],weights=weights)) for k,v in mechanism.items()}
        info['phase_difference_max_rad']=float(np.max(abs(plus['phase'][ix]-minus['phase'][ix])))
        info['frequency_difference_max_hz']=float(np.max(abs(plus['frequency'][ix]-minus['frequency'][ix])))
        info['wing_joint_velocity_difference_max_rad_s']=float(np.max(abs((plus['joint_velocity']-minus['joint_velocity'])[ix][:,[p['joint_names'].index(s+'_wing') for s in ('left','right')]])))
        summary[str(start)]=info
        fig,axes=plt.subplots(2,2,figsize=(11,7))
        for row,name in enumerate(('Kalpha','Dq')):
            for col,x,sel in [(0,time,np.ones(len(time),bool)),(1,(phase-phase[1])/(2*np.pi),cycle)]:
                axes[row,col].plot(x[sel],deriv[name][sel],label='nominal step');axes[row,col].plot(x[sel],deriv[name+'_half'][sel],'--',label='half step');axes[row,col].axhline(0,color='k',lw=.5);axes[row,col].grid();axes[row,col].set_ylabel(name+(' [Nm/rad]' if row==0 else ' [Nm/(rad/s)]'));axes[row,col].set_xlabel('time [s]' if col==0 else 'phase advance / 2pi');axes[row,col].legend()
        fig.tight_layout();fig.savefig(a.output/f'T{start}_sensitivities.png',dpi=160);plt.close(fig)
        fig,axes=plt.subplots(3,1,figsize=(11,10),sharex=True)
        for name,x in [('actual',actual),('Kalpha delta alpha',pa),('Dq delta q',pq),('aq sum',pa+pq)]: axes[0].plot(time,x,label=name)
        for name,x in [('actual',actual),('aqV sum',pa+pq+pv),('nonlinear aqV',nl),('AM difference',amdiff)]:axes[1].plot(time,x,label=name)
        for name,x in [('delta alpha [rad]',da),('delta q [rad/s]',dq),('delta V [m/s]',dv)]:axes[2].plot(time,x,label=name)
        for ax in axes:ax.grid();ax.legend();ax.axhline(0,color='k',lw=.5)
        axes[0].set_ylabel('Delta My [Nm]');axes[1].set_ylabel('Delta My [Nm]');axes[2].set_xlabel('time since first changed tail load [s]')
        fig.tight_layout();fig.savefig(a.output/f'T{start}_attribution.png',dpi=160);plt.close(fig)
    csvwrite(a.output/'samples.csv',sample_rows)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (a.output/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    (a.output/'configuration.json').write_text(json.dumps(dataclasses.asdict(cfg),indent=2)+'\n')
    print(json.dumps({'completed':True,'checks':checks,'summary':summary},indent=2))

if __name__=='__main__':main()
