"""Summarize native periodic/paired traces without changing raw experiment data."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear
from scipy.integrate import trapezoid


def interp(t, x, at):
    """Linear sample interpolation; caller unwraps periodic coordinates first."""
    return np.array([np.interp(at,t,col) for col in x.reshape(len(t),-1).T]).reshape(x.shape[1:])


def phase_sections(trace,i):
    ph=trace['phase_unwrapped'][:,i]
    targets=np.arange(0,np.floor((ph[-1]-ph[0])/(2*np.pi))+1)*2*np.pi+ph[0]
    out={}
    for key in ['t','position','velocity','body_velocity','omega','euler','frequency','joint','joint_velocity']:
        val=trace[key][:,i]; shape=val.shape[1:]
        flat=val.reshape(len(val),-1)
        if key=='euler': flat=np.unwrap(flat,axis=0)
        out[key]=np.stack([np.interp(targets,ph,flat[:,j]) for j in range(flat.shape[1])],-1).reshape((len(targets),)+shape)
    return out

def response(z):
    t=z['t'][:,0]; onset=float(z['onset']); f=float(z['xs'][0,2])
    ans={}
    denominator=2*float(z['tail_small']) if 'tail_small' in z else 10.
    for horizon in [1/f,2/f,1.]:
        vals={k:interp(t,np.unwrap(z[k],axis=0) if k=='euler' else z[k],onset+horizon) for k in ['omega','euler','position','velocity','air','beta','joint']}
        delta={k:v-v[0] for k,v in vals.items()}
        # p/r and roll/yaw are retained separately; solve rate compensation only.
        eff=np.stack([(delta['omega'][8,[0,2]]-delta['omega'][9,[0,2]])/denominator,
                      (delta['omega'][10,[0,2]]-delta['omega'][11,[0,2]])/denominator],axis=-1)
        wind=delta['omega'][1,[0,2]]
        tail_limit=np.degrees(.7105000019073486)
        diff_limit=tail_limit-abs(float(z['xs'][0,3]))
        rudder_bias=float(z['xs'][0,4]) if z['xs'].shape[1]>4 else 0.
        diff_bias=float(z['xs'][0,5]) if z['xs'].shape[1]>5 else 0.
        fit=lsq_linear(eff,-wind,bounds=([-tail_limit-rudder_bias,-diff_limit-diff_bias],[tail_limit-rudder_bias,diff_limit-diff_bias]))
        ans[f'{horizon:.6f}']=dict(horizon_s=horizon,
            increments={str(case):{k:delta[k][i].tolist() for k in delta} for i,case in enumerate(z['cases'])},
            actual_tail_deg=np.degrees(vals['joint'][:,2:]).tolist(),
            effect_pr_per_deg=eff.tolist(), compensation_rudder_diff_deg=fit.x.tolist(),
            predicted_residual_pr=(eff@fit.x+wind).tolist(), singular_values=np.linalg.svd(eff,compute_uv=False).tolist(),
            warning='finite-horizon linear prediction only; saturation bound is not verified nonlinear authority')
    pre=t<=onset
    qw,qx,qy,qz=np.moveaxis(z['quat'],-1,0)
    tilt=np.degrees(np.arccos(np.clip(1-2*(qx*qx+qy*qy),-1,1)))
    checks=dict(max_tilt_deg=float(tilt.max()), tilt_over_75_cases=[str(z['cases'][i]) for i in np.flatnonzero((tilt>75).any(0))], preperturbation_max_velocity_difference=float(np.max(np.abs(z['velocity'][pre]-z['velocity'][pre,0:1]))),
                duplicate_max_velocity_error=float(np.max(np.abs(z['velocity'][:,16]-z['velocity'][:,0]))),
                duplicate_max_omega_error=float(np.max(np.abs(z['omega'][:,16]-z['omega'][:,0]))),
                minimum_height=float(z['position'][...,2].min()),
                max_tail_deg=float(np.degrees(np.abs(z['joint'][...,2:])).max()))
    return dict(horizons=ans,checks=checks)


def sustain(folder):
    raw=json.loads((folder/'sustain.json').read_text()); d={k:np.array(v) for k,v in raw['cycle_deltas'].items()}
    s={k:np.array(v) for k,v in raw['sections'].items()}
    z=np.load(folder/'sustain_best.npz'); j=z['joint'][:,0]; ph=z['phase_unwrapped'][:,0]
    # Left-wing midpoint recovered from phase-zero reset; amplitude from plant (~30 deg).
    fit=np.linalg.lstsq(np.stack([np.ones_like(ph),np.sin(ph),np.cos(ph)],axis=1),j[:,0],rcond=None)[0]
    tracking=j[:,0]-(fit[0]+fit[1]*np.sin(ph)+fit[2]*np.cos(ph))
    cycle_means={}
    tt=z['t'][:,0]
    for key in ['velocity','body_velocity','omega','euler','air']:
        values=np.unwrap(z[key][:,0],axis=0) if key=='euler' else z[key][:,0]
        means=[]
        for begin,end in zip(s['t'][:-1],s['t'][1:]):
            mid=(tt>begin)&(tt<end); times=np.r_[begin,tt[mid],end]
            yy=np.vstack([interp(tt,values,begin),values[mid],interp(tt,values,end)])
            means.append(trapezoid(yy,times,axis=0)/(end-begin))
        cycle_means[key]=np.array(means).tolist()
    last=slice(-5,None)
    maxima=dict(displacement_yz_m=float(np.abs(d['position'][last,1:]).max()),velocity_mps=float(max(np.abs(d['velocity'][last]).max(),np.abs(d['body_velocity'][last]).max())),
                attitude_deg=float(np.degrees(np.abs(d['euler'][last])).max()),omega_rad_s=float(np.abs(d['omega'][last]).max()),
                frequency_hz=float(np.abs(d['frequency'][last]).max()),wing_position_deg=float(np.degrees(np.abs(d['joint'][last,:2])).max()),
                wing_velocity_rad_s=float(np.abs(d['joint_velocity'][last,:2]).max()),tail_deg=float(np.degrees(np.abs(d['joint'][last,2:])).max()))
    tol=json.loads((folder/'protocol.json').read_text())['tolerances']
    return dict(best=raw['best'],cycle_means=cycle_means,maxima_last5=maxima,passes={k:v<=tol[k] for k,v in maxima.items()},
                total_position_change=(s['position'][-1]-s['position'][0]).tolist(),
                first_cycle_position_change=d['position'][0].tolist(),
                total_attitude_change_deg=np.degrees(s['euler'][-1]-s['euler'][0]).tolist(),
                final_velocity=s['velocity'][-1].tolist(),
                min_height_change=float(z['position'][:,0,2].min()-z['position'][0,0,2]),
                section_measured_frequency_hz=(1/np.diff(s['t'])).tolist(),
                wing_harmonic_fit_amplitude_deg=float(np.degrees(np.hypot(fit[1],fit[2]))),
                wing_harmonic_fit_phase_deg=float(np.degrees(np.arctan2(fit[2],fit[1]))),
                wing_fit_max_error_deg=float(np.degrees(np.abs(tracking)).max()),
                tail_actual_mean_deg=np.degrees(z['joint'][-160:,0,2:].mean(0)).tolist(),
                mean_force_last_cycle=z['force'][-int(480/raw['best'][2]):,0].mean(0).tolist(),
                mean_moment_last_cycle=z['moment'][-int(480/raw['best'][2]):,0].mean(0).tolist())


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--final',action='store_true');args=p.parse_args()
    result={}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for model in ['L3D','I3D']:
        folder=args.input/(model+('_late' if model=='L3D' else '_refined')) if args.final else args.input/(model+'_host')
        result[model]={'sustain':sustain(folder)}
        for label in ['own','common']:
            result[model][label]=response(np.load(folder/('response_'+label+'.npz')))
        if args.final:
            compensated=args.input/(model+'_compensated')/'response_own.npz'
            if compensated.exists(): result[model]['compensated']=response(np.load(compensated))
            result[model]['near']=response(np.load(args.input/(model+'_near')/'response_own.npz'))
            horizons=list(result[model]['near']['horizons'].values())
            commands=[horizons[0]['compensation_rudder_diff_deg'],horizons[-1]['compensation_rudder_diff_deg']]
            command_path=args.input/(model+'_compensations.json')
            if not command_path.exists(): command_path.write_text(json.dumps(commands,indent=2))
            for label in ['matched','local']:
                result[model][label]=response(np.load(args.input/(model+'_'+label)/'response_common.npz'))
        z=np.load(folder/'sustain_best.npz');t=z['t'][:,0]
        for ax,key,col,scale in [(axes[0,0],'position',2,1),(axes[0,1],'velocity',0,1),(axes[0,2],'euler',1,-180/np.pi)]:
            y=z[key][:,0,col]*scale
            if key=='position':y=y-y[0]
            ax.plot(t,y,label=model)
        z=np.load(args.input/(model+'_local')/'response_common.npz') if args.final else np.load(folder/'response_common.npz');t=z['t'][:,0]-z['onset'];mask=t>=0
        for ax,key,col,scale in [(axes[1,0],'omega',0,1),(axes[1,1],'omega',2,1),(axes[1,2],'euler',0,180/np.pi)]:
            ax.plot(t[mask],(z[key][:,1,col]-z[key][:,0,col])[mask]*scale,label=model+(' wind +0.1' if args.final else ' wind +0.5'))
            ax.plot(t[mask],(z[key][:,10,col]-z[key][:,11,col])[mask]*scale/2,ls='--',label=model+(' diff ±1 half difference' if args.final else ' diff ±5 half difference'))
    for ax,title in zip(axes.ravel(),['Own candidate Δz (m)','Own candidate Vx (m/s)','Own candidate nose-up pitch (deg)','Common transient Δp (rad/s)','Common transient Δr (rad/s)','Common transient Δroll (deg)']):
        ax.set_title(title);ax.set_xlabel('Time (s)');ax.grid();ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(args.input/('final_trajectories.png' if args.final else 'trajectories.png'),dpi=170)
    (args.input/('final_analysis.json' if args.final else 'analysis.json')).write_text(json.dumps(result,indent=2))
    for model,r in result.items():
        print(model,json.dumps(r['sustain']))
        for label in ['common','own']:
            print(label,'checks',r[label]['checks'])
            for h,v in r[label]['horizons'].items():
                print(h,'wind p/r',np.array(v['increments']['wind+.5']['omega'])[[0,2]],'eff',v['effect_pr_per_deg'],'comp',v['compensation_rudder_diff_deg'])

if __name__=='__main__':main()
