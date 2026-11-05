"""Summarize complete native traces, produce plots and revisable evidence rules."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flapping_bot.path_tracking.unified_trajectory import Trajectory,TrajectorySpec
from flapping_bot.path_tracking.trajectory_feasibility import features


def analyze(folder):
    protocol=json.loads((folder/'protocol.json').read_text())
    if not (folder/'completed.json').exists(): raise RuntimeError(f'Incomplete run: {folder}')
    tr=np.load(folder/'trace.npz'); gate=protocol['acceptance']; results=[]
    fig,axes=plt.subplots(3,2,figsize=(12,10))
    for i,case in enumerate(protocol['cases']):
        path=Trajectory(TrajectorySpec(**case['trajectory']));t=tr['t'][:,i]
        err=tr['position'][:,i]-tr['reference'][:,i]
        horizontal=np.linalg.norm(err[:,:2],axis=1); vertical=np.abs(err[:,2])
        speed=np.linalg.norm(tr['velocity'][:,i],axis=1)
        angles=np.rad2deg(tr['euler'][:,i]);omega=np.linalg.norm(tr['omega'][:,i],axis=1)
        violation=(horizontal>gate['horizontal_max_m'])|(vertical>gate['vertical_max_m'])|(np.abs(angles[:,0])>gate['roll_max_deg'])|(np.abs(angles[:,1])>gate['pitch_max_deg'])|(omega>gate['omega_max_rad_s'])|(speed<gate['speed_min_mps'])|(speed>gate['speed_max_mps'])
        violation|=tr['position'][:,i,2]<1.
        complete=tr['progress'][:,i]>=gate['completion_fraction']*path.spec.length_m
        # Evaluate up to first mission completion, otherwise entire budget. Never erase earlier violations.
        end=int(np.flatnonzero(complete)[0]) if complete.any() else len(t)-1
        sl=slice(0,end+1);last=(t>=t[end]-1)&(t<=t[end])
        recovery=(float(np.mean(horizontal[last]))<=gate['recovery_horizontal_m'] and float(np.mean(vertical[last]))<=gate['recovery_vertical_m'] and abs(float(np.mean(tr['velocity'][last,i,2])))<=gate['recovery_vertical_speed_mps'])
        valid=complete.any() and not violation[sl].any() and recovery
        raw=tr['requested_physical'][:,i]; lim=np.rad2deg(protocol['tail_limits_rad'])
        clipped=(raw[:,0]<0)|(raw[:,0]>5)|np.any((raw[:,1:]<lim[:,0])|(raw[:,1:]>lim[:,1]),axis=1)
        bad=np.flatnonzero(violation[sl]); before=int(bad[0]) if len(bad) else end
        heading=np.unwrap(np.arctan2(tr['velocity'][:,i,1],tr['velocity'][:,i,0]))
        # Wingbeat-averaged realized heading derivative; no claim of instantaneous turn equilibrium.
        width=max(3,int(round(1/protocol['initial'][2]/np.median(np.diff(t)))))
        hs=np.convolve(heading,np.ones(width)/width,'valid')
        ss=np.convolve(speed,np.ones(width)/width,'valid')
        kk=np.gradient(hs,np.median(np.diff(t)))/np.maximum(ss,1e-9)
        r=dict(name=case['name'],run=folder.name,backend=protocol['backend'],status='VERIFIED FEASIBLE' if valid else 'NOT FOUND',
            completed=bool(complete.any()),completion_time_s=float(t[end]),progress_m=float(tr['progress'][end,i]),
            first_violation_s=float(t[bad[0]]) if len(bad) else None,
            first_violation_state=dict(horizontal_m=float(horizontal[before]),vertical_m=float(vertical[before]),euler_deg=angles[before].tolist(),omega_rad_s=float(omega[before]),speed_mps=float(speed[before])) if len(bad) else None,progress_before_violation_m=float(tr['progress'][before,i]),
            horizontal_max_m=float(horizontal[sl].max()),vertical_max_m=float(vertical[sl].max()),
            horizontal_rms_m=float(np.sqrt(np.mean(horizontal[sl]**2))),vertical_rms_m=float(np.sqrt(np.mean(vertical[sl]**2))),
            roll_max_deg=float(np.abs(angles[sl,0]).max()),pitch_max_deg=float(np.abs(angles[sl,1]).max()),omega_max_rad_s=float(omega[sl].max()),
            speed_range_mps=[float(speed[sl].min()),float(speed[sl].max())],height_change_m=float(tr['position'][end,i,2]-20),
            requested_clipped_fraction=float(clipped[sl].mean()),governor_limited_fraction=float(tr['governor_limited'][sl,i].mean()),
            actual_frequency_range_hz=[float(tr['frequency'][sl,i].min()),float(tr['frequency'][sl,i].max())],
            actual_joint_range_rad=[tr['joint'][sl,i].min(axis=0).tolist(),tr['joint'][sl,i].max(axis=0).tolist()],
            recovery_pass=bool(recovery),recovery_mean_vz_mps=float(tr['velocity'][last,i,2].mean()),
            geometry=features(path),normal_demand_max_mps2=float(tr['normal_accel_demand'][sl,i].max()))
        qref=path.query(tr['progress'][:,i])
        realized_kappa=np.gradient(heading,t)/np.maximum(speed,1e-9)
        realized_slope=np.arctan2(tr['velocity'][:,i,2],np.linalg.norm(tr['velocity'][:,i,:2],axis=1))
        r['plateau_response']={}
        for key,actual in [('kappa',realized_kappa),('gamma',realized_slope)]:
            magnitude=r['geometry'][key]
            for sign in [1,-1]:
                plateau=(sign*qref[key]>=magnitude-1e-8)&(np.abs(qref[key+'_prime'])<1e-8)&(t<=t[end])
                if magnitude>1e-10 and plateau.sum()>2:
                    r['plateau_response'][key+('_positive' if sign==1 else '_negative')]=dict(
                        samples=int(plateau.sum()),reference=float(sign*magnitude),actual_mean=float(actual[plateau].mean()),
                        speed_mean_mps=float(speed[plateau].mean()),frequency_mean_hz=float(tr['frequency'][plateau,i].mean()))
        results.append(r)
        label=case['name']
        axes[0,0].plot(t,tr['position'][:,i,2]-20,label=label)
        axes[0,1].plot(t,speed,label=label)
        axes[1,0].plot(t,horizontal,label=label);axes[1,1].plot(t,angles[:,0],label=label)
        axes[2,0].plot(t,tr['normal_accel_demand'][:,i],label=label)
        axes[2,1].plot(t[:len(kk)],kk,label=label)
    for ax,title in zip(axes.flat,['Height change (m)','Actual speed (m/s)','Horizontal error (m)','Roll (deg)','Reference normal demand at actual speed (m/s2)','Wingbeat-smoothed realized d(heading)/ds (1/m)']):
        ax.set_title(title);ax.set_xlabel('Time (s)');ax.grid(alpha=.3)
    axes[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(folder/'response.png',dpi=140);plt.close(fig)
    (folder/'results.json').write_text(json.dumps(results,indent=2))
    return protocol,results


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    rules=dict(schema_version=1,policy='Exact successful recipe and conditions only; no peak-box extrapolation. NOT FOUND lowers priority, never hard excludes.',
        plants={b:dict(limits={k:None for k in ['kappa','gamma','kappa_prime','gamma_prime','sustained_distance','net_climb','coupling']}) for b in ['L3D','I3D']},evidence=[])
    all_results=[]
    # Limits stay unknown globally; individual successful recipes carry measured limits.
    rules['feature_priority_policy']=dict(status='UNTESTED / UNCERTAIN',comparison='componentwise to one successful recipe with identical conditions',fields=['kappa','gamma','kappa_prime','gamma_prime','length_m','kappa_max_active_length_m','gamma_max_active_length_m','net_climb_m','climb_excursion_m','coupled_length_m','coupling_abs_integral'],hard_exclusion=False)
    for folder in sorted(a.root.iterdir()):
        if not (folder/'completed.json').exists(): continue
        protocol,results=analyze(folder);all_results+=results
        for c,r in zip(protocol['cases'],results):
            path=Trajectory(TrajectorySpec(**c['trajectory']))
            conditions={k:protocol[k] for k in ['initial','initial_height_m','initial_vertical_speed','wind','physics_dt','control_dt','sections','twist_deg','plant']}
            conditions['controller']=c.get('gains',{});conditions['acceptance']=protocol['acceptance']
            conditions.update(controller_version='phase2_feedback_v1',seed=protocol['seed'],initial_phase_rad=0.,initial_omega_rad_s=[0.,0.,0.])
            rules['evidence'].append(dict(backend=protocol['backend'],trajectory=path.spec.to_dict(),conditions=conditions,
                status=r['status'],source=str(folder/'results.json'),features=r['geometry'],observed_speed_range_mps=r['speed_range_mps']))
    (a.root/'screening_rules.json').write_text(json.dumps(rules,indent=2))
    (a.root/'summary.json').write_text(json.dumps(all_results,indent=2))
    print(json.dumps([{k:r[k] for k in ['backend','name','status','progress_m','first_violation_s','vertical_max_m','horizontal_max_m','speed_range_mps','recovery_mean_vz_mps']} for r in all_results],indent=2))
if __name__=='__main__': main()
