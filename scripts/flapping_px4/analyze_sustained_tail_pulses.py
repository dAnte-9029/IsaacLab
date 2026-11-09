"""Offline common-O-frame finite pulse responses; no simulation or tuning."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flapping_bot.path_tracking.unified_trajectory import Trajectory, TrajectorySpec
from flapping_bot.analysis.unified_retrospective import rotate_inverse
from run_sustained_tail_pulses import GROUPS


def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def save_csv(p,data):
    with p.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(data);w.writerows(zip(*data.values()))
def extract(folder, normal, up, position):
    c=read(folder/'contract.json');slot=c['slot'];ids=c['joint_ids']
    entries=[json.loads(s) for s in (folder/'trace.jsonl').read_text().splitlines()]
    controls=[e['controls'][slot] for e in entries]
    x=np.array([e['root_state'] for e in controls]);v=x[:,7:10]
    path=Trajectory(TrajectorySpec(**c['tasks'][slot]));q=path.query(np.array([e['projection']['progress'] for e in controls]))
    ownerr=x[:,:3]-q['position']
    chi=np.unwrap(np.arctan2(v[:,1],v[:,0]));vh=np.linalg.norm(v[:,:2],axis=1)
    dot=np.gradient(chi,1/60,edge_order=2);acc=np.gradient(v,1/60,axis=0,edge_order=2)
    angles=np.rad2deg(np.unwrap(Rotation.from_quat(x[:,[4,5,6,3]]).as_euler('xyz'),axis=0))
    rates=np.rad2deg(rotate_inverse(x[:,3:7],x[:,10:13]))
    proj=lambda a,b:np.einsum('ij,ij->i',a,b)
    data=dict(time_s=np.arange(1,len(x)+1)/60-c['switch_after_step']/60,
        vy=proj(v,normal),py=proj(x[:,:3]-position,normal),vz=proj(v,up),pz=proj(x[:,:3]-position,up),
        ey=proj(ownerr,q['normal']),ez=proj(ownerr,q['up']),vy_own=proj(v,q['normal']),
        an=proj(acc,normal),roll=angles[:,0],roll_rate=rates[:,0],yaw_rate=rates[:,2],
        chi=np.rad2deg(chi),chi_rate=dot,kappa=np.divide(dot,vh,out=np.zeros_like(dot),where=vh>0.1),vh=vh,
        phase=np.array([e['phase'] for e in controls]))
    data['required_an_approx']=vh**2*np.gradient(np.unwrap(q['heading']),np.array([e['projection']['progress'] for e in controls]))
    raw=np.array([e['raw_command'] for e in controls]);target=np.rad2deg([e['targets_rad'][slot] for e in entries])
    jp=np.rad2deg(np.array([e['joint_pos'] for e in controls])[:,ids]);jv=np.rad2deg(np.array([e['joint_vel'] for e in controls])[:,ids])
    for j,name in enumerate(('frequency','rudder','left','right')):data[name+'_raw']=raw[:,j]
    for j,name in enumerate(('rudder','left','right')):
        data[name+'_target']=target[:,j];data[name+'_joint']=jp[:,j];data[name+'_velocity']=jv[:,j]
    for name,key in [('actual_hz','_freq'),('requested_hz','_requested_frequency_hz'),('governor_hz','_applied_frequency_hz')]:
        data[name]=np.array([e['actuator_state'][key] for e in controls])
    data['clipped']=np.any(abs(raw)>1,axis=1).astype(float)
    # Centered 18-sample descriptive average, never used to define raw effects/onset.
    for key in ('an','chi_rate','kappa'):
        data[key+'_mean300ms']=np.array([data[key][max(0,k-8):min(len(x),k+10)].mean() for k in range(len(x))])
    return {k:np.asarray(v,dtype=np.float64) for k,v in data.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);a=p.parse_args();root=a.input
    assert read(root/'status.json')['status']=='completed'
    out=root/'analysis';out.mkdir(exist_ok=False)
    metrics=[];audit={};onsets=[]
    fields=('vy','py','vz','pz','ey','ez','an','roll','roll_rate','yaw_rate','chi','chi_rate','kappa','actual_hz','phase')
    for start in (120,144):
        folder=root/f'T{start}_O';c=read(folder/'contract.json');slot=c['slot']
        e=[json.loads(s) for s in (folder/'trace.jsonl').read_text().splitlines()]
        path=Trajectory(TrajectorySpec(**c['tasks'][slot]));q=path.query(np.array([r['controls'][slot]['projection']['progress'] for r in e]))
        ds={g:extract(root/f'T{start}_{g}',q['normal'],q['up'],q['position']) for g in GROUPS}
        for g,d in ds.items():
            save_csv(out/f'T{start}_{g}.csv',d)
            s=read(root/f'T{start}_{g}/summary.json')
            audit[f'T{start}_{g}']=dict(steps=s['completed_control_steps'],historical_max=max(s['historical_max_abs'].values()),prefix_max=max(s['common_prefix_max_abs'].values(),default=0),observation_max=s['historical_observation_max_abs'],terminated=s['terminated'])
        for direction in 'RSD':
            plus,minus,base=ds[direction+'+'],ds[direction+'-'],ds['O']
            response={'time_s':base['time_s']}
            for key in fields:
                response['J_'+key]=(plus[key]-minus[key])/.2
                response['N_'+key]=(plus[key]+minus[key])/2-base[key]
                response['plus_minus_'+key]=plus[key]-minus[key]
                response['plus_O_'+key]=plus[key]-base[key];response['minus_O_'+key]=minus[key]-base[key]
            save_csv(out/f'T{start}_{direction}_sensitivity.csv',response)
            for low,high in ((0,18),(18,30),(30,60)):
                sl=slice(start+low,start+high);end=start+high-1
                row=dict(start=start,direction=direction,window=[low/60,high/60],excitation_symmetric=not bool(plus['clipped'][start:start+18].any() or minus['clipped'][start:start+18].any()))
                for key,arr in response.items():
                    if key=='time_s':continue
                    row[key+'_end']=float(arr[end]);row[key+'_mean']=float(arr[sl].mean());row[key+'_max_abs']=float(abs(arr[sl]).max())
                for g,d in ((direction+'+',plus),(direction+'-',minus),('O',base)):
                    row[g]={key:float(d[key][end]) for key in fields}
                metrics.append(row)
            # Operational latency thresholds, descriptive not significance tests.
            # Three consecutive post-step samples above threshold; no interpolation.
            for key,threshold in [('vy',.01),('an',.1),('chi_rate',.01),('roll',.1),('yaw_rate',1.),('vz',.01),('rudder_joint',.1),('left_joint',.1),('right_joint',.1)]:
                gap=abs(plus[key]-minus[key]);hits=[k for k in range(start,start+58) if np.all(gap[k:k+3]>=threshold)]
                onsets.append(dict(start=start,direction=direction,metric=key,pair_threshold=threshold,first_three_sample_s=None if not hits else (hits[0]+1-start)/60))
            panels=['rudder_raw','left_raw','right_raw','rudder_target','left_target','right_target','rudder_joint','left_joint','right_joint','rudder_velocity','left_velocity','right_velocity','an','vy','py','chi_rate','kappa','roll','roll_rate','yaw_rate','vz','pz','actual_hz','phase']
            fig,axes=plt.subplots(8,3,figsize=(15,22),layout='constrained')
            for ax,key in zip(axes.flat,panels):
                for g in ('O',direction+'+',direction+'-'):
                    d=ds[g];sel=d['time_s']>=-.1;ax.plot(d['time_s'][sel],d[key][sel],label=g)
                ax.axvspan(0,.3,alpha=.1,color='gray');ax.set_title(key);ax.grid(alpha=.2)
            axes.flat[0].legend();fig.suptitle(f'T{start} {direction}: raw input -> actuator -> motion (deg, deg/s, SI)');fig.savefig(out/f'T{start}_{direction}.png',dpi=120);plt.close(fig)
        fig,axes=plt.subplots(3,2,figsize=(12,10),layout='constrained')
        for ax,key,unit in zip(axes.flat,('vy','an','chi_rate','kappa','vz','roll'),('m/s','m/s2','rad/s','1/m','m/s','deg')):
            for color,direction in zip(('C0','C1','C2'),'RSD'):
                d=(ds[direction+'+'][key]-ds[direction+'-'][key])/.2
                ax.plot(ds['O']['time_s'][start-1:],d[start-1:],label=direction,color=color)
                if key in ('an','chi_rate','kappa'):
                    avg=(ds[direction+'+'][key+'_mean300ms']-ds[direction+'-'][key+'_mean300ms'])/.2
                    ax.plot(ds['O']['time_s'][start-1:],avg[start-1:],ls='--',alpha=.6,color=color)
            ax.axvspan(0,.3,alpha=.1,color='gray');ax.set_title('J_'+key+' ['+unit+'/normalized action]');ax.set_xlabel('seconds after pulse start');ax.grid(alpha=.2)
        axes.flat[0].legend();fig.suptitle(f'T{start}: solid raw, dashed 300ms descriptive mean');fig.savefig(out/f'T{start}_sensitivities.png',dpi=150);plt.close(fig)
    write(out/'results.json',dict(audit=audit,metrics=metrics,onsets=onsets,notes='Common O normal/up/position; world velocity differentiated before projection. 60Hz centered derivatives; 300ms descriptive mean includes neighboring times and is not causal latency. Curvature masked to zero below 0.1m/s. Latencies are three consecutive raw samples above declared descriptive thresholds.'))
    print('TAIL_ANALYSIS_PASSED')
if __name__=='__main__':main()
