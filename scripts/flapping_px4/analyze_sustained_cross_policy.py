"""Read-only analysis of the eight counterfactual trajectories (no simulation)."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flapping_bot.path_tracking.unified_trajectory import Trajectory,TrajectorySpec
from flapping_bot.analysis.unified_retrospective import rotate_inverse


def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def wrap(x):return (x+180)%360-180


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',type=Path,required=True)
    args=ap.parse_args();root=args.input.resolve()
    assert read(root/'status.json')['status']=='completed'
    out=root/'analysis';out.mkdir(exist_ok=False)
    traces={};tables=[];audits={};starts=[]
    for start in (120,144):
        for group in 'ABCD':
            folder=root/f'T{start}_{group}';contract=read(folder/'contract.json');summary=read(folder/'summary.json')
            assert summary['passed'] and summary['policies_unchanged'] and summary['optimizer_updates']==0
            slot=contract['slot'];ids=contract['joint_ids'];path=Trajectory(TrajectorySpec(**contract['tasks'][slot]))
            entries=[json.loads(s) for s in (folder/'trace.jsonl').read_text().splitlines()]
            assert entries[-1]['step']==start+60 or summary['terminated'] is not None
            assert all(e['group']==('D' if e['step']<=start else group) for e in entries)
            controls=[e['controls'][slot] for e in entries]
            x=np.array([c['root_state'] for c in controls]);progress=np.array([c['projection']['progress'] for c in controls]);q=path.query(progress)
            basis=np.stack([q[k] for k in ('tangent','normal','up')],1)
            err=np.einsum('nij,nj->ni',basis,x[:,:3]-q['position']);v=np.einsum('nij,nj->ni',basis,x[:,7:10])
            euler=Rotation.from_quat(x[:,[4,5,6,3]]).as_euler('xyz',degrees=True)
            rates=np.rad2deg(rotate_inverse(x[:,3:7],x[:,10:13]))
            joint=np.rad2deg(np.array([c['joint_pos'] for c in controls])[:,ids]);jointvel=np.rad2deg(np.array([c['joint_vel'] for c in controls])[:,ids])
            raw=np.array([c['raw_command'] for c in controls]);clipped=np.array([c['command'] for c in controls]);applied=np.array([c['applied'] for c in controls])
            target=np.rad2deg(np.array([e['targets_rad'][slot] for e in entries]))
            data=dict(step=np.arange(1,len(entries)+1),time_s=np.arange(1,len(entries)+1)/60,
                relative_time_s=np.arange(1,len(entries)+1)/60-start/60,progress_m=progress,
                e_y_m=err[:,1],e_z_m=err[:,2],v_y_mps=v[:,1],de_y_dt_mps=np.gradient(err[:,1],1/60),
                course_error_deg=wrap(np.rad2deg(np.arctan2(x[:,8],x[:,7])-q['heading'])),
                yaw_error_deg=wrap(euler[:,2]-np.rad2deg(q['heading'])),roll_deg=euler[:,0],pitch_deg=euler[:,1],yaw_deg=euler[:,2],
                roll_rate_deg_s=rates[:,0],yaw_rate_deg_s=rates[:,2],phase_rad=np.array([c['phase'] for c in controls]))
            for name,col in [('requested_hz','_requested_frequency_hz'),('governor_hz','_applied_frequency_hz'),('actual_hz','_freq')]:
                data[name]=np.array([c['actuator_state'][col] for c in controls])
            data['raw_frequency_hz']=(raw[:,0]+1)*2.5;data['clipped_frequency_hz']=(clipped[:,0]+1)*2.5
            for j,name in enumerate(['frequency','rudder','left','right']):
                for kind,arr in [('raw',raw),('clipped',clipped),('applied',applied)]:data[f'{name}_{kind}_norm']=arr[:,j]
            for j,name in enumerate(['rudder','left','right']):
                data[f'{name}_actual_deg']=joint[:,j];data[f'{name}_velocity_deg_s']=jointvel[:,j];data[f'{name}_target_deg']=target[:,j]
            for name,arr in [('actual',joint),('target',target)]:
                data[f'symmetric_{name}_deg']=(arr[:,1]+arr[:,2])/2
                data[f'differential_{name}_deg']=(arr[:,1]-arr[:,2])/2
            with (out/f'T{start}_{group}.csv').open('w',newline='') as f:
                w=csv.writer(f);w.writerow(data);w.writerows(zip(*data.values()))
            traces[start,group]=data
            audits[f'T{start}_{group}']=dict(historical_max=max(summary['historical_max_abs'].values()),
                observation_max=summary['historical_observation_max_abs'],cross_process_max=max(summary['common_prefix_max_abs'].values(),default=None),
                terminated=summary['terminated'],steps=len(entries))
            for window in (30,60):
                end=start+window-1
                if end>=len(entries):
                    tables.append(dict(start_s=start/60,group=group,window_s=window/60,available=False,termination=summary['terminated']));continue
                row=dict(start_s=start/60,group=group,window_s=window/60,available=True)
                for key in ('v_y_mps','e_y_m','e_z_m','course_error_deg','yaw_error_deg','roll_deg','roll_rate_deg_s','yaw_rate_deg_s','actual_hz','progress_m'):
                    row[key]=float(data[key][end]);row['delta_'+key]=float(data[key][end]-data[key][start-1])
                sl=slice(start,end+1)
                row['raw_clip_fraction']=float(np.mean(np.any(abs(raw[sl])>1,axis=1)))
                row['governor_limited_fraction']=float(np.mean(abs(data['requested_hz'][sl]-data['governor_hz'][sl])>1e-5))
                row['mean_actual_hz']=float(data['actual_hz'][sl].mean())
                row['max_abs_e_z_m']=float(abs(data['e_z_m'][sl]).max())
                row['max_abs_roll_deg']=float(abs(data['roll_deg'][sl]).max())
                row['roll_min_deg']=float(data['roll_deg'][sl].min());row['roll_max_deg']=float(data['roll_deg'][sl].max())
                tables.append(row)
            if group=='D':starts.append(dict(start_s=start/60,**{k:float(v[start-1]) for k,v in data.items()}))
    effects=[]
    for start in (2.,2.4):
        for window in (.5,1.):
            rows={r['group']:r for r in tables if r['start_s']==start and r['window_s']==window}
            if not all(r['available'] for r in rows.values()):continue
            for key in rows['A']:
                if key in ('start_s','window_s','available','group'):continue
                a,b,c,d=[rows[g][key] for g in 'ABCD']
                effects.append(dict(start_s=start,window_s=window,metric=key,delta_T=((a-b)+(c-d))/2,
                    delta_F=((a-c)+(b-d))/2,delta_FT=a-b-c+d,
                    T_at_F25=a-b,T_at_F150=c-d,F_at_T25=a-c,F_at_T150=b-d))
    write(out/'results.json',dict(schema='sustained_cross_policy_analysis_v1',audit=audits,starts=starts,metrics=tables,factor_effects=effects,
        derivative_note='de_y/dt includes moving path frame; not identical to v_y',statistical_scope='deterministic local contrasts; no p values'))
    panels={
        'response':['v_y_mps','e_y_m','e_z_m','de_y_dt_mps','course_error_deg','yaw_error_deg','roll_deg','roll_rate_deg_s','yaw_rate_deg_s','progress_m'],
        'frequency':['frequency_raw_norm','raw_frequency_hz','clipped_frequency_hz','requested_hz','governor_hz','actual_hz','phase_rad','frequency_applied_norm'],
        'tails':['rudder_raw_norm','left_raw_norm','right_raw_norm','rudder_target_deg','left_target_deg','right_target_deg','rudder_actual_deg','left_actual_deg','right_actual_deg','symmetric_actual_deg','differential_actual_deg','rudder_velocity_deg_s']}
    for start in (120,144):
        for title,fields in panels.items():
            fig,axs=plt.subplots((len(fields)+1)//2,2,figsize=(12,2.5*((len(fields)+1)//2)),layout='constrained')
            for ax,key in zip(axs.flat,fields):
                for group in 'ABCD':
                    d=traces[start,group];sel=d['relative_time_s']>=-.1
                    ax.plot(d['relative_time_s'][sel],d[key][sel],label=group)
                ax.axvline(0,color='k',lw=.6);ax.axvline(.5,color='k',ls=':',lw=.6)
                ax.set_title(key);ax.set_xlabel('seconds after intervention');ax.grid(alpha=.2)
            axs.flat[0].legend();fig.suptitle(f'Sustained T={start/60:.1f}s: '+title)
            fig.savefig(out/f'T{start}_{title}.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(passed=True,output=str(out),trajectories=len(traces),windows=len(tables))))

if __name__=='__main__':main()
