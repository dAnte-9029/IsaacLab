import json,csv,math
from pathlib import Path
import numpy as np
import torch
from flapping_bot.physics.tail_aero import TailAeroCfg,TailAeroModel
root=Path(__file__).resolve().parents[2];out=root/'docs/results/tail_v2_v1';logs=root/'logs/tail_v2_v1'
rows=[]
for backend in ('legacy','v2'):
    folder=logs/backend
    if not (folder/'control_results.json').exists():continue
    protocol=json.loads((folder/'protocol.json').read_text());limits=np.array(protocol['tail_limits_rad'])
    for case in ('hold','recovery','tracking'):
        d=np.load(folder/(case+'.npz')); e=d['euler'][:,0,1]-d['target'][:,0];q=d['omega'][:,0,1]
        # resolved joint order is [wing R, wing L, rudder, tail L, tail R] from protocol
        names=protocol['joint_names']; tail=[i for i,n in enumerate(names) if 'tail' in n.lower()]
        # Use explicit names stored by the plant; no assumed actuator order.
        jp=d['joint'][:,0];sat=np.zeros(len(jp),dtype=bool)
        for i in tail:
            sat|=(jp[:,i]<limits[1,0]+math.radians(.5))|(jp[:,i]>limits[1,1]-math.radians(.5))
        row=dict(backend=backend,case=case,pitch_rmse_deg=float(np.sqrt(np.mean(e*e))*180/math.pi),q_rms_rad_s=float(np.sqrt(np.mean(q*q))),max_error_deg=float(np.max(abs(e))*180/math.pi),height_change_m=float(d['position'][-1,0,2]-d['position'][0,0,2]),vertical_velocity_final_mps=float(d['velocity'][-1,0,2]),saturation_fraction=float(sat.mean()),recovery_time_s='not achieved' if np.any(abs(e[-480:])>math.radians(3)) else 'within final second',wing_My_mean_Nm=float(d['wing_moment'][:,0,1].mean()),tail_My_mean_Nm=float(d['tail_moment'][:,0,1].mean()))
        model=TailAeroModel(TailAeroCfg(backend=backend,fixed_horizontal_effectiveness=.5,elevon_effectiveness=1.2),'cpu')
        v=torch.tensor(d['air'][:,0]);om=torch.tensor(d['omega'][:,0]);j=torch.tensor(jp)
        surfaces=model.compute_surface_results(root_lin_vel_b=v,root_ang_vel_b=om,left_elevon_rad=j[:,names.index('left_tail')],right_elevon_rad=j[:,names.index('right_tail')],rudder_rad=j[:,names.index('rudder')],base_com_pos_b=v.new_tensor([[-.12154,.00541,-.01298]]).expand_as(v))
        row['CL_min']=min(float(r.lift_coefficient.min()) for r in surfaces)
        row['CL_max']=max(float(r.lift_coefficient.max()) for r in surfaces)
        row['alpha_abs_max_deg']=max(float(r.alpha_rad.abs().max())*180/math.pi for r in surfaces)
        row['actual_tail_abs_max_deg']=float(np.max(abs(jp[:,tail]))*180/math.pi)
        row['coefficient_reconstruction']='nominal base COM; postprocessed from recorded state'
        rows.append(row)
with (out/'closed_loop_summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
historical=[]
for start in (120,144):
    d=np.load(root/f'docs/results/tail_moment_attribution_v1/T{start}_O_physics.npz')
    idx=min(start*8,len(d['air'])-1)
    for backend in ('legacy','v2'):
        model=TailAeroModel(TailAeroCfg(backend=backend,fixed_horizontal_effectiveness=.5,elevon_effectiveness=1.2),'cpu')
        def m(delta):
            v=torch.tensor(d['air'][idx:idx+1]);o=torch.tensor(d['omega'][idx:idx+1]);z=v[:,0]*0
            return float(model.compute_wrench(root_lin_vel_b=v,root_ang_vel_b=o,elevator_rad=z+delta,rudder_rad=z,base_com_pos_b=torch.tensor(d['base_com_b'][idx:idx+1]))[1][0,1])
        historical.append(dict(state=f'T{start}',backend=backend,comparison='common historical air/omega/COM; symmetric joint sweep about zero',delta_difference_deg=6.82,moment_difference_Nm=m(math.radians(3.41))-m(-math.radians(3.41))))
with (out/'historical_authority.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(historical[0]));w.writeheader();w.writerows(historical)
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(2,1,figsize=(9,7))
for backend in ('legacy','v2'):
    d=np.load(logs/backend/'hold.npz');t=d['t'][:,0]
    axs[0].plot(t,-np.rad2deg(d['euler'][:,0,1]),label=backend)
    axs[1].plot(t,d['position'][:,0,2]-d['position'][0,0,2],label=backend)
for ax in axs:ax.legend();ax.grid();ax.set_xlabel('time (s)')
axs[0].set_ylabel('nose-up pitch (deg)');axs[1].set_ylabel('height change (m)');fig.tight_layout();fig.savefig(out/'closed_loop.png')
print(json.dumps(rows,indent=2));print(historical)
