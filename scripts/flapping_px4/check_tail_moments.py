"""Focused independent export checks, including whole-system linear momentum."""
import argparse,json,csv
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);a=p.parse_args();root=a.input
meta=json.loads((root/'parameters.json').read_text());mass=np.array(meta['masses']);dt=meta['dt'];checks={};vertical=[]
for start in (120,144):
    data={g:dict(np.load(root/f'T{start}_{g}_physics.npz')) for g in ('O','S+','S-')};j=start*8
    for g,d in data.items():
        R=Rotation.from_quat(d['quat'][:,[1,2,3,0]]).as_matrix()
        momentum=np.sum(d['body_velocity']*mass[None,:,None],axis=1)
        force=np.einsum('nij,nj->ni',R,d['force_b'])+np.array([0,0,-9.81*mass.sum()])
        impulse=force[j:-1].sum(0)*dt;error=momentum[-1]-momentum[j]-impulse
        checks[f'T{start}_{g}_linear_impulse_error_Ns']=error.tolist()
        assert np.linalg.norm(error)<.03,(g,error)
        for key in ('frequency','phase'):
            assert np.array_equal(d[key],data['O'][key]),key
        torque=np.cross(d['surface_aerodynamic_center_b_from_com'],d['surface_force_b'])
        assert np.max(abs(torque-d['surface_moment_b_about_com']))<2e-6
        assert np.max(abs(d['surface_force_b'].sum(1)-d['tail_force_b']))<2e-6
    for k in range(480):
        idx=j+k
        vertical.append(dict(start=start,time_s=(k-1)*dt,delta_root_z=float(data['S+']['position'][idx,2]-data['S-']['position'][idx,2]),delta_base_com_z=float(data['S+']['body_com'][idx,0,2]-data['S-']['body_com'][idx,0,2]),delta_base_com_vz=float(data['S+']['velocity'][idx,2]-data['S-']['velocity'][idx,2])))
# Check decomposition independently against saved query table.
for r in csv.DictReader((root/'state_swap.csv').open()):
    assert abs(float(r['direct_Nm'])+float(r['state_feedback_Nm'])-float(r['total_Nm']))<1e-12
for start in (120,144):
    rows=list(csv.DictReader((root/f'T{start}_differences.csv').open()))
    signed=sum(float(r['total_My_Nm']) for r in rows)*dt
    saved=next(r for r in csv.DictReader((root/'impulses.csv').open()) if int(r['start'])==start and r['component']=='total')
    assert abs(signed-float(saved['signed_Nms']))<1e-6
checks['passed']=True
(root/'focused_checks.json').write_text(json.dumps(checks,indent=2))
with (root/'vertical_reference.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=vertical[0],lineterminator="\n");w.writeheader();w.writerows(vertical)
print(json.dumps(checks,indent=2))
