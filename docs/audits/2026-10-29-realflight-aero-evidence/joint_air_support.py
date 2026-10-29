"""Supplement joint sensor-TAS / candidate-beta / p / r / f support, no fits."""
from pathlib import Path
import sys
import numpy as np,pandas as pd
from scipy.spatial.transform import Rotation
from pyulog import ULog
P=Path(__file__).resolve().parent/'results'
D=Path('/home/zn/flap-system-identification/dataset/velocity_rate_multistep_v1')
fl=pd.read_csv(D/'flights.csv');fl=fl[fl.split.isin(['train','validation'])]
cols=['flight_id','timestamp_us','segment_id','valid_core','valid_frequency','flap_frequency_hz']+['attitude_q_'+a for a in 'wxyz']+['velocity_ned_m_s_'+a for a in 'xyz']+['angular_velocity_body_rad_s_'+a for a in ['x','z']]
rows=[];coverage=[]
def asof(d,t,key,age):
    if d is None or key not in d:return np.full(len(t),np.nan)
    ts=np.asarray(d['timestamp'],dtype=np.int64);o=np.argsort(ts,kind='stable');ts=ts[o];i=np.searchsorted(ts,t,'right')-1;j=np.maximum(i,0)
    z=np.asarray(d[key],dtype=float)[o][j];z[(i<0)|(t-ts[j]>age)]=np.nan;return z
if '--summary-only' in sys.argv:
    j=pd.read_csv(P/'air_motion_joint.csv');c=pd.read_csv(P/'air_motion_joint_coverage.csv')
else:
    for split in ['train','validation']:
        data=pd.read_parquet(D/f'samples_{split}.parquet',columns=cols)
        for fid,g in data.groupby('flight_id',sort=False):
            m=fl.set_index('flight_id').loc[fid];g=g.sort_values('timestamp_us');t=g.timestamp_us.to_numpy(np.int64);dt=np.r_[np.diff(t)/1e6,0.];vld=g.valid_core.to_numpy(bool);seg=g.segment_id.to_numpy();cap=.04 if m.sampling_group=='main_50hz' else .16
            wt=np.where(vld&np.r_[vld[1:],False]&np.r_[seg[1:]==seg[:-1],False]&(dt>0)&(dt<=cap),dt,0.)
            u=ULog(str(Path('/home/zn/QgcLogs')/m.log_path),message_name_filter_list=['wind','airspeed_validated']);dd={(d.name,d.multi_id):d.data for d in u.data_list};air=dd.get(('airspeed_validated',0));wind=dd.get(('wind',0))
            tas=asof(air,t,'true_airspeed_m_s',200000);src=asof(air,t,'airspeed_source',200000);tas[~np.isin(src,[1,2,3])|(tas<=0)]=np.nan
            q=g[['attitude_q_'+a for a in 'wxyz']].to_numpy(float);bad=~np.isfinite(q).all(1)|(np.linalg.norm(q,axis=1)==0);q[bad]=[1,0,0,0]
            vel=g[['velocity_ned_m_s_'+a for a in 'xyz']].to_numpy(float);vel-=np.column_stack([asof(wind,t,'windspeed_north',500000),asof(wind,t,'windspeed_east',500000),np.zeros(len(t))]);vb=Rotation.from_quat(q[:,[1,2,3,0]]).inv().apply(vel);beta=np.degrees(np.arctan2(vb[:,1],np.hypot(vb[:,0],vb[:,2])));beta[np.linalg.norm(vb,axis=1)<3]=np.nan
            f=g.flap_frequency_hz.to_numpy(float);f[~g.valid_frequency.to_numpy(bool)]=np.nan
            vals=[tas,abs(g.angular_velocity_body_rad_s_x.to_numpy()),abs(g.angular_velocity_body_rad_s_z.to_numpy()),f,abs(beta)]
            bins=[[0,4,6,8,10,12,np.inf],[0,.5,1,2,np.inf],[0,.25,.5,1,np.inf],[0,2,3,4,5,np.inf],[0,5,10,20,90.0001]]
            names=['sensor_tas_bin','abs_p_bin','abs_r_bin','frequency_bin','candidate_abs_beta_bin']
            tab=pd.DataFrame({name:pd.cut(x,b,right=False).astype(str) for name,x,b in zip(names,vals,bins)});tab['seconds']=wt
            base=dict(flight_id=fid,split=split,date=m.flight_date,group=m.sampling_group)
            measured=np.isfinite(np.column_stack(vals[:4])).all(1);candidate=measured&np.isfinite(beta)
            coverage.append({**base,'valid_s':wt.sum(),'tas_p_r_f_support_s':wt[measured].sum(),'with_candidate_beta_support_s':wt[candidate].sum()})
            for include_beta,mask in [(False,measured),(True,candidate)]:
                keys=names if include_beta else names[:4]
                for key,h in tab[mask&(wt>0)].groupby(keys):rows.append({**base,'kind':'candidate_beta_joint' if include_beta else 'sensor_motion_joint',**dict(zip(keys,key)),'seconds':h.seconds.sum()})
        print('finished',split,flush=True)
    j=pd.DataFrame(rows);c=pd.DataFrame(coverage);j.to_csv(P/'air_motion_joint.csv',index=False);c.to_csv(P/'air_motion_joint_coverage.csv',index=False)
for kind,field in [('sensor_motion_joint','tas_p_r_f_support_s'),('candidate_beta_joint','with_candidate_beta_support_s')]:assert np.allclose(j[j.kind==kind].groupby('flight_id').seconds.sum().reindex(c.flight_id,fill_value=0).to_numpy(),c[field].to_numpy())
print(c.groupby(['split','group']).sum(numeric_only=True))
print(j.query("kind=='sensor_motion_joint'").groupby(['split','group','sensor_tas_bin','abs_p_bin','abs_r_bin','frequency_bin']).seconds.sum().sort_values(ascending=False).head(12))
print('JOINT_CHECKS_PASSED')
