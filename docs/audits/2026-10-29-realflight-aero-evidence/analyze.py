"""Read-only train/validation coverage; no fitting, test loading or force labels."""
from pathlib import Path
import json, sys
import faulthandler
faulthandler.enable()
print("starting imports",flush=True)
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
from pyulog import ULog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/home/zn/flap-system-identification')
DATA=ROOT/'dataset/velocity_rate_multistep_v1'
OUT=Path(__file__).resolve().parent/'results'
OUT.mkdir(exist_ok=True)
registry=ROOT/'.worktrees/velocity-rate-dataset-v1/configs/data/velocity_rate_dataset_registry.yaml'
import yaml
reg=yaml.safe_load(registry.read_text()); assert reg['default_dataset_id']=='velocity_rate_multistep_v1'
manifest=json.loads((DATA/'dataset_manifest.json').read_text())
assert manifest['dataset_id']==reg['default_dataset_id']
flights=pd.read_csv(DATA/'flights.csv'); flights=flights[flights.split.isin(['train','validation'])].copy()
assert len(flights)==161
prov={x['flight_id']:x for x in json.loads((DATA/'flight_provenance.json').read_text()) if x['split'] in ['train','validation']}
read_files=[str(registry),str(DATA/'dataset_manifest.json'),str(DATA/'flights.csv'),str(DATA/'flight_provenance.json')]
cols=[c for c in pq.read_schema(DATA/'samples_train.parquet').names if not c.startswith('target_')]
print("imports and metadata ready",flush=True)
cols=["flight_id","split","sampling_group","timestamp_us","segment_id","valid_core","valid_frequency","review","flap_frequency_hz"] + ["attitude_q_"+a for a in "wxyz"] + ["velocity_ned_m_s_"+a for a in "xyz"] + ["angular_velocity_body_rad_s_"+a for a in "xyz"] + ["control_"+a+"_normalized" for a in ["flap_motor","left_elevon","right_elevon","rudder"]]
frames=[]
for split in ['train','validation']:
    path=DATA/f'samples_{split}.parquet'; read_files.append(str(path))
    f=pd.read_parquet(path,columns=cols); assert set(f.split)=={split}
    frames.append(f)
    print("loaded",split,len(f),flush=True)
samples=pd.concat(frames,ignore_index=True)
assert set(samples.flight_id)==set(flights.flight_id)
stat=[]; cond=[]; joint=[]; resp=[]; rawrows=[]; stable=[]; allstates=[]
def quant(x,w):
    x=np.asarray(x); w=np.asarray(w); ok=np.isfinite(x)&(w>0)
    if not ok.any(): return [np.nan]*5
    x=x[ok];w=w[ok];ii=np.argsort(x);x=x[ii];w=w[ii]
    return np.interp([.01,.1,.5,.9,.99],np.cumsum(w)/w.sum(),x).tolist()
def asof(d,t,field,maxage=None):
    if d is None or field not in d: return np.full(len(t),np.nan)
    ts=d['timestamp'].astype(np.int64); order=np.argsort(ts,kind='stable');ts=ts[order]
    ix=np.searchsorted(ts,t,side='right')-1; j=np.maximum(ix,0)
    v=np.asarray(d[field])[order][j].astype(float); ok=ix>=0
    if maxage is not None: ok &= t-ts[j]<=maxage
    v[~ok]=np.nan;return v
for n,meta in enumerate(flights.itertuples()):
    print("flight",n+1,flush=True)
    f=samples[samples.flight_id==meta.flight_id].sort_values('timestamp_us').copy()
    t=f.timestamp_us.to_numpy(np.int64); valid=f.valid_core.to_numpy(bool)
    seg=f.segment_id.to_numpy(); dt=np.r_[np.diff(t)/1e6,0.]
    cap=.04 if meta.sampling_group=='main_50hz' else .16
    wt=np.where(valid & np.r_[valid[1:],False] & np.r_[seg[1:]==seg[:-1],False] & (dt>0)&(dt<=cap),dt,0.)
    q=f[['attitude_q_'+a for a in 'wxyz']].to_numpy(float); goodq=np.isfinite(q).all(1)&(np.linalg.norm(q,axis=1)>0)
    q[~goodq]=[1,0,0,0]; rot=Rotation.from_quat(q[:,[1,2,3,0]])
    euler=rot.as_euler('ZYX',degrees=True); v=f[['velocity_ned_m_s_'+a for a in 'xyz']].to_numpy(float)
    rate=f[['angular_velocity_body_rad_s_'+a for a in 'xyz']].to_numpy(float)
    freq=f.flap_frequency_hz.to_numpy(float);freq[~f.valid_frequency.to_numpy(bool)]=np.nan
    values={'ground_speed':np.linalg.norm(v,axis=1),'roll_deg':euler[:,2],'pitch_deg':euler[:,1], 'p_rad_s':rate[:,0], 'q_rad_s':rate[:,1], 'r_rad_s':rate[:,2], 'frequency_hz':freq}
    for a in ['flap_motor','left_elevon','right_elevon','rudder']: values[a]=f['control_'+a+'_normalized'].to_numpy(float)
    base=dict(flight_id=meta.flight_id,split=meta.split,date=meta.flight_date,group=meta.sampling_group,uuid=prov[meta.flight_id].get('sys_uuid','unknown'),firmware=meta.firmware)
    rawpath=Path(manifest['config']['source_root'])/meta.log_path
    row={**base,'path':str(rawpath),'valid_s':wt.sum(),'status':'ok'}
    try:
        u=ULog(str(rawpath),message_name_filter_list=['wind','airspeed','airspeed_validated','estimator_status_flags','estimator_selector_status'])
        read_files.append(str(rawpath)); topics={(d.name,d.multi_id):d.data for d in u.data_list}
        row['wind_instances']=','.join(str(i) for name,i in topics if name=='wind')
        for name in ['wind','airspeed','airspeed_validated']:
            d=topics.get((name,0));row[name+'_rows']=0 if d is None else int(((d['timestamp']>=meta.start_us)&(d['timestamp']<=meta.end_us)).sum())
        wind=topics.get(('wind',0)); air=topics.get(('airspeed_validated',0)); selector=topics.get(('estimator_selector_status',0))
        wn=asof(wind,t,'windspeed_north',500000);we=asof(wind,t,'windspeed_east',500000)
        va=v-np.column_stack([wn,we,np.zeros(len(t))]);vb=rot.inv().apply(va)
        beta=np.rad2deg(np.arctan2(vb[:,1],np.hypot(vb[:,0],vb[:,2])))
        beta[np.linalg.norm(vb,axis=1)<3]=np.nan
        values['candidate_beta_deg']=beta
        values['wind_horizontal_m_s']=np.hypot(wn,we)
        values['wind_reported_sigma_m_s']=np.sqrt(asof(wind,t,'variance_north',500000)+asof(wind,t,'variance_east',500000))
        src=asof(air,t,'airspeed_source',200000); tas=asof(air,t,'true_airspeed_m_s',200000)
        sensor=np.isin(src,[1,2,3])&(tas>0);values['sensor_tas_m_s']=np.where(sensor,tas,np.nan)
        values['candidate_air_speed']=np.linalg.norm(va,axis=1)
        for label,mask in {'wind_fresh':np.isfinite(wn)&np.isfinite(we),'sensor_tas':sensor,'source_ground_wind':src==0,'source_synthetic':src==4,'source_missing':~np.isfinite(src),'candidate_beta_abs_gt10':abs(beta)>10}.items(): row[label+'_s']=float(wt[mask].sum())
        primary=asof(selector,t,'primary_instance'); beta_fused=np.full(len(t),np.nan)
        for (name,instance),d in topics.items():
            if name=='estimator_status_flags':
                flags=asof(d,t,'cs_fuse_beta') # event-driven status is held, no freshness rejection
                beta_fused[primary==instance]=flags[primary==instance]
        row['selected_beta_fusion_known_s']=float(wt[np.isfinite(beta_fused)].sum());row['selected_beta_fusion_on_s']=float(wt[beta_fused==1].sum())
        # Deliberately hypothetical +/-1 m/s in each horizontal wind axis, not calibrated uncertainty.
        candidates=[]
        for dn,de in [(-1,-1),(-1,1),(1,-1),(1,1)]:
            b=rot.inv().apply(va-np.array([dn,de,0.]));candidates.append(np.rad2deg(np.arctan2(b[:,1],np.hypot(b[:,0],b[:,2]))))
        lo=np.min(candidates,axis=0);hi=np.max(candidates,axis=0)
        row['beta_sign_sensitive_1m_s']=float(wt[np.isfinite(beta)&(lo<=0)&(hi>=0)].sum())
        values['beta_sensitivity_width_deg']=hi-lo
    except (OSError,ValueError,KeyError,IndexError) as e: row['status']=type(e).__name__+': '+str(e)
    rawrows.append(row)
    for name,x in values.items():
        stat.append({**base,'metric':name,'support_s':float(wt[np.isfinite(x)].sum()),**dict(zip(['p01','p10','p50','p90','p99'],quant(x,wt)))})
    masks={'straight_proxy':(abs(euler[:,2])<10)&(abs(rate[:,2])<.15)&(values['ground_speed']>4), 'turn_proxy':(abs(euler[:,2])>=15)&(abs(rate[:,2])>=.15), 'bank_gt30':abs(euler[:,2])>30,'bank_gt45':abs(euler[:,2])>45,'abs_p_gt1':abs(rate[:,0])>1,'abs_r_gt0.5':abs(rate[:,2])>.5,'low_speed_lt4':values['ground_speed']<4,'frequency_lt2':freq<2,'frequency_gt5':freq>5,'review':f.review.to_numpy(bool)}
    for name,mask in masks.items():cond.append({**base,'condition':name,'seconds':float(wt[mask].sum()),'total_s':float(wt.sum())})
    # Native-time joint occupancy; bins are descriptive, not RL admissibility thresholds.
    sub=pd.DataFrame({'speed_bin':pd.cut(values['ground_speed'],[0,4,6,8,10,12,np.inf],right=False).astype(str),'bank_bin':pd.cut(abs(euler[:,2]),[0,10,30,45,90,181],right=False).astype(str),'freq_bin':pd.cut(freq,[0,2,3,4,5,np.inf],right=False).astype(str),'seconds':wt})
    for keys,g in sub[sub.seconds>0].groupby(['speed_bin','bank_bin','freq_bin']):joint.append({**base,**dict(zip(['speed_bin','bank_bin','freq_bin'],keys)),'seconds':g.seconds.sum()})
    # Endpoint changes: disjoint intervals per flight and horizon; no control-effect regression.
    for h in [.1,.2,.5,1.]:
        last=-np.inf; observations=[]
        for i in np.flatnonzero(wt>0):
            if t[i]<=last:continue
            j=np.searchsorted(t,t[i]+h*1e6)
            if j>=len(t) or t[j]-t[i]>h*1e6+cap*1e6 or seg[j]!=seg[i] or not valid[j]:continue
            if not (wt[i:j]>0).all():continue
            last=t[j]
            a=(rot[i].inv()*rot[j]).magnitude()*180/np.pi
            observations.append([np.linalg.norm(v[j]-v[i]),np.linalg.norm(rate[j]-rate[i]),a,(t[j]-t[i])/1e6])
            if h==1.:
                sl=slice(i,j+1)
                if (abs(euler[sl,2])<10).all() and (abs(rate[sl,2])<.15).all() and np.linalg.norm(v[j]-v[i])<.5 and (values['ground_speed'][sl]>4).all():
                    stable.append({**base,'t0_us':int(t[i]),'elapsed_s':(t[j]-t[i])/1e6,**{k:float(np.nanmean(values[k][sl])) for k in ['ground_speed','pitch_deg','frequency_hz','flap_motor','left_elevon','right_elevon','rudder']}})
        arr=np.asarray(observations)
        if len(arr):
            for k,name in enumerate(['dv_norm_m_s','domega_norm_rad_s','dattitude_deg','actual_dt_s']):resp.append({**base,'horizon_s':h,'metric':name,'intervals':len(arr),**dict(zip(['p10','p50','p90','p99'],np.quantile(arr[:,k],[.1,.5,.9,.99])))})
    keep=wt>0
    allstates.append(pd.DataFrame({'split':meta.split,'group':meta.sampling_group,'speed':values['ground_speed'][keep],'bank':abs(euler[keep,2]),'freq':freq[keep],'seconds':wt[keep]}))
    if (n+1)%20==0:print(f'completed {n+1}/161',flush=True)
for name,rows in [('state_per_flight',stat),('conditions_per_flight',cond),('joint_occupancy',joint),('response_per_flight',resp),('air_wind_per_flight',rawrows),('cruise_windows',stable)]:pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)
pd.DataFrame(rawrows).groupby(['split','group']).sum(numeric_only=True).to_csv(OUT/'air_wind_summary.csv')
pd.DataFrame(cond).groupby(['split','group','condition'])[['seconds','total_s']].sum().to_csv(OUT/'conditions_summary.csv')
summary=pd.DataFrame(stat).groupby(['split','group','metric'])[['support_s','p01','p10','p50','p90','p99']].agg({'support_s':'sum',**{x:'median' for x in ['p01','p10','p50','p90','p99']}})
summary.to_csv(OUT/'state_summary.csv')
pd.DataFrame(resp).groupby(['split','group','horizon_s','metric'])[['intervals','p10','p50','p90','p99']].agg({'intervals':'sum',**{x:'median' for x in ['p10','p50','p90','p99']}}).to_csv(OUT/'response_summary.csv')
st=pd.concat(allstates,ignore_index=True)
fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for row,split in enumerate(['train','validation']):
    a=st[(st.split==split)&(st.group=='main_50hz')]
    for col,(name,bins) in enumerate([('bank',np.array([0,10,20,30,45,60,90])),('freq',np.arange(0,7,.5))]):
        hh,xe,ye=np.histogram2d(a.speed,a[name],bins=[np.arange(0,15,1),bins],weights=a.seconds)
        im=axes[row,col].pcolormesh(xe,ye,np.log10(1+hh.T),cmap='viridis');fig.colorbar(im,ax=axes[row,col],label='log10(1 + seconds)')
        axes[row,col].set(xlabel='Ground speed (m/s), NOT airspeed',ylabel='|Roll| (deg)' if name=='bank' else 'Frequency (Hz)',title=split+' main_50hz')
fig.savefig(OUT/'joint_support.png',dpi=160);plt.close(fig)
(OUT/'read_scope.json').write_text(json.dumps({'dataset_id':manifest['dataset_id'],'registry':str(registry),'partitions':['train','validation'],'flights':len(flights),'sample_rows':len(samples),'valid_core_rows':int(samples.valid_core.sum()),'opened_paths':read_files,'raw_failures':[r for r in rawrows if r['status']!='ok'],'python':sys.executable,'method':'native causal state coverage; duration weights only adjacent valid same-segment intervals; per-flight quantiles then median across flights; no parameter fit','raw_alignment':'past published wind <=500ms, validated airspeed <=200ms; wind instance 0 candidate only; vertical wind assumed zero; status flags selected using primary_instance if available','beta_sensitivity':'hypothetical +-1m/s each N/E axis, not an estimated uncertainty interval','test_payload_access':False},indent=2))
assert not any('_test.parquet' in p for p in read_files)
print('DONE',len(samples),len(stable),'raw failures',sum(r['status']!='ok' for r in rawrows),flush=True)
