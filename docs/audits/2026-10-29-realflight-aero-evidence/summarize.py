from pathlib import Path
import pandas as pd,numpy as np,json,subprocess
P=Path(__file__).resolve().parent/'results'
a=pd.read_csv(P/'air_wind_per_flight.csv'); s=pd.read_csv(P/'state_per_flight.csv');c=pd.read_csv(P/'conditions_per_flight.csv');r=pd.read_csv(P/'response_per_flight.csv');j=pd.read_csv(P/'joint_occupancy.csv');cr=pd.read_csv(P/'cruise_windows.csv')
for df in [a,s,c,r,j,cr]: assert set(df.split)<= {'train','validation'}
assert len(a)==161 and a.flight_id.nunique()==161 and (a.status=='ok').all()
assert (a.wind_fresh_s<=a.valid_s+1e-6).all()
assert np.allclose(j.groupby('flight_id').seconds.sum().sort_index(),a.set_index('flight_id').valid_s.sort_index())
assert np.allclose(c[c.condition=='review'].set_index('flight_id').total_s.sort_index(),a.set_index('flight_id').valid_s.sort_index())
rows=[]
for (split,group,date,metric),g in s.groupby(['split','group','date','metric']):rows.append(dict(split=split,group=group,date=date,metric=metric,flights=len(g),median=g.p50.median(),p10_median=g.p10.median(),p90_median=g.p90.median(),support_s=g.support_s.sum()))
pd.DataFrame(rows).to_csv(P/'by_date.csv',index=False)
print('AIR',a.groupby(['split','group'])[['valid_s','wind_fresh_s','sensor_tas_s','selected_beta_fusion_on_s','beta_sign_sensitive_1m_s']].sum().round(2).to_string())
print('STATE',pd.read_csv(P/'state_summary.csv').query("group=='main_50hz' and metric in ['ground_speed','sensor_tas_m_s','roll_deg','pitch_deg','frequency_hz','p_rad_s','r_rad_s','candidate_beta_deg','beta_sensitivity_width_deg']").to_string(index=False))
print('CONDITIONS',pd.read_csv(P/'conditions_summary.csv').query("group=='main_50hz'").to_string(index=False))
print('RESPONSES',pd.read_csv(P/'response_summary.csv').query("group=='main_50hz' and horizon_s<1").to_string(index=False))
print('CRUISE',cr.groupby(['split','group']).agg(windows=('flight_id','size'),flights=('flight_id','nunique')).to_string())
print('CRUISE RANGE',cr[['ground_speed','pitch_deg','frequency_hz','flap_motor','left_elevon','right_elevon','rudder']].quantile([.1,.5,.9]).to_string())
print('DATES',pd.DataFrame(rows).query("group=='main_50hz' and metric in ['sensor_tas_m_s','frequency_hz','roll_deg']").pivot(index='date',columns='metric',values='median').round(3).to_string())
print('RAW WIND INSTANCES',a.wind_instances.value_counts().to_dict(),'UUID',a.uuid.value_counts().to_dict())
# Verify actual recorded firmware revisions using existing local git objects only.
checks=[]
for rev in sorted(a.firmware.dropna().unique()):
    if len(rev)!=40: continue
    paths=['src/modules/ekf2/EKF/aid_sources/sideslip/sideslip_fusion.cpp','msg/versioned/AirspeedValidated.msg']
    row={'firmware':rev,'flights':int((a.firmware==rev).sum())}
    for path,key,needle in zip(paths,['zero_beta_observation','sensor_source_enum'],['float observation = 0.f','SOURCE_SENSOR_1 = 1']):
        x=subprocess.run(['git','-C','/home/zn/PX4-Autopilot','show',rev+':'+path],capture_output=True,text=True)
        row[key]=needle in ' '.join(x.stdout.split()) if key=='sensor_source_enum' else needle in x.stdout
        row[key+'_available']=x.returncode==0
    checks.append(row)
pd.DataFrame(checks).to_csv(P/'firmware_semantics.csv',index=False)
print('FIRMWARE',pd.DataFrame(checks).to_string(index=False))
(P/'verification.json').write_text(json.dumps({'authorized_flights':161,'raw_success':161,'duration_joint_closure':True,'condition_denominator_closure':True,'test_rows_in_outputs':0,'firmware_versions_checked':len(checks),'no_model_fit':True},indent=2))
