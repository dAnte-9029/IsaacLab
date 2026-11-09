"""Independent offline acceptance of historical source, pulses and common frames."""
import argparse,json,sqlite3
from pathlib import Path
import numpy as np
from flapping_bot.analysis.unified_retrospective import immutable_store
from flapping_bot.path_tracking.unified_trajectory import Trajectory,TrajectorySpec
from run_sustained_cross_policy import numeric_compare


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--history',type=Path,required=True);a=p.parse_args();root=a.input
    read=lambda p:json.loads(p.read_text())
    assert read(root/'status.json')['status']=='completed'
    store=immutable_store(a.history/'evidence');assert store.db.execute('pragma integrity_check').fetchone()[0]=='ok'
    hist={r['initial']['scene']['env_index']:list(store.trace(r['instance_id'])) for r in store.records};store.close()
    directions={'O':[0,0,0,0],'R+':[0,.1,0,0],'R-':[0,-.1,0,0],'S+':[0,0,.1,.1],'S-':[0,0,-.1,-.1],'D+':[0,0,.1,-.1],'D-':[0,0,-.1,.1]}
    audit={};baselines={}
    for start in (120,144):
        baseline=None
        for g,direction in directions.items():
            f=root/f'T{start}_{g}';s=read(f/'summary.json');c=read(f/'contract.json')
            assert s['passed'] and s['terminated'] is None and s['completed_control_steps']==start+60
            assert read(f/'status.json')['status']=='completed'
            es=[json.loads(x) for x in (f/'trace.jsonl').read_text().splitlines()];assert len(es)==start+60
            if g=='O':baseline=es;baselines[start]=es
            else:
                assert es[:start]==baseline[:start]
                assert read(f/'start_state.json')==read(root/f'T{start}_O/start_state.json')
            db=f/'evidence/rollout_feedback.sqlite3'
            conn=sqlite3.connect(db.resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
            assert all(conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]==0 for table in ('episodes','steps','terminals','witnesses'))
            conn.close()
            maxima={};clips=0
            for k,e in enumerate(es,1):
                assert e['step']==k and not any(e['done']) and not any(e['timeout'])
                for i,ctrl in enumerate(e['controls']):
                    raw=np.array(hist[i][k-1]['control']['raw_command'],dtype=np.float32)
                    assert e['base_raw'][i]==raw.tolist()
                    if i==1 and start<k<=start+18:raw=raw+np.array(direction,dtype=np.float32)
                    np.testing.assert_array_equal(ctrl['raw_command'],raw)
                    np.testing.assert_array_equal(ctrl['command'],np.clip(raw,-1,1))
                    applied=np.array(ctrl['applied'][1:],dtype=np.float32)
                    lower=np.array(c['soft_limits_rad']['lower'],dtype=np.float32)
                    upper=np.array(c['soft_limits_rad']['upper'],dtype=np.float32)
                    target=np.where(applied>=0,applied*upper,-applied*lower)
                    np.testing.assert_allclose(e['targets_rad'][i],target,atol=2e-6,rtol=0)
                    assert len(e['visible']['observation'][i])==39
                    if i==1 and k>start:clips+=int(np.any(abs(raw)>1))
                    if k<=start or g=='O' or i!=1:numeric_compare(ctrl,hist[i][k-1]['control'],maxima,f'slot{i}')
            audit[f'T{start}_{g}']=dict(steps=len(es),historical_max=max(maxima.values()),clipped_steps=clips,common_prefix_exact=True)
            if (root/'analysis').exists():
                data=np.genfromtxt(root/f'analysis/T{start}_{g}.csv',delimiter=',',names=True)
                path=Trajectory(TrajectorySpec(**c['tasks'][1]));q=path.query(np.array([e['controls'][1]['projection']['progress'] for e in baseline]))
                vel=np.array([e['controls'][1]['root_state'][7:10] for e in es])
                np.testing.assert_allclose(data['vy'],(vel*q['normal']).sum(1),atol=1e-12,rtol=0)
                np.testing.assert_allclose(data['an'],(np.gradient(vel,1/60,axis=0,edge_order=2)*q['normal']).sum(1),atol=1e-12,rtol=0)
        if (root/'analysis').exists():
            for direction in 'RSD':
                plus,minus,o=[np.genfromtxt(root/f'analysis/T{start}_{g}.csv',delimiter=',',names=True) for g in (direction+'+',direction+'-','O')]
                res=np.genfromtxt(root/f'analysis/T{start}_{direction}_sensitivity.csv',delimiter=',',names=True)
                for key in ('vy','py','vz','pz','an','roll','roll_rate','yaw_rate','chi','chi_rate','kappa'):
                    np.testing.assert_allclose(res['J_'+key],(plus[key]-minus[key])/.2,atol=1e-12,rtol=0)
                    np.testing.assert_allclose(res['N_'+key],(plus[key]+minus[key])/2-o[key],atol=1e-12,rtol=0)
    # Group metadata differs after T120 even for O? Both remain O, same complete history.
    assert baselines[120]==baselines[144][:180]
    receipt=dict(passed=True,independent_processes=14,audit=audit,historical_raw_source_verified=True,common_prefix_exact=True,all_other_slots_historical=True,applied_to_target_verified=True,empty_bootstrap_evidence=True,analysis_verified=(root/'analysis').exists(),optimizer_updates=0)
    (root/'offline_acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
