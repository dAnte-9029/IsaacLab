"""Independent offline receipt/action/factor checks; never launch PhysX."""
import argparse
import json
from pathlib import Path
import sqlite3
import numpy as np
from flapping_bot.analysis.unified_retrospective import immutable_store


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--history',type=Path,required=True)
    args=p.parse_args();base=args.input.resolve();read=lambda p:json.loads(p.read_text())
    assert read(base/'status.json')['status']=='completed'
    result=read(base/'analysis/results.json');assert len(result['audit'])==8
    source=immutable_store(args.history/'evidence');assert source.db.execute('pragma integrity_check').fetchone()[0]=='ok'
    history={r['initial']['scene']['env_index']:list(source.trace(r['instance_id'])) for r in source.records};source.close()
    traces={};total=0;prefix=0;digest_by_update={};stale_status=[]
    for start in (120,144):
        for group in 'ABCD':
            folder=base/f'T{start}_{group}';c=read(folder/'contract.json');s=read(folder/'summary.json')
            assert s['passed']
            status=read(folder/'status.json')['status']
            assert status in ('running','completed')
            if status=='running':stale_status.append(folder.name)
            assert s['terminated'] is None and s['optimizer_updates']==0
            assert s['policies_unchanged'] and s['rng_unchanged_by_inference']
            assert c['schema']=='sustained_cross_policy_intervention_v1'
            assert c['switch_after_step']==start and c['group']==group
            assert c['slot']==1 and c['dt']==1/60
            for u,v in c['checkpoint'].items():
                assert digest_by_update.setdefault(u,v['digest'])==v['digest']
            trace=[json.loads(x) for x in (folder/'trace.jsonl').read_text().splitlines()]
            assert len(trace)==start+60
            traces[start,group]=trace;total+=len(trace);prefix+=start
            for k,e in enumerate(trace,1):
                assert e['step']==k and e['time_s']==k/60
                expected_group='D' if k<=start else group
                assert e['group']==expected_group
                f,t=dict(A=(25,25),B=(25,150),C=(150,25),D=(150,150))[expected_group]
                assert (e['source_F'],e['source_T'])==(f,t)
                for i,ctl in enumerate(e['controls']):
                    expected=np.array(e['mu150'][i])
                    if i==1:expected=np.r_[e[f'mu{f}'][i][0],e[f'mu{t}'][i][1:]]
                    np.testing.assert_array_equal(ctl['raw_command'],expected)
                    np.testing.assert_array_equal(ctl['command'],np.clip(expected,-1,1))
                    if k<=start or group=='D':assert ctl==history[i][k-1]['control']
                assert not any(e['done']) and not any(e['timeout'])
            db=folder/'evidence/rollout_feedback.sqlite3'
            conn=sqlite3.connect(db.as_uri()+'?mode=ro&immutable=1',uri=True)
            counts={table:conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0] for table in ('episodes','steps','terminals','witnesses')}
            conn.close();assert not any(counts.values()),counts
        for group in 'ABC':
            assert traces[start,group][:start]==traces[start,'D'][:start]
            assert read(base/f'T{start}_{group}/start_state.json')==read(base/f'T{start}_D/start_state.json')
    assert traces[120,'D']==traces[144,'D'][:180]
    # Recalculate every factor from the endpoint table, independent of analysis loops.
    for e in result['factor_effects']:
        rows={r['group']:r for r in result['metrics'] if r['start_s']==e['start_s'] and r['window_s']==e['window_s']}
        y=np.array([rows[g][e['metric']] for g in 'ABCD'])
        np.testing.assert_allclose([e['delta_T'],e['delta_F'],e['delta_FT']],
            np.array([[.5,-.5,.5,-.5],[.5,.5,-.5,-.5],[1,-1,-1,1]])@y,rtol=0,atol=1e-12)
    # Endpoints in tables must map to exact CSV cycle, not a nearby time sample.
    for row in result['metrics']:
        start=round(row['start_s']*60);end=start+round(row['window_s']*60)
        data=np.genfromtxt(base/f"analysis/T{start}_{row['group']}.csv",delimiter=',',names=True)
        for key in ('v_y_mps','e_y_m','e_z_m','course_error_deg','roll_deg','actual_hz'):
            assert row[key]==float(data[key][end-1])
            assert row['delta_'+key]==float(data[key][end-1]-data[key][start-1])
    receipt=dict(schema='sustained_intervention_offline_acceptance_v1',passed=True,
        independent_processes=8,sustained_control_steps_total=total,all_scene_control_transitions=total*5,
        sustained_common_prefix_steps=prefix,sustained_intervention_steps=480,
        historical_controls_exact=True,all_visible_common_prefix_exact=True,
        D120_equals_D144_first_180_steps=True,action_composition_and_clipping_exact=True,
        endpoint_time_alignment=True,factor_independent_matrix_check=True,empty_bootstrap_evidence=True,
        checkpoint_digests=digest_by_update,no_termination=True,optimizer_updates=0,
        stale_child_status_files=stale_status,completion_evidence='parent completed receipt, complete traces and all child summaries passed; Kit close exited before child main status write')
    (base/'offline_acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':main()
