"""Focused artifact and numerical-identity checks; no PhysX or training."""
import json
import sys
from pathlib import Path
import numpy as np

out=Path(sys.argv[1])
checks=json.loads((out/'checks.json').read_text())
summary=json.loads((out/'summary.json').read_text())
for start in (120,144):
    d=np.genfromtxt(out/f'T{start}_derivatives.csv',delimiter=',',names=True)
    assert len(d)==480 and all(np.isfinite(d[n]).all() for n in d.dtype.names)
    assert np.max(abs(np.diff(d['time_s'])-1/480))<1e-12
    for name,pieces in [('prediction_aq_Nm',['alpha_term_Nm','q_term_Nm']),('prediction_aqV_Nm',['alpha_term_Nm','q_term_Nm','V_term_Nm']),('Kalpha',['alpha_rz_dFx','alpha_minus_rx_dFz']),('Dq',['q_rz_dFx','q_minus_rx_dFz'])]:
        assert np.max(abs(d[name]-sum(d[x] for x in pieces)))<1e-10,name
    for name,delta,term in [('Kalpha','delta_alpha_rad','alpha_term_Nm'),('Dq','delta_q_rad_s','q_term_Nm'),('KV','delta_V_mps','V_term_Nm')]:
        assert np.max(abs(d[name]*d[delta]-d[term]))<1e-12
    for group in ('O','S+','S-'):
        assert checks[f'T{start}_{group}_reconstruction_max_Nm']<5e-5
    assert checks[f'T{start}_q_station_flow_error_mps']<1e-10
    assert checks[f'T{start}_relative_motion_error']==0
    assert summary[str(start)]['phase_difference_max_rad']==0
    assert summary[str(start)]['frequency_difference_max_hz']==0
    # Historical signed impulses also detect a mistaken time/window or S sign.
    target=.85015 if start==120 else .79042
    assert abs(d['actual_Nm'].sum()/480-target)<1e-5
print('PASS: both starts, six reconstructions, kinematics, finite differences, force-arm closure, original signed impulses')
