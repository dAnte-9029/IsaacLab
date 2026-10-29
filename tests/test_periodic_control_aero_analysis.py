"""Focused checks for finite-horizon interpolation and paired differencing."""
import importlib.util
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('periodic_analysis',Path(__file__).parents[1]/'scripts/flapping_px4/analyze_periodic_control_aero_3d.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_fractional_time_interpolation_preserves_batch_axes():
    t=np.array([0.,.2,.4]); x=t[:,None,None]*np.arange(6).reshape(1,2,3)
    np.testing.assert_allclose(m.interp(t,x,.3),.3*np.arange(6).reshape(2,3))

def test_paired_increment_cancels_common_drift():
    t=np.array([0.,.2,.4]); drift=t**2
    x=np.stack([drift,drift+2*t],axis=1)
    v=m.interp(t,x,.3)
    np.testing.assert_allclose(v[1]-v[0],.6)

def test_phase_sections_noninteger_steps_and_angle_wrap():
    phase=np.arange(0,4*np.pi+.4,.4)
    trace={'phase_unwrapped':phase[:,None]}
    for key in ['t','frequency']:
        trace[key]=(phase/(2*np.pi*3.375) if key=='t' else np.full_like(phase,3.375))[:,None]
    for key in ['position','velocity','body_velocity','omega','euler','joint','joint_velocity']:
        trace[key]=np.repeat(phase[:,None,None],3,axis=2)
    trace['euler']=(trace['euler']+np.pi)%(2*np.pi)-np.pi
    out=m.phase_sections(trace,0)
    np.testing.assert_allclose(out['t'],np.arange(3)/3.375,atol=1e-12)
    np.testing.assert_allclose(out['euler'][:,0],np.arange(3)*2*np.pi,atol=1e-12)
