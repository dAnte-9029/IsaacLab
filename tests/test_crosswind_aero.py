"""Targeted adapter checks: reference, velocity origin, symmetry, force closure."""
import math
import numpy as np
import torch
from flapping_bot.analysis.crosswind_aero import Experiment, cases, CG, transform_checks
from flapping_bot.config.ullt import ULLTConfig
from flapping_bot.physics.ullt_state_space import ULLTBackend


def test_geometry_reference_and_angular_velocity_origin():
    exp=Experiment(cases(),10)
    g,m=exp.sample(.023,4)
    # Local velocity includes translation at COM, body omega and flap velocity.
    t=.023; qd=exp.amplitude*8*math.pi*math.cos(8*math.pi*t)
    for i,sign in enumerate([-1,1]):
        sl=slice(i*10,(i+1)*10)
        flap=torch.zeros_like(exp.omega); flap[:,0]=sign*qd
        expected=-(exp.velocity[:,None]+torch.linalg.cross(exp.omega[:,None],g.three_quarter_chord_w_m[:,sl]-CG)
            +torch.linalg.cross(flap[:,None],g.three_quarter_chord_w_m[:,sl]-exp.hinges[i]))
        torch.testing.assert_close(m.flow_three_quarter_w_mps[:,sl],expected)
    backend=ULLTBackend(ULLTConfig(model='four_state'))
    state=backend.zero_state(g); loads=backend.evaluate(g,m,state,exp.reference)
    assert max(transform_checks(exp,backend,state,g,m,loads).values())<1e-9
    torch.testing.assert_close(loads.force_w_n,loads.section_force_w_n.sum(1))
    torch.testing.assert_close(loads.moment_about_reference_w_nm,loads.section_moment_about_reference_w_nm.sum(1))


def test_ideal_mirror_and_production_delaurier_input_reduction():
    table=cases(); exp=Experiment(table,10)
    exp.reference=exp.reference.clone(); exp.reference[:,1]=0
    backend=ULLTBackend(ULLTConfig(model='four_state'))
    g,m=exp.sample(.031,4)
    loads=backend.evaluate(g,m,backend.zero_state(g),exp.reference)
    values=torch.cat([loads.force_w_n,loads.moment_about_reference_w_nm],1).numpy()
    parity=np.array([1,-1,1,-1,1,-1])
    for a,b in [('beta10','beta-10'),('p0.25','p-0.25'),('r0.25','r-0.25')]:
        ia=table.index[table.case==a][0]; ib=table.index[table.case==b][0]
        np.testing.assert_allclose(values[ia],values[ib]*parity,atol=1e-10)
    d=exp.delaurier(.031,4).sum(1)
    for name in ['p0.25','r0.25','p-0.25','r-0.25']:
        np.testing.assert_allclose(d[table.index[table.case==name][0]],d[0],atol=1e-12)
    tails=exp.tails()
    # Five surfaces close to the public total wrench; mirrored differential control.
    tail=sum(tails.values())
    ia=table.index[table.case=='differential5'][0]; ib=table.index[table.case=='differential-5'][0]
    np.testing.assert_allclose(tail[ia],tail[ib]*parity,atol=1e-12)
