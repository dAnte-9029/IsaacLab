"""Reproducible FLU crosswind experiment using unchanged physics backends.

Loads are N/Nm about the measured base-link COM reference, not the moving
whole-articulation COM. Air velocity means vehicle minus wind. Rates are
instantaneous frozen-body operating points, not a kinematically integrated turn.
"""
from dataclasses import asdict
from pathlib import Path
import json
import math
import time
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import torch
from flapping_bot.config.ullt import ULLTConfig
from flapping_bot.physics.ullt_state_space import ULLTBackend, ULLTGeometry, ULLTMotion
from flapping_bot.physics.ullt_geometry import load_csv_urdf_geometry, rotation_xyz
from flapping_bot.physics.wing_geom_csv import build_wing_geometry_from_csv
from flapping_bot.physics.qsm_delaurier1993 import (
    DeLaurierParams, compute_delaurier_strip_loads, integrate_delaurier_strip_wrench,
    transform_wang_wrench_to_link,
)
from flapping_bot.physics.tail_aero import TailAeroCfg, TailAeroModel
from .ullt_flight_replay import flight_kinematics
from flapping_bot.physics.measured_mass_properties import BODY_LINK_MASS_PROPERTIES

ROOT = Path(__file__).resolve().parents[4]
CSV = ROOT / 'outputs_DeLaurier/right_wing_te_fit_poly5_gap50.csv'
URDF = ROOT / 'source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf'
CG = torch.tensor(BODY_LINK_MASS_PROPERTIES.com_m, dtype=torch.float64)
AXES = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
PARAMS = DeLaurierParams(alpha0_rad=0, eta_s=.65, cd_cf=1.95,
    alpha_stall_min_rad=math.radians(-12), alpha_stall_max_rad=math.radians(12), c_mac=0, cd_f=.028)
TAIL = TailAeroCfg(fixed_horizontal_effectiveness=.5, elevon_effectiveness=1.2)


def cases():
    rows = []
    def add(name, **kw):
        rows.append(dict(case=name, beta=0., rudder=0., symmetric=0., differential=0., p=0., r=0.))
        rows[-1].update(kw)
    add('neutral')
    for b in [-10, -5, -.1, .1, 5, 10]:
        add(f'beta{b:g}', beta=b)
    for channel in ['rudder', 'symmetric', 'differential']:
        for a in [-10, -5, 5, 10]:
            add(f'{channel}{a}', **{channel: a})
    for channel in ['p', 'r']:
        for a in [-.5, -.25, .25, .5]:
            add(f'{channel}{a:g}', **{channel: a})
    for b in [-10, 10]:
        for d in [-10, 10]:
            add(f'combined_b{b}_rud{d}', beta=b, rudder=d)
        add(f'combined_b{b}_diff', beta=b, differential=-b)
    return pd.DataFrame(rows)


class Experiment:
    def __init__(self, table, sections):
        self.table = table
        self.geometry = load_csv_urdf_geometry(CSV, URDF, sections_per_wing=sections)
        # DeLaurier uses its production 80-strip quadrature. Same full CSV planform.
        self.dg, self.info = build_wing_geometry_from_csv(CSV, N=80, device='cpu', dtype=torch.float64)
        root = ET.parse(URDF).getroot()
        origins = [root.find(f"joint[@name='{side}_wing']/origin") for side in ['right', 'left']]
        self.hinges = torch.tensor([[float(x) for x in o.attrib['xyz'].split()] for o in origins], dtype=torch.float64)
        self.rolls = torch.tensor([float(o.attrib['rpy'].split()[0]) for o in origins], dtype=torch.float64)
        self.amplitude = min(math.pi/6, float(root.find("joint[@name='left_wing']/limit").attrib['upper']))
        self.velocity = torch.zeros((len(table), 3), dtype=torch.float64)
        beta = torch.tensor(table.beta.to_numpy()) * math.pi/180
        self.velocity[:, 0], self.velocity[:, 1] = 8*beta.cos(), 8*beta.sin()
        self.omega = torch.tensor(np.stack([table.p, table.p*0, table.r], 1))
        self.reference = CG.expand(len(table), -1)

    def sample(self, t, frequency):
        inputs = torch.zeros((len(self.table), 16), dtype=torch.float64)
        inputs[:, 0], inputs[:, 1] = 2*math.pi*frequency*t, 2*math.pi*frequency
        # Shared velocity at reference. flight_kinematics expects origin velocity.
        inputs[:, 3:6] = self.velocity - torch.linalg.cross(self.omega, self.reference)
        inputs[:, 9:12] = self.omega
        inputs[:, 15] = 1.225
        return flight_kinematics(self.geometry, self.hinges, inputs, self.amplitude)

    def delaurier(self, t, frequency):
        batch = len(self.table)
        rate = 2*math.pi*frequency
        q, qd, qdd = [self.amplitude*x for x in (math.sin(rate*t), rate*math.cos(rate*t), -rate**2*math.sin(rate*t))]
        y = self.dg.x_mid[None].expand(batch, -1)
        zero = torch.zeros_like(y)
        # Exactly the production wing input reduction; body p/r and lateral v absent.
        loads = compute_delaurier_strip_loads(-q*y, -qd*y, -qdd*y, zero, zero, zero,
            self.dg, rho=1.225, U=self.velocity[:, :1], theta_a=0., theta_bar=0.,
            omega_ref=rate, params=PARAMS, enable_separation=False)
        wrench = integrate_delaurier_strip_wrench(loads)
        result = []
        for i, sign in enumerate([-1, 1]):
            a = y.new_tensor([[0, 0, 1], [sign, 0, 0], [0, 1, 0]]).expand(batch, -1, -1)
            f, m = transform_wang_wrench_to_link(wrench.force_wang, wrench.moment_wang_about_wing_origin, a)
            angles = y.new_zeros((batch, 3)); angles[:, 0] = self.rolls[i] + sign*q
            rot = rotation_xyz(angles)
            f, m = [torch.einsum('bij,bj->bi', rot, x) for x in [f, m]]
            m += torch.linalg.cross(self.hinges[i] - self.reference, f)
            result.append(torch.cat([f, m], 1))
        return torch.stack(result, 1).numpy()

    def tails(self):
        def angle(name):
            return torch.tensor(self.table[name].to_numpy())*math.pi/180
        results = TailAeroModel(TAIL, 'cpu').compute_surface_results(
            root_lin_vel_b=self.velocity, root_ang_vel_b=self.omega,
            left_elevon_rad=angle('symmetric')+angle('differential'),
            right_elevon_rad=angle('symmetric')-angle('differential'),
            rudder_rad=angle('rudder'), base_com_pos_b=self.reference)
        return {s.name: torch.cat([s.force_b, s.moment_b_about_com], 1).numpy() for s in results}


def transform_checks(exp, backend, state, g, m, loads):
    """Independent proper-rotation covariance, origin translation and wind checks."""
    rot = rotation_xyz(CG.new_tensor([.27, -.19, .41]))
    shift = CG.new_tensor([2., -3., .7])
    def vector(x): return x @ rot.T
    gg = ULLTGeometry(*(vector(getattr(g, n)) + (shift if n in ['edge_start_w_m', 'edge_end_w_m', 'quarter_chord_w_m'] else 0)
        if n != 'chord_m' else g.chord_m for n in vars(g)))
    mm = ULLTMotion(*(vector(getattr(m, n)) for n in vars(m)))
    changed = backend.evaluate(gg, mm, state, vector(exp.reference)+shift)
    cov = max(float((changed.force_w_n-vector(loads.force_w_n)).abs().max()),
              float((changed.moment_about_reference_w_nm-vector(loads.moment_about_reference_w_nm)).abs().max()))
    other = backend.evaluate(g, m, state, exp.reference+shift)
    expected = loads.moment_about_reference_w_nm-torch.linalg.cross(shift.expand_as(exp.reference), loads.force_w_n)
    ref_error = float((other.moment_about_reference_w_nm-expected).abs().max())
    # A rotated world wind must produce the same relative body velocity.
    ground = exp.velocity + CG.new_tensor([1., 2., 0.])
    wind = ground-exp.velocity
    wind_error = float(((ground@rot.T-wind@rot.T)@rot-exp.velocity).abs().max())
    return dict(rotation_covariance_max=cov, reference_translation_max=ref_error, wind_transform_max=wind_error)


def run(output, steps=128, sections=10, max_cycles=32, models=None, frequencies=(3.,4.), case_names=None, resume_from=None):
    if steps < 32 or sections < 3 or max_cycles < 4:
        raise ValueError('Need >=32 samples/cycle, >=3 sections/wing and >=4 cycles')
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    table = cases()
    if case_names is not None:
        table = table[table.case.isin(case_names)].reset_index(drop=True)
        if len(table) != len(case_names): raise ValueError('Unknown or duplicate cases')
    table.to_csv(output/'cases.csv', index=False)
    exp = Experiment(table, sections)
    # Wing loads independent of tail commands: integrate each distinct aerodynamic case once.
    unique = table.drop_duplicates(['beta', 'p', 'r']).reset_index(drop=True)
    wing_exp = Experiment(unique, sections)
    lookup = {(r.beta, r.p, r.r): i for i,r in unique.iterrows()}
    ids = [lookup[(r.beta,r.p,r.r)] for _,r in table.iterrows()]
    tails = exp.tails()
    metadata = dict(frame='FLU: x forward, y left, z up', reference_m=CG.tolist(),
        beta_definition='atan2(vehicle_minus_wind_y, vehicle_minus_wind_x), constant total speed 8 m/s',
        geometry_csv=str(CSV.relative_to(ROOT)), geometry_urdf=str(URDF.relative_to(ROOT)),
        amplitude_rad=exp.amplitude, twist='disabled identically in both models (production default)',
        rho=1.225, delaurier=asdict(PARAMS), tail=asdict(TAIL), body_CdA=0., induced_drag_correction=0.,
        steps_per_cycle=steps, sections_per_wing_ullt=sections, delaurier_strips=80,
        max_cycles=max_cycles, dtype='float64', threads=1, runs={})
    mean_rows, phase_rows = [], []
    if resume_from is not None:
        prior = json.loads((resume_from/'summary.json').read_text())
        for name in ['reference_m','steps_per_cycle','sections_per_wing_ullt','delaurier','tail','amplitude_rad','twist']:
            if prior[name] != json.loads(json.dumps(metadata[name])): raise ValueError(f'Resume configuration mismatch: {name}')
        pd.testing.assert_frame_equal(table, pd.read_csv(resume_from/'cases.csv'), check_dtype=False)
        valid = {k:v for k,v in prior['runs'].items() if 'failure' not in v}
        for key, value in valid.items():
            if key.startswith('delaurier_'): value['max_incidence_deg'] = None
        metadata['runs'].update(valid)
        metadata['reused_successful_runs_from'] = str(resume_from)
        mean_rows = pd.read_csv(resume_from/'means.csv').to_dict('records')
        phase_rows = pd.read_csv(resume_from/'phase.csv.gz').to_dict('records')
    for frequency in frequencies:
        for model in models or ['delaurier','ullt4','ullt4_A']:
            start = time.perf_counter(); key=f'{model}_{frequency:g}Hz'
            if key in metadata['runs']: continue
            backend = None
            if model != 'delaurier':
                cfg = ULLTConfig(model='four_state', section_model='nonlinear_flat_plate' if model=='ullt4_A' else 'linear')
                backend = ULLTBackend(cfg)
                g, m = wing_exp.sample(0, frequency); state=backend.zero_state(g)
            previous = None; periodic_error=None; history=[]; checks={}; incidence=0.
            try:
                for cycle in range(max_cycles if backend else 1):
                    trace=[]
                    for j in range(steps):
                        t=(cycle+j/steps)/frequency
                        if backend:
                            g,m=wing_exp.sample(t,frequency)
                            loads=backend.evaluate(g,m,state,wing_exp.reference)
                            incidence=max(incidence, float(loads.geometric_incidence_rad.abs().max())*180/math.pi)
                            six=torch.cat([loads.section_force_w_n,loads.section_moment_about_reference_w_nm],-1)
                            trace.append(torch.stack([six[:,:sections].sum(1),six[:,sections:].sum(1)],1).numpy())
                            if cycle==0 and j==steps//3:
                                checks=transform_checks(wing_exp,backend,state,g,m,loads)
                            state=backend.advance(state, lambda tt:wing_exp.sample(tt,frequency),t,1/(frequency*steps))
                        else:
                            trace.append(wing_exp.delaurier(t,frequency))
                    trace=np.array(trace)
                    if not np.isfinite(trace).all(): raise FloatingPointError('Non-finite loads')
                    if previous is not None:
                        periodic_error=float(np.max(np.abs(trace-previous)/(1+np.abs(trace))))
                        history.append(periodic_error)
                    previous=trace
                    print(f'{key} cycle {cycle+1}, periodic error {periodic_error}', flush=True)
                    if cycle>=3 and len(history)>=2 and max(history[-2:])<1e-5: break
                if backend and (len(history)<2 or max(history[-2:])>=1e-5):
                    raise RuntimeError('Wake has not converged to periodic loads')
                elapsed=time.perf_counter()-start
                metadata['runs'][key]=dict(seconds=elapsed, cycles=cycle+1, periodic_error=periodic_error,
                    periodic_history=history, checks=checks, max_incidence_deg=incidence if backend else None,
                    wing_batch=len(unique), batch_step_ms=elapsed*1000/((cycle+1)*steps),
                    configuration=None if not backend else asdict(cfg))
                all_parts={name:np.broadcast_to(value,(steps,*value.shape)) for name,value in tails.items()}
                all_parts['tail']=sum(all_parts.values())
                all_parts['right_wing']=trace[:,ids,0]; all_parts['left_wing']=trace[:,ids,1]
                all_parts['wings']=all_parts['right_wing']+all_parts['left_wing']
                all_parts['body']=np.zeros_like(all_parts['tail'])
                all_parts['aircraft']=all_parts['wings']+all_parts['tail']+all_parts['body']
                for component, values in all_parts.items():
                    for i,row in table.iterrows():
                        base=dict(model=model,frequency=frequency,component=component,**row.to_dict())
                        avg, rms=values[:,i].mean(0),values[:,i].std(0)
                        mean_rows.append(dict(**base,**dict(zip(AXES,avg)),**dict(zip([a+'_std' for a in AXES],rms))))
                        for j in range(steps):
                            phase_rows.append(dict(**base,phase_deg=j*360/steps,**dict(zip(AXES,values[j,i]))))
            except (ValueError,RuntimeError,FloatingPointError) as error:
                metadata['runs'][key]=dict(failure=str(error),seconds=time.perf_counter()-start, periodic_history=history, checks=checks)
                if previous is not None: np.save(output/f'{key}_unscored_last_cycle.npy', previous)
                print(f'{key}: FAILED: {error}',flush=True)
            (output/'summary.json').write_text(json.dumps(metadata,indent=2))
            pd.DataFrame(mean_rows).to_csv(output/'means.csv',index=False)
    phase=pd.DataFrame(phase_rows); phase.to_csv(output/'phase.csv.gz',index=False)
    analyze(output, pd.DataFrame(mean_rows), phase)
    return metadata


def analyze(output, means, phase):
    derivatives=[]
    for keys, group in means.groupby(['model','frequency','component']):
        idx=group.set_index('case')
        for channel, pair, delta in [('beta',('beta-5','beta5'),math.radians(10)),
              ('beta_local',('beta-0.1','beta0.1'),math.radians(.2)),
              ('rudder',('rudder-5','rudder5'),math.radians(10)),
              ('symmetric',('symmetric-5','symmetric5'),math.radians(10)),
              ('differential',('differential-5','differential5'),math.radians(10)),
              ('p',('p-0.25','p0.25'),.5),('r',('r-0.25','r0.25'),.5)]:
            if not all(p in idx.index for p in pair): continue
            d=(idx.loc[pair[1],AXES]-idx.loc[pair[0],AXES])/delta
            derivatives.append(dict(model=keys[0],frequency=keys[1],component=keys[2],derivative=channel,**d.to_dict()))
    pd.DataFrame(derivatives).to_csv(output/'derivatives.csv',index=False)
    # Mirror about y=0; translate moments first because reference has measured lateral offset.
    symmetry=[]
    for keys,group in phase.groupby(['model','frequency','component']):
        if keys[2] not in ['wings','tail','aircraft','body']: continue
        for name, other in [('beta5','beta-5'),('beta10','beta-10'),('rudder5','rudder-5'),
                            ('differential5','differential-5'),('p0.25','p-0.25'),('r0.25','r-0.25')]:
            a=group[group.case.eq(name)][AXES].to_numpy().copy()
            b=group[group.case.eq(other)][AXES].to_numpy().copy()
            for arr in [a,b]:
                arr[:,3:]+=np.cross(np.array([0.,float(CG[1]),0.]),arr[:,:3])
            if not len(a) or not len(b): continue
            err=np.max(abs(a-b*np.array([1,-1,1,-1,1,-1])),axis=0)
            symmetry.append(dict(model=keys[0],frequency=keys[1],component=keys[2],pair=name,**dict(zip(AXES,err))))
    pd.DataFrame(symmetry).to_csv(output/'symmetry.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for ri,f in enumerate([3,4]):
        for model in means.model.unique():
            s=means[(means.frequency==f)&(means.model==model)&(means.component=='aircraft')&
                (means.case.isin(['neutral','beta-10','beta-5','beta5','beta10']))].sort_values('beta')
            for col,a in enumerate(['Fy','Mx','Mz']):
                axes[ri,col].plot(s.beta,s[a],'-o',label=model)
                axes[ri,col].set(xlabel='beta FLU (deg)',ylabel=a+(' (N)' if a=='Fy' else ' (Nm)'),title=f'{f} Hz, aircraft')
                axes[ri,col].grid(alpha=.3)
        axes[ri,0].legend()
    fig.savefig(output/'crosswind.png',dpi=160); plt.close(fig)
    fig, axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for ri,f in enumerate([3,4]):
        for model in means.model.unique():
            s=phase[(phase.frequency==f)&(phase.model==model)&(phase.component=='wings')&(phase.case=='beta10')]
            for col,a in enumerate(['Fy','Mx','Mz']):
                axes[ri,col].plot(s.phase_deg,s[a],label=model)
                axes[ri,col].set(xlabel='mechanical phase (deg)',ylabel=a+(' (N)' if a=='Fy' else ' (Nm)'),title=f'{f} Hz, wings, beta=10 deg')
                axes[ri,col].grid(alpha=.3)
        axes[ri,0].legend()
    fig.savefig(output/'phase.png',dpi=160); plt.close(fig)
    fig, axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for col,(channel,a) in enumerate([('rudder','Mz'),('differential','Mx'),('symmetric','My')]):
        s=means[(means.frequency==3)&(means.model=='delaurier')&(means.component=='tail')&
            (means.case.isin(['neutral']+[channel+str(v) for v in [-10,-5,5,10]]))].sort_values(channel)
        axes[col].plot(s[channel],s[a],'-o'); axes[col].set(xlabel=channel+' (deg)',ylabel=a+' (Nm)'); axes[col].grid(alpha=.3)
    fig.savefig(output/'control.png',dpi=160); plt.close(fig)
    fig, axes = plt.subplots(len(means.model.unique()), 3, figsize=(13, 9), layout='constrained', squeeze=False)
    for row, model in enumerate(means.model.unique()):
        for part, style in [('left_wing', '-'), ('right_wing', '--'), ('wings', ':')]:
            s = phase[(phase.frequency == 4) & (phase.model == model) & (phase.component == part) & (phase.case == 'beta10')]
            for col, axis in enumerate(['Fy', 'Mx', 'Mz']):
                axes[row, col].plot(s.phase_deg, s[axis], style, label=part)
                axes[row, col].set(xlabel='phase (deg)', ylabel=axis+(' (N)' if axis=='Fy' else ' (Nm)'), title=model + ', 4 Hz, beta=10 deg')
                axes[row, col].grid(alpha=.3)
        axes[row, 0].legend(fontsize=8)
    fig.savefig(output/'left_right.png', dpi=160)
    plt.close(fig)
