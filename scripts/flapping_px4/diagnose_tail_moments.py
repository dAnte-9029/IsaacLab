"""Read-only physics-step instrumentation of the unchanged historical pulse runner."""
import sys
import dataclasses
import json
from pathlib import Path
import numpy as np


def install(env, output):
    import torch
    from isaaclab.utils.math import quat_apply_inverse
    rows=[]; slot=1
    def arr(x): return x.detach().cpu().numpy().copy()
    d=env._robot.data; view=env._robot.root_physx_view
    meta=dict(tail=dataclasses.asdict(env._tail_model.cfg),body_names=env._robot.body_names,
              joint_names=env._robot.joint_names,masses=arr(view.get_masses())[slot].tolist(),
              inertias=arr(view.get_inertias())[slot].tolist(),coms=arr(view.get_coms())[slot].tolist(),
              base_id=int(env._base_body_ids[0]),wing_ids=list(map(int,env._wing_body_ids)),
              dt=env.physics_dt,tail_bias_deg=env.cfg.tail_elevator_bias_deg,
              settings={k:getattr(env.cfg,k) for k in ('wing_aero_coupling_mode','wing_link_aero_load_mode','virtual_pitch_moment_gain','virtual_pitch_moment_damping','virtual_roll_moment_gain','virtual_roll_moment_damping','native_holonomic_load_diagnostics')})
    (output/'parameters.json').write_text(json.dumps(meta,indent=2))
    original=env._apply_action
    def capture():
        original()
        q=d.root_quat_w
        row={k:arr(v)[slot] for k,v in dict(quat=q,position=d.root_pos_w,velocity=d.root_lin_vel_w,
            omega=d.root_ang_vel_b,air=d.root_lin_vel_b-quat_apply_inverse(q,env._wind_w),wind=env._wind_w,
            joint=d.joint_pos,joint_velocity=d.joint_vel,phase=env._phase,frequency=env._freq,
            body_com=d.body_com_pos_w,body_quat=d.body_link_quat_w,body_velocity=d.body_com_lin_vel_w,
            body_omega=d.body_com_ang_vel_w,base_com_b=d.body_com_pos_b[:,meta['base_id']],
            body_link_pos=d.body_link_pos_w,applied_torque=d.applied_torque).items()}
        for k in ('wing_force_b','wing_moment_b_about_base_com_nm','tail_force_b','tail_moment_b_about_base_com_nm','force_b','torque_b','wing_force_link_n','wing_moment_link_about_com_nm'):
            row[k]=arr(getattr(env,'_debug_last_'+k))[slot]
        kw=dict(root_lin_vel_b=torch.as_tensor(row['air'])[None],root_ang_vel_b=torch.as_tensor(row['omega'])[None],
            base_com_pos_b=torch.as_tensor(row['base_com_b'])[None],rudder_rad=env._debug_last_actual_rudder_rad[slot:slot+1],
            left_elevon_rad=env._debug_last_actual_left_elevon_rad[slot:slot+1]+np.deg2rad(meta['tail_bias_deg']),
            right_elevon_rad=env._debug_last_actual_right_elevon_rad[slot:slot+1]+np.deg2rad(meta['tail_bias_deg']))
        surfaces=env._tail_model.compute_surface_results(**kw)
        for field in ('force_b','moment_b_about_com','local_velocity_b','alpha_raw_rad','alpha_rad','dynamic_pressure_pa','lift_coefficient','drag_coefficient','deflection_rad','aerodynamic_center_b_from_com'):
            row['surface_'+field]=np.stack([arr(getattr(s,field))[0] for s in surfaces])
        rows.append(row)
    env._apply_action=capture
    def save():
        np.savez_compressed(output/'physics.npz',**{k:np.stack([r[k] for r in rows]) for k in rows[0]})
    return save


def main():
    # Reuse original child verbatim except attaching the observer after reset and saving before its summary.
    if "--group" not in sys.argv:
        raise ValueError("Specify one --group and --start; use reproduce_tail_moments.sh for the six-run protocol")
    import run_sustained_tail_pulses as original
    source=Path(original.__file__).read_text()
    assert source.count("obs,_=env.reset();check_initial")==1
    assert source.count("write(args.output/'summary.json',dict(schema=SCHEMA,passed=True")==1
    source=source.replace("obs,_=env.reset();check_initial", "obs,_=env.reset();save_diagnostics=install(env,args.output);check_initial")
    source=source.replace("write(args.output/'summary.json',dict(schema=SCHEMA,passed=True", "save_diagnostics()\n        write(args.output/'summary.json',dict(schema=SCHEMA,passed=True")
    scope=dict(__name__='diagnostic_replay',__file__=original.__file__,install=install)
    exec(compile(source,original.__file__,'exec'),scope)
    scope['main']()

if __name__=='__main__': main()
