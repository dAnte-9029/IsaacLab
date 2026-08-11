"""Run the isolated CPU-native versus GPU-implicit tail-servo gate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Mapping, Sequence

import numpy as np


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_PROJECT_SOURCE = _REPO_ROOT / "source/flapping_bot"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_PROJECT_SOURCE) not in sys.path:
    sys.path.insert(0, str(_PROJECT_SOURCE))

from pure_rl_backend_diagnostics import compare_paired_tail_servo_traces


_BACKEND_DEVICES = {
    "cpu_native_authority": "cpu",
    "gpu_implicit_candidate": "cuda",
}
_MODES = ("no_aero", "aero")
_SURFACE_NAMES = ("rudder", "left_elevon", "right_elevon")
_PHYSICS_DT_S = 1.0 / 480.0
_POLICY_DECIMATION = 8
_DWELL_S = 0.75
_TAIL_AMPLITUDE_DEG = 20.0
_TAIL_LEVEL_SIGNS = (0.0, 1.0, 0.0, -1.0, 0.0)
_FLAP_FREQUENCY_HZ = 2.66958
_AIRSPEED_MPS = 8.0
_MAX_SOFT_LIMIT_VIOLATION_DEG = 0.25
_GPU_CANDIDATE = "phase_matched"


@dataclass(frozen=True)
class WorkerJob:
    """One backend and aerodynamic mode executed in a fresh process."""

    mode: str
    backend_id: str
    device: str
    output_dir: Path
    command: tuple[str, ...]


def validate_worker_request(
    *,
    backend_id: str,
    device: str,
    mode: str,
    seed: int,
) -> None:
    """Fail closed when a worker differs from the frozen experiment contract."""

    if backend_id not in _BACKEND_DEVICES:
        raise ValueError(f"unknown tail-servo backend: {backend_id}")
    if not str(device).startswith(_BACKEND_DEVICES[backend_id]):
        raise ValueError(f"backend {backend_id} is incompatible with device {device}")
    if mode not in _MODES:
        raise ValueError(f"tail-servo mode must be one of {_MODES}")
    if int(seed) < 0:
        raise ValueError("tail-servo seed must be nonnegative")


def tail_level_sign(time_s: float, *, dwell_s: float = _DWELL_S) -> float:
    """Return the zero-order-held tail command level for nonnegative time."""

    time_value = float(time_s)
    dwell_value = float(dwell_s)
    if time_value < 0.0:
        raise ValueError("time_s must be nonnegative")
    if dwell_value <= 0.0:
        raise ValueError("dwell_s must be positive")
    index = min(int(time_value / dwell_value), len(_TAIL_LEVEL_SIGNS) - 1)
    return _TAIL_LEVEL_SIGNS[index]


def tensor_snapshot(value) -> np.ndarray:
    """Return an owning NumPy snapshot for either CPU or GPU tensors."""

    return value.detach().cpu().numpy().copy()


def build_worker_jobs(
    *,
    script_path: Path,
    output_dir: Path,
    cuda_device: str,
    seed: int,
    headless: bool,
    python_executable: str | None = None,
) -> tuple[WorkerJob, ...]:
    """Build the four required fresh-process worker commands."""

    executable = python_executable or sys.executable
    jobs: list[WorkerJob] = []
    for mode in _MODES:
        for backend_id in ("cpu_native_authority", "gpu_implicit_candidate"):
            device = "cpu" if backend_id == "cpu_native_authority" else str(cuda_device)
            worker_output = output_dir / f"{mode}_{backend_id}"
            command = [
                executable,
                str(script_path.resolve()),
                "--worker",
                "--backend-id",
                backend_id,
                "--mode",
                mode,
                "--seed",
                str(seed),
                "--output-dir",
                str(worker_output),
                "--device",
                device,
            ]
            if headless:
                command.append("--headless")
            jobs.append(
                WorkerJob(
                    mode=mode,
                    backend_id=backend_id,
                    device=device,
                    output_dir=worker_output,
                    command=tuple(command),
                )
            )
    return tuple(jobs)


def worker_subprocess_environment(
    base_environment: Mapping[str, str],
    *,
    project_source: Path,
) -> dict[str, str]:
    """Prefer this worktree's project package in fresh worker processes."""

    environment = dict(base_environment)
    entries = [str(project_source.resolve())]
    existing = environment.get("PYTHONPATH", "")
    if existing:
        entries.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(entries)
    return environment


def _parse_parent_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cuda-device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args(argv)


def _parse_worker_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--backend-id", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args(argv)
    return args


def _configure_environment(args: argparse.Namespace):
    from flapping_bot.direct.flapping_bot.straight_flight_env import (
        FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg,
        FlappingBotStraightFlightDeLaurierMeasuredPureRLGpuPhaseMatchedEnvCfg,
    )

    if args.backend_id == "cpu_native_authority":
        cfg = FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg()
    else:
        cfg = FlappingBotStraightFlightDeLaurierMeasuredPureRLGpuPhaseMatchedEnvCfg()
    aerodynamics_enabled = args.mode == "aero"
    cfg.seed = int(args.seed)
    cfg.scene.num_envs = len(_SURFACE_NAMES)
    cfg.scene.env_spacing = 5.0
    cfg.scene.replicate_physics = args.backend_id == "gpu_implicit_candidate"
    cfg.sim.device = str(args.device)
    cfg.sim.dt = _PHYSICS_DT_S
    cfg.sim.render_interval = _POLICY_DECIMATION
    cfg.sim.gravity = (0.0, 0.0, 0.0)
    cfg.decimation = _POLICY_DECIMATION
    cfg.episode_length_s = _DWELL_S * len(_TAIL_LEVEL_SIGNS) + 0.25
    cfg.freeze_steps_after_reset = 0
    cfg.reset_forward_speed_mps = 0.0
    cfg.reset_pitch_deg = 0.0
    cfg.reset_flap_hz = _FLAP_FREQUENCY_HZ
    cfg.reset_rudder_deg = 0.0
    cfg.reset_elevon_pitch_deg = 0.0
    cfg.reset_elevon_roll_deg = 0.0
    cfg.randomize_straight_line_heading = False
    cfg.randomize_flap_phase_at_reset = False
    cfg.randomize_commands = False
    cfg.frequency_governor_enabled = False
    cfg.act_lpf_tau_s = 0.0
    cfg.act_rate_limit_per_s = 0.0
    cfg.enable_wing_aero = aerodynamics_enabled
    cfg.enable_tail_aero = aerodynamics_enabled
    cfg.tail_aero_deflection_source = "actual_joint_position"
    cfg.fuselage_drag_cda = 0.0
    if not aerodynamics_enabled:
        cfg.wing_aero_coupling_mode = "commanded_base_equivalent"
    cfg.wind_enabled = aerodynamics_enabled
    cfg.randomize_wind = False
    cfg.wind_ou_enabled = False
    cfg.wind_curriculum_enabled = False
    cfg.wind_xy_mps = (-_AIRSPEED_MPS, 0.0) if aerodynamics_enabled else (0.0, 0.0)
    cfg.robot = cfg.robot.replace(
        spawn=cfg.robot.spawn.replace(
            usd_dir=str(args.output_dir / "generated_assets"),
            rigid_props=cfg.robot.spawn.rigid_props.replace(
                disable_gravity=True,
                retain_accelerations=False,
            ),
            articulation_props=cfg.robot.spawn.articulation_props.replace(fix_root_link=True),
        )
    )
    return cfg


def _tail_actions(env: object, *, time_s: float):
    import torch

    from flapping_bot.direct.flapping_bot.action_contract import (
        frequency_hz_to_normalized_action,
        joint_position_to_normalized_action,
    )

    actions = torch.zeros((len(_SURFACE_NAMES), 4), dtype=torch.float32, device=env.device)
    actions[:, 0] = frequency_hz_to_normalized_action(
        torch.full((len(_SURFACE_NAMES),), _FLAP_FREQUENCY_HZ, device=env.device),
        minimum_frequency_hz=float(env.cfg.min_flap_hz),
        maximum_frequency_hz=float(env.cfg.max_flap_hz),
    )
    sign = tail_level_sign(time_s)
    surface_specs = (
        (env._IDX_RUDDER, 1),
        (env._IDX_LEFT_TAIL, 2),
        (env._IDX_RIGHT_TAIL, 3),
    )
    for env_index, (local_joint_index, action_index) in enumerate(surface_specs):
        lower = env._joint_lower_limits[local_joint_index]
        upper = env._joint_upper_limits[local_joint_index]
        target = torch.clamp(
            torch.as_tensor(
                math.radians(_TAIL_AMPLITUDE_DEG) * sign,
                device=env.device,
                dtype=lower.dtype,
            ),
            min=lower,
            max=upper,
        )
        actions[env_index, action_index] = joint_position_to_normalized_action(
            target,
            lower_limit_rad=lower,
            upper_limit_rad=upper,
        )
    return actions


def _run_worker(args: argparse.Namespace) -> int:
    from isaaclab.app import AppLauncher

    validate_worker_request(
        backend_id=str(args.backend_id),
        device=str(args.device),
        mode=str(args.mode),
        seed=int(args.seed),
    )
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)

    if args.backend_id == "cpu_native_authority":
        extension_parent = _REPO_ROOT / "source/flapping_bot/native_extensions"
        required = (
            f"--ext-folder {extension_parent} "
            "--enable omni.flapping_bot.holonomic_constraint"
        )
        args.kit_args = f"{str(getattr(args, 'kit_args', '') or '').strip()} {required}".strip()

    simulation_app = None
    env = None
    try:
        simulation_app = AppLauncher(args).app
        import torch

        from flapping_bot.analysis.pure_rl_action_step_validation import summarize_step_response
        from flapping_bot.direct.flapping_bot.straight_flight_env import FlappingBotStraightFlightEnv
        from isaaclab.utils.math import quat_apply_inverse

        cfg = _configure_environment(args)
        env = FlappingBotStraightFlightEnv(cfg)
        env.reset()
        if str(env.device) != str(args.device):
            raise RuntimeError(f"environment device mismatch: {env.device} != {args.device}")
        if bool(env.cfg.scene.replicate_physics) != (
            args.backend_id == "gpu_implicit_candidate"
        ):
            raise RuntimeError("physics replication does not match backend contract")
        if str(env.cfg.tail_aero_deflection_source) != "actual_joint_position":
            raise RuntimeError("tail_aero_deflection_source does not use actual_joint_position")

        env._wind_w.zero_()
        if args.mode == "aero":
            env._wind_w[:, 0] = -_AIRSPEED_MPS
        env._wind_mean_w.copy_(env._wind_w)

        tail_local_indices = [env._IDX_RUDDER, env._IDX_LEFT_TAIL, env._IDX_RIGHT_TAIL]
        tail_joint_ids = [int(env._joint_ids[index]) for index in tail_local_indices]
        diagonal = torch.arange(len(_SURFACE_NAMES), device=env.device)
        lower_limits = tensor_snapshot(env._joint_lower_limits[tail_local_indices])
        upper_limits = tensor_snapshot(env._joint_upper_limits[tail_local_indices])
        trace: dict[str, list[object]] = {
            "time_s": [],
            "tail_command_rad": [],
            "tail_actual_rad": [],
            "tail_all_actual_rad": [],
            "tail_actual_velocity_rad_s": [],
            "air_velocity_body_mps": [],
            "wind_world_mps": [],
            "root_linear_velocity_body_mps": [],
            "root_angular_velocity_body_rad_s": [],
            "base_com_position_body_m": [],
            "tail_moment_pre_sim_body_nm": [],
            "tail_moment_after_write_body_nm": [],
            "tail_moment_after_sim_body_nm": [],
            "tail_moment_after_scene_update_body_nm": [],
            "tail_moment_after_recompute_body_nm": [],
            "tail_moment_body_nm": [],
            "tail_recomputed_moment_body_nm": [],
        }
        actions = _tail_actions(env, time_s=0.0)
        total_steps = round(_DWELL_S * len(_TAIL_LEVEL_SIGNS) / _PHYSICS_DT_S)
        for step in range(total_steps):
            if step % _POLICY_DECIMATION == 0:
                actions = _tail_actions(env, time_s=step * _PHYSICS_DT_S)
                env._pre_physics_step(actions)
            env._sim_step_counter += 1
            env._apply_action()
            tail_moment_pre_sim = (
                env._debug_last_tail_moment_b_about_base_com_nm.detach().clone()
            )
            env.scene.write_data_to_sim()
            tail_moment_after_write = (
                env._debug_last_tail_moment_b_about_base_com_nm.detach().clone()
            )
            env.sim.step(render=False)
            tail_moment_after_sim = (
                env._debug_last_tail_moment_b_about_base_com_nm.detach().clone()
            )
            env.scene.update(dt=env.physics_dt)
            tail_moment_after_scene_update = (
                env._debug_last_tail_moment_b_about_base_com_nm.detach().clone()
            )

            tail_command = torch.stack(
                (env._rudder_cmd, env._left_elevon_cmd, env._right_elevon_cmd),
                dim=1,
            )
            tail_actual = env._robot.data.joint_pos[:, tail_joint_ids]
            tail_velocity = env._robot.data.joint_vel[:, tail_joint_ids]
            wind_body = quat_apply_inverse(env._robot.data.root_quat_w, env._wind_w)
            air_velocity_body = env._robot.data.root_lin_vel_b - wind_body
            base_com_position_body = env._robot.data.body_com_pos_b[
                :, int(env._base_body_ids[0]), :
            ]
            _, tail_recomputed_moment = env._tail_model.compute_wrench(
                root_lin_vel_b=air_velocity_body,
                root_ang_vel_b=env._robot.data.root_ang_vel_b,
                left_elevon_rad=tail_actual[:, 1],
                right_elevon_rad=tail_actual[:, 2],
                rudder_rad=tail_actual[:, 0],
                base_com_pos_b=base_com_position_body,
            )
            tail_moment_after_recompute = (
                env._debug_last_tail_moment_b_about_base_com_nm.detach().clone()
            )
            finite_tensors = (
                env._robot.data.root_state_w,
                tail_command,
                tail_actual,
                tail_velocity,
                air_velocity_body,
                env._wind_w,
                base_com_position_body,
                tail_moment_pre_sim,
                tail_moment_after_write,
                tail_moment_after_sim,
                tail_moment_after_scene_update,
                tail_moment_after_recompute,
                env._debug_last_tail_moment_b_about_base_com_nm,
                tail_recomputed_moment,
            )
            if any(not bool(torch.all(torch.isfinite(value))) for value in finite_tensors):
                raise RuntimeError(f"nonfinite tail-servo state at physics step {step}")
            trace["time_s"].append((step + 1) * _PHYSICS_DT_S)
            trace["tail_command_rad"].append(
                tensor_snapshot(tail_command[diagonal, diagonal])
            )
            trace["tail_actual_rad"].append(
                tensor_snapshot(tail_actual[diagonal, diagonal])
            )
            trace["tail_all_actual_rad"].append(tensor_snapshot(tail_actual))
            trace["tail_actual_velocity_rad_s"].append(
                tensor_snapshot(tail_velocity[diagonal, diagonal])
            )
            trace["air_velocity_body_mps"].append(
                tensor_snapshot(air_velocity_body)
            )
            trace["wind_world_mps"].append(tensor_snapshot(env._wind_w))
            trace["root_linear_velocity_body_mps"].append(
                tensor_snapshot(env._robot.data.root_lin_vel_b)
            )
            trace["root_angular_velocity_body_rad_s"].append(
                tensor_snapshot(env._robot.data.root_ang_vel_b)
            )
            trace["base_com_position_body_m"].append(
                tensor_snapshot(base_com_position_body)
            )
            trace["tail_moment_pre_sim_body_nm"].append(
                tensor_snapshot(tail_moment_pre_sim)
            )
            trace["tail_moment_after_write_body_nm"].append(
                tensor_snapshot(tail_moment_after_write)
            )
            trace["tail_moment_after_sim_body_nm"].append(
                tensor_snapshot(tail_moment_after_sim)
            )
            trace["tail_moment_after_scene_update_body_nm"].append(
                tensor_snapshot(tail_moment_after_scene_update)
            )
            trace["tail_moment_after_recompute_body_nm"].append(
                tensor_snapshot(tail_moment_after_recompute)
            )
            trace["tail_moment_body_nm"].append(
                tensor_snapshot(env._debug_last_tail_moment_b_about_base_com_nm)
            )
            trace["tail_recomputed_moment_body_nm"].append(
                tensor_snapshot(tail_recomputed_moment)
            )

        arrays = {name: np.asarray(values, dtype=np.float64) for name, values in trace.items()}
        responses = {
            name: summarize_step_response(
                arrays["time_s"],
                arrays["tail_command_rad"][:, index],
                arrays["tail_actual_rad"][:, index],
            )
            for index, name in enumerate(_SURFACE_NAMES)
        }
        limit_violation = np.maximum(
            np.maximum(lower_limits[None, :] - arrays["tail_actual_rad"], 0.0),
            np.maximum(arrays["tail_actual_rad"] - upper_limits[None, :], 0.0),
        )
        trace_path = output_dir / "trace.npz"
        np.savez_compressed(trace_path, **arrays)
        summary = {
            "schema_version": "pure_rl_tail_servo_worker_v1",
            "completed": True,
            "backend_id": str(args.backend_id),
            "device": str(args.device),
            "mode": str(args.mode),
            "gpu_candidate": _GPU_CANDIDATE,
            "seed": int(args.seed),
            "physics_hz": 1.0 / _PHYSICS_DT_S,
            "policy_hz": 1.0 / (_PHYSICS_DT_S * _POLICY_DECIMATION),
            "dwell_s": _DWELL_S,
            "tail_amplitude_deg": _TAIL_AMPLITUDE_DEG,
            "airspeed_mps": _AIRSPEED_MPS if args.mode == "aero" else 0.0,
            "sample_count": int(arrays["time_s"].size),
            "mean_air_speed_mps": float(
                np.mean(np.linalg.norm(arrays["air_velocity_body_mps"], axis=2))
            ),
            "max_air_speed_mps": float(
                np.max(np.linalg.norm(arrays["air_velocity_body_mps"], axis=2))
            ),
            "wing_drive_variant": str(env.cfg.wing_drive_variant),
            "tail_aero_deflection_source": str(env.cfg.tail_aero_deflection_source),
            "max_soft_limit_violation_deg": math.degrees(float(np.max(limit_violation))),
            "responses": responses,
            "trace_path": str(trace_path),
        }
        result_path = output_dir / "result.json"
        result_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest = {
            "completed": True,
            "backend_id": str(args.backend_id),
            "mode": str(args.mode),
            "result_path": str(result_path),
            "trace_path": str(trace_path),
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return 0
    except BaseException as error:
        (output_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "completed": False,
                    "backend_id": str(args.backend_id),
                    "mode": str(args.mode),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        traceback.print_exception(type(error), error, error.__traceback__)
        return 1
    finally:
        if env is not None:
            env.close()
        if simulation_app is not None:
            simulation_app.close()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _run_parent(args: argparse.Namespace) -> int:
    if int(args.seed) < 0:
        raise ValueError("tail-servo seed must be nonnegative")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    jobs = build_worker_jobs(
        script_path=Path(__file__),
        output_dir=output_dir,
        cuda_device=str(args.cuda_device),
        seed=int(args.seed),
        headless=bool(args.headless),
    )
    worker_environment = worker_subprocess_environment(
        os.environ,
        project_source=_PROJECT_SOURCE,
    )
    records: list[dict[str, object]] = []
    runtime_failure = False
    for job in jobs:
        print(f"[tail-servo-gate] {job.mode} {job.backend_id} on {job.device}", flush=True)
        returncode = subprocess.run(
            job.command,
            cwd=_REPO_ROOT,
            check=False,
            env=worker_environment,
        ).returncode
        manifest_path = job.output_dir / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else {"completed": False, "error": "worker manifest missing"}
        )
        records.append(
            {
                "mode": job.mode,
                "backend_id": job.backend_id,
                "device": job.device,
                "returncode": returncode,
                "manifest": manifest,
            }
        )
        if returncode != 0 or not bool(manifest.get("completed", False)):
            runtime_failure = True
            break

    mode_results: list[dict[str, object]] = []
    if not runtime_failure:
        for mode in _MODES:
            mode_jobs = {job.backend_id: job for job in jobs if job.mode == mode}
            cpu_job = mode_jobs["cpu_native_authority"]
            gpu_job = mode_jobs["gpu_implicit_candidate"]
            cpu_summary = json.loads((cpu_job.output_dir / "result.json").read_text(encoding="utf-8"))
            gpu_summary = json.loads((gpu_job.output_dir / "result.json").read_text(encoding="utf-8"))
            comparison = compare_paired_tail_servo_traces(
                _load_npz(cpu_job.output_dir / "trace.npz"),
                _load_npz(gpu_job.output_dir / "trace.npz"),
                aerodynamics_enabled=mode == "aero",
            )
            failures = list(comparison["gate_failures"])
            for backend_id, worker_summary in (
                ("cpu_native_authority", cpu_summary),
                ("gpu_implicit_candidate", gpu_summary),
            ):
                if float(worker_summary["max_soft_limit_violation_deg"]) > _MAX_SOFT_LIMIT_VIOLATION_DEG:
                    failures.append(f"{backend_id}.max_soft_limit_violation_deg")
            mode_results.append(
                {
                    "mode": mode,
                    "airspeed_mps": _AIRSPEED_MPS if mode == "aero" else 0.0,
                    "cpu_worker": cpu_summary,
                    "gpu_worker": gpu_summary,
                    "comparison": comparison,
                    "gate_failures": failures,
                    "gate_passed": not failures,
                }
            )

    all_modes_passed = bool(
        not runtime_failure
        and len(mode_results) == len(_MODES)
        and all(bool(result["gate_passed"]) for result in mode_results)
    )
    summary = {
        "schema_version": "pure_rl_tail_servo_gate_v1",
        "completed": not runtime_failure and len(mode_results) == len(_MODES),
        "tail_servo_gate_passed": all_modes_passed,
        "runtime_failure": runtime_failure,
        "gpu_candidate": _GPU_CANDIDATE,
        "seed": int(args.seed),
        "physics_hz": 1.0 / _PHYSICS_DT_S,
        "policy_hz": 1.0 / (_PHYSICS_DT_S * _POLICY_DECIMATION),
        "tail_amplitude_deg": _TAIL_AMPLITUDE_DEG,
        "mode_results": mode_results,
        "worker_records": records,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "completed": bool(summary["completed"]),
        "tail_servo_gate_passed": all_modes_passed,
        "gpu_candidate": _GPU_CANDIDATE,
        "seed": int(args.seed),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary_path": str(summary_path),
        "workers": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    if runtime_failure:
        return 1
    return 0 if all_modes_passed else 2


def main(argv: Sequence[str] | None = None) -> int:
    resolved_argv = list(sys.argv[1:] if argv is None else argv)
    if "--worker" in resolved_argv:
        return _run_worker(_parse_worker_args(resolved_argv))
    return _run_parent(_parse_parent_args(resolved_argv))


if __name__ == "__main__":
    raise SystemExit(main())
