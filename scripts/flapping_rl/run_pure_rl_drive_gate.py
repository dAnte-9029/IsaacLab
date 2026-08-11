"""Run the frozen CPU-native versus GPU-implicit fixed-root wing-drive gate."""

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
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from pure_rl_backend_diagnostics import summarize_drive_gate_trace


_BACKEND_DEVICES = {
    "cpu_native_authority": "cpu",
    "gpu_implicit_candidate": "cuda",
}
_FREQUENCIES_HZ = (4.0, 5.0)
_PHYSICS_DT_S = 1.0 / 480.0
_DURATION_S = 1.0
_AIRSPEED_MPS = 8.0
_GPU_CANDIDATES = ("baseline", "phase_matched")


@dataclass(frozen=True)
class WorkerJob:
    """One backend/frequency case executed in a fresh Isaac process."""

    backend_id: str
    frequency_hz: float
    device: str
    output_dir: Path
    command: tuple[str, ...]


def validate_worker_request(
    *,
    backend_id: str,
    device: str,
    frequency_hz: float,
    seed: int,
) -> None:
    """Fail closed when a worker differs from the frozen drive-gate contract."""

    if backend_id not in _BACKEND_DEVICES:
        raise ValueError(f"unknown drive backend: {backend_id}")
    if not str(device).startswith(_BACKEND_DEVICES[backend_id]):
        raise ValueError(f"backend {backend_id} is incompatible with device {device}")
    if float(frequency_hz) not in _FREQUENCIES_HZ:
        raise ValueError(f"drive frequency must be one of {_FREQUENCIES_HZ}")
    if int(seed) < 0:
        raise ValueError("drive-gate seed must be nonnegative")


def build_worker_jobs(
    *,
    script_path: Path,
    output_dir: Path,
    cuda_device: str,
    seed: int,
    headless: bool,
    gpu_candidate: str = "baseline",
    python_executable: str | None = None,
) -> tuple[WorkerJob, ...]:
    """Build the four required fresh-process worker commands."""

    executable = python_executable or sys.executable
    if gpu_candidate not in _GPU_CANDIDATES:
        raise ValueError(f"unknown GPU candidate: {gpu_candidate!r}")
    jobs: list[WorkerJob] = []
    for backend_id in ("cpu_native_authority", "gpu_implicit_candidate"):
        device = "cpu" if backend_id == "cpu_native_authority" else str(cuda_device)
        for frequency_hz in _FREQUENCIES_HZ:
            case_output = output_dir / f"{backend_id}_{int(frequency_hz)}hz"
            command = [
                executable,
                str(script_path.resolve()),
                "--worker",
                "--backend-id",
                backend_id,
                "--frequency-hz",
                f"{frequency_hz:g}",
                "--seed",
                str(seed),
                "--output-dir",
                str(case_output),
                "--device",
                device,
            ]
            if headless:
                command.append("--headless")
            if backend_id == "gpu_implicit_candidate":
                command.extend(("--gpu-candidate", gpu_candidate))
            jobs.append(
                WorkerJob(
                    backend_id=backend_id,
                    frequency_hz=frequency_hz,
                    device=device,
                    output_dir=case_output,
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
    existing = environment.get("PYTHONPATH", "")
    entries = [str(project_source.resolve())]
    if existing:
        entries.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(entries)
    return environment


def prescribed_target_position_rad(
    *,
    time_s: float,
    frequency_hz: float,
    amplitude_rad: float,
) -> float:
    """Return the common wing target at the post-step state timestamp."""

    return float(amplitude_rad) * math.sin(
        2.0 * math.pi * float(frequency_hz) * float(time_s)
    )


def _parse_parent_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cuda-device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-candidate", choices=_GPU_CANDIDATES, default="baseline")
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args(argv)


def _parse_worker_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--backend-id", required=True)
    parser.add_argument("--frequency-hz", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-candidate", choices=_GPU_CANDIDATES, default="baseline")
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args(argv)
    return args


def _configure_environment(args: argparse.Namespace):
    from flapping_bot.direct.flapping_bot.straight_flight_env import (
        FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg,
        FlappingBotStraightFlightDeLaurierMeasuredPureRLGpuImplicitEnvCfg,
        FlappingBotStraightFlightDeLaurierMeasuredPureRLGpuPhaseMatchedEnvCfg,
    )

    if args.backend_id == "cpu_native_authority":
        cfg = FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg()
    elif args.gpu_candidate == "baseline":
        cfg = FlappingBotStraightFlightDeLaurierMeasuredPureRLGpuImplicitEnvCfg()
    else:
        cfg = FlappingBotStraightFlightDeLaurierMeasuredPureRLGpuPhaseMatchedEnvCfg()
    cfg.seed = int(args.seed)
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 5.0
    cfg.scene.replicate_physics = args.backend_id == "gpu_implicit_candidate"
    cfg.sim.device = str(args.device)
    cfg.sim.dt = _PHYSICS_DT_S
    cfg.sim.render_interval = 1
    cfg.sim.gravity = (0.0, 0.0, 0.0)
    cfg.decimation = 1
    cfg.episode_length_s = 2.0
    cfg.freeze_steps_after_reset = 0
    cfg.reset_forward_speed_mps = 0.0
    cfg.reset_pitch_deg = 0.0
    cfg.reset_flap_hz = float(args.frequency_hz)
    cfg.reset_rudder_deg = 0.0
    cfg.reset_elevon_pitch_deg = 0.0
    cfg.reset_elevon_roll_deg = 0.0
    cfg.randomize_straight_line_heading = False
    cfg.randomize_flap_phase_at_reset = False
    cfg.randomize_commands = False
    cfg.enable_tail_aero = False
    cfg.fuselage_drag_cda = 0.0
    cfg.frequency_governor_enabled = False
    cfg.act_lpf_tau_s = 0.0
    cfg.act_rate_limit_per_s = 0.0
    cfg.wind_enabled = True
    cfg.randomize_wind = False
    cfg.wind_ou_enabled = False
    cfg.wind_curriculum_enabled = False
    cfg.wind_xy_mps = (-_AIRSPEED_MPS, 0.0)
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


def _frequency_action(*, frequency_hz: float, device: str):
    import torch

    actions = torch.zeros((1, 4), dtype=torch.float32, device=device)
    actions[0, 0] = 2.0 * float(frequency_hz) / 5.0 - 1.0
    return actions


def _run_worker(args: argparse.Namespace) -> int:
    from isaaclab.app import AppLauncher

    validate_worker_request(
        backend_id=str(args.backend_id),
        device=str(args.device),
        frequency_hz=float(args.frequency_hz),
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

        from flapping_bot.direct.flapping_bot.straight_flight_env import FlappingBotStraightFlightEnv

        cfg = _configure_environment(args)
        env = FlappingBotStraightFlightEnv(cfg)
        env.reset()
        if str(env.device) != str(args.device):
            raise RuntimeError(f"environment device mismatch: {env.device} != {args.device}")
        expected_replication = args.backend_id == "gpu_implicit_candidate"
        if bool(env.cfg.scene.replicate_physics) != expected_replication:
            raise RuntimeError("physics replication does not match backend contract")

        env._wind_w.zero_()
        env._wind_w[:, 0] = -_AIRSPEED_MPS
        env._wind_mean_w.copy_(env._wind_w)
        env._pre_physics_step(
            _frequency_action(frequency_hz=float(args.frequency_hz), device=str(env.device))
        )

        left_joint_id = int(env._joint_ids[env._IDX_LEFT_WING])
        right_joint_id = int(env._joint_ids[env._IDX_RIGHT_WING])
        time_s: list[float] = []
        target_position_rad: list[float] = []
        drive_command_position_rad: list[float] = []
        actual_position_rad: list[list[float]] = []
        actual_velocity_rad_s: list[list[float]] = []
        actual_frequency_hz: list[float] = []
        wing_force_body_n: list[list[float]] = []
        wing_moment_body_nm: list[list[float]] = []
        total_steps = round(_DURATION_S / _PHYSICS_DT_S)

        for step in range(total_steps):
            env._sim_step_counter += 1
            env._apply_action()
            drive_command = env._q_cmd.detach().clone()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)

            left_position = env._robot.data.joint_pos[:, left_joint_id] - float(env._wing_mid_L)
            right_position = -(
                env._robot.data.joint_pos[:, right_joint_id] - float(env._wing_mid_R)
            )
            left_velocity = env._robot.data.joint_vel[:, left_joint_id]
            right_velocity = -env._robot.data.joint_vel[:, right_joint_id]
            finite_tensors = (
                drive_command,
                left_position,
                right_position,
                left_velocity,
                right_velocity,
                env._debug_last_wing_force_b,
                env._debug_last_torque_b,
            )
            if any(not bool(torch.all(torch.isfinite(value))) for value in finite_tensors):
                raise RuntimeError(f"nonfinite drive-gate state at physics step {step}")

            state_time_s = (step + 1) * _PHYSICS_DT_S
            time_s.append(state_time_s)
            target_position_rad.append(
                prescribed_target_position_rad(
                    time_s=state_time_s,
                    frequency_hz=float(args.frequency_hz),
                    amplitude_rad=float(env._wing_amp),
                )
            )
            drive_command_position_rad.append(float(drive_command[0].item()))
            actual_position_rad.append(
                [float(left_position[0].item()), float(right_position[0].item())]
            )
            actual_velocity_rad_s.append(
                [float(left_velocity[0].item()), float(right_velocity[0].item())]
            )
            actual_frequency_hz.append(float(env._freq[0].item()))
            wing_force_body_n.append(env._debug_last_wing_force_b[0].detach().cpu().tolist())
            wing_moment_body_nm.append(env._debug_last_torque_b[0].detach().cpu().tolist())

        trace = {
            "time_s": np.asarray(time_s, dtype=np.float64),
            "target_position_rad": np.asarray(target_position_rad, dtype=np.float64),
            "drive_command_position_rad": np.asarray(
                drive_command_position_rad,
                dtype=np.float64,
            ),
            "actual_position_rad": np.asarray(actual_position_rad, dtype=np.float64),
            "actual_velocity_rad_s": np.asarray(actual_velocity_rad_s, dtype=np.float64),
            "actual_frequency_hz": np.asarray(actual_frequency_hz, dtype=np.float64),
            "wing_force_body_n": np.asarray(wing_force_body_n, dtype=np.float64),
            "wing_moment_body_nm": np.asarray(wing_moment_body_nm, dtype=np.float64),
        }
        summary = summarize_drive_gate_trace(
            time_s=trace["time_s"],
            target_position_rad=trace["target_position_rad"],
            actual_position_rad=trace["actual_position_rad"],
            frequency_hz=float(args.frequency_hz),
        )
        summary.update(
            {
                "backend_id": str(args.backend_id),
                "device": str(args.device),
                "seed": int(args.seed),
                "gpu_candidate": str(args.gpu_candidate),
                "physics_dt_s": _PHYSICS_DT_S,
                "airspeed_mps": _AIRSPEED_MPS,
                "target_amplitude_deg": math.degrees(float(env._wing_amp)),
                "fixed_root": True,
                "wing_drive_variant": str(env.cfg.wing_drive_variant),
                "wing_aero_coupling_mode": str(env.cfg.wing_aero_coupling_mode),
            }
        )
        trace_path = output_dir / "trace.npz"
        np.savez_compressed(trace_path, **trace)
        (output_dir / "result.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "completed": True,
            "gate_passed": bool(summary["gate_passed"]),
            "backend_id": str(args.backend_id),
            "frequency_hz": float(args.frequency_hz),
            "seed": int(args.seed),
            "result_path": str(output_dir / "result.json"),
            "trace_path": str(trace_path),
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return 0 if bool(summary["gate_passed"]) else 2
    except BaseException as error:
        (output_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "completed": False,
                    "backend_id": str(args.backend_id),
                    "frequency_hz": float(args.frequency_hz),
                    "seed": int(args.seed),
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


def _run_parent(args: argparse.Namespace) -> int:
    if int(args.seed) < 0:
        raise ValueError("drive-gate seed must be nonnegative")
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
        gpu_candidate=str(args.gpu_candidate),
    )

    case_results: list[dict[str, object]] = []
    worker_records: list[dict[str, object]] = []
    runtime_failure = False
    worker_environment = worker_subprocess_environment(
        os.environ,
        project_source=_REPO_ROOT / "source/flapping_bot",
    )
    for job in jobs:
        print(
            f"[drive-gate] {job.backend_id} {job.frequency_hz:g} Hz on {job.device}",
            flush=True,
        )
        completed = subprocess.run(
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
        worker_records.append(
            {
                "backend_id": job.backend_id,
                "frequency_hz": job.frequency_hz,
                "device": job.device,
                "returncode": completed,
                "manifest": manifest,
            }
        )
        if completed not in (0, 2) or not bool(manifest.get("completed", False)):
            runtime_failure = True
            break
        case_results.append(
            json.loads((job.output_dir / "result.json").read_text(encoding="utf-8"))
        )

    all_cases_passed = bool(
        not runtime_failure
        and len(case_results) == len(jobs)
        and all(bool(case["gate_passed"]) for case in case_results)
    )
    summary = {
        "schema_version": "pure_rl_drive_gate_v1",
        "physics_rate_hz": 480.0,
        "duration_s": _DURATION_S,
        "steady_start_s": 0.20,
        "airspeed_mps": _AIRSPEED_MPS,
        "seed": int(args.seed),
        "gpu_candidate": str(args.gpu_candidate),
        "case_count": len(case_results),
        "all_cases_passed": all_cases_passed,
        "runtime_failure": runtime_failure,
        "cases": case_results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "completed": not runtime_failure and len(case_results) == len(jobs),
        "drive_gate_passed": all_cases_passed,
        "seed": int(args.seed),
        "gpu_candidate": str(args.gpu_candidate),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "summary_path": str(output_dir / "summary.json"),
        "workers": worker_records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    if runtime_failure:
        return 1
    return 0 if all_cases_passed else 2


def main(argv: Sequence[str] | None = None) -> int:
    resolved_argv = list(sys.argv[1:] if argv is None else argv)
    if "--worker" in resolved_argv:
        return _run_worker(_parse_worker_args(resolved_argv))
    return _run_parent(_parse_parent_args(resolved_argv))


if __name__ == "__main__":
    raise SystemExit(main())
