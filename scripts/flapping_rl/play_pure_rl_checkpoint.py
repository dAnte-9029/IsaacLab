"""Replay a success-gated MeasuredPureRL checkpoint and create MP4/GIF evidence.

The script keeps the authoritative plant on CPU, loads the project-local native
holonomic extension, fixes one route heading and initial flap phase, renders a
visual target line plus actual trajectory, and writes a provenance manifest.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from typing import Any

from isaaclab.app import AppLauncher


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from pure_rl_eval_common import (  # noqa: E402
    MEASURED_PURE_RL_TASK_ID,
    aggregate_pure_rl_case_row,
    read_pure_rl_step_metrics,
    summarize_pure_rl_episode,
)
from pure_rl_playback_common import (  # noqa: E402
    MEASURED_PURE_RL_C2B_TASK_ID,
    PLAYBACK_SCHEMA_VERSION,
    build_gif_command,
    compute_route_visual_geometry,
    numeric_metrics_are_finite,
    playback_case_succeeded,
    resolve_explicit_checkpoint,
    resolve_playback_case,
    resolve_successful_checkpoint,
    sha256_file,
    validate_new_playback_outputs,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a successful PureRL checkpoint with route visualization and MP4/GIF recording."
    )
    parser.add_argument(
        "--task",
        type=str,
        default=MEASURED_PURE_RL_TASK_ID,
        choices=(MEASURED_PURE_RL_TASK_ID, MEASURED_PURE_RL_C2B_TASK_ID),
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Training run containing the checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Explicit promoted checkpoint inside --run-dir; otherwise use eval/best_checkpoint.json.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for MP4, GIF and JSON manifest.")
    parser.add_argument("--name", type=str, required=True, help="Filename stem for playback artifacts.")
    parser.add_argument("--heading-deg", type=float, default=0.0, help="Fixed world route heading in degrees.")
    parser.add_argument("--flap-phase-deg", type=float, default=0.0, help="Fixed initial flap phase in degrees.")
    parser.add_argument("--duration-s", type=float, default=12.0, help="Maximum replay duration in seconds.")
    parser.add_argument("--record-fps", type=int, default=30, help="MP4 frames per second; must divide policy rate.")
    parser.add_argument("--gif-fps", type=int, default=15, help="GIF frames per second.")
    parser.add_argument("--gif-width", type=int, default=640, help="GIF output width in pixels.")
    parser.add_argument("--width", type=int, default=1280, help="Rendered MP4 width in pixels.")
    parser.add_argument("--height", type=int, default=720, help="Rendered MP4 height in pixels.")
    parser.add_argument("--route-behind-m", type=float, default=10.0)
    parser.add_argument("--route-ahead-m", type=float, default=100.0)
    parser.add_argument("--trail-stride", type=int, default=6, help="Policy steps between trail markers.")
    parser.add_argument("--longitudinal-slope-deg", type=float, default=6.0)
    parser.add_argument("--longitudinal-entry-length-m", type=float, default=17.5)
    parser.add_argument("--longitudinal-slope-length-m", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=0, help="Environment seed for reproducibility.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.duration_s <= 0.0:
        parser.error("--duration-s must be positive.")
    if args.record_fps <= 0 or args.gif_fps <= 0 or args.width <= 0 or args.height <= 0:
        parser.error("Frame rates and render dimensions must be positive.")
    if args.trail_stride <= 0:
        parser.error("--trail-stride must be positive.")
    return args


args_cli = _parse_args()
playback_case = resolve_playback_case(
    task=args_cli.task,
    heading_deg=args_cli.heading_deg,
    flap_phase_deg=args_cli.flap_phase_deg,
    longitudinal_slope_deg=args_cli.longitudinal_slope_deg,
    longitudinal_entry_length_m=args_cli.longitudinal_entry_length_m,
    longitudinal_slope_length_m=args_cli.longitudinal_slope_length_m,
)
selection = (
    resolve_successful_checkpoint(args_cli.run_dir)
    if args_cli.checkpoint is None
    else resolve_explicit_checkpoint(args_cli.run_dir, args_cli.checkpoint)
)
mp4_path, gif_path, manifest_path = validate_new_playback_outputs(args_cli.output_dir, args_cli.name)
ffmpeg_executable = shutil.which("ffmpeg")
if ffmpeg_executable is None:
    raise FileNotFoundError("ffmpeg is required to create GIF output.")

args_cli.device = "cpu"
args_cli.enable_cameras = True
extension_parent = _REPO_ROOT / "source" / "flapping_bot" / "native_extensions"
required_kit_args = (
    f"--ext-folder {extension_parent} "
    "--enable omni.flapping_bot.holonomic_constraint"
)
args_cli.kit_args = " ".join(part for part in (getattr(args_cli, "kit_args", None), required_kit_args) if part)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _git_provenance() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def _spawn_route_visuals(env, heading_rad: float):
    import isaaclab.sim as sim_utils
    import torch
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

    env_origin_w = env.unwrapped.scene.env_origins[0].detach().cpu().tolist()
    target_height_m = float(env.unwrapped._height_cmd[0].item())
    geometry = compute_route_visual_geometry(
        env_origin_w=env_origin_w,
        target_height_m=target_height_m,
        heading_rad=heading_rad,
        behind_m=float(args_cli.route_behind_m),
        ahead_m=float(args_cli.route_ahead_m),
        width_m=0.08,
        thickness_m=0.025,
        vertical_offset_m=-0.18,
    )
    path_markers = None
    if playback_case.stage_id is None:
        route_cfg = sim_utils.CuboidCfg(
            size=geometry.size_xyz_m,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.05, 0.85, 0.08),
                emissive_color=(0.0, 0.35, 0.0),
            ),
        )
        route_cfg.func(
            "/World/Visuals/PureRLPlayback/TargetRoute",
            route_cfg,
            translation=geometry.midpoint_w,
            orientation=geometry.orientation_wxyz,
        )
    else:
        path = env.unwrapped._pure_rl_longitudinal_path
        if path is None:
            raise RuntimeError("C2 playback did not allocate longitudinal path state.")
        progress_m = torch.linspace(
            -float(args_cli.route_behind_m),
            float(args_cli.route_ahead_m),
            96,
            device=env.unwrapped.device,
        )
        env_origin_tensor_w = env.unwrapped.scene.env_origins[0]
        tangent_xy = env.unwrapped._straight_line_tangent_w[0, :2]
        xy_w = env_origin_tensor_w[:2] + progress_m.unsqueeze(1) * tangent_xy.unsqueeze(0)
        slope_progress_m = torch.clamp(
            progress_m - path.entry_length_m[0],
            min=0.0,
            max=float(path.slope_length_m[0].item()),
        )
        altitude_m = path.initial_altitude_m[0] + torch.tan(path.signed_slope_rad[0]) * slope_progress_m
        target_points_w = torch.cat((xy_w, altitude_m.unsqueeze(1)), dim=1)
        path_marker_cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/PureRLPlayback/TargetPath",
            markers={
                "target": sim_utils.SphereCfg(
                    radius=0.065,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.05, 0.95, 0.08),
                        emissive_color=(0.0, 0.4, 0.0),
                    ),
                )
            },
        )
        path_markers = VisualizationMarkers(path_marker_cfg)
        path_markers.visualize(translations=target_points_w)

    marker_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/PureRLPlayback/Markers",
        markers={
            "start": sim_utils.SphereCfg(
                radius=0.16,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.1, 1.0, 0.1),
                    emissive_color=(0.0, 0.3, 0.0),
                ),
            ),
            "closest_target": sim_utils.SphereCfg(
                radius=0.13,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.75, 0.05),
                    emissive_color=(0.35, 0.18, 0.0),
                ),
            ),
            "trajectory": sim_utils.SphereCfg(
                radius=0.055,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.05, 0.3, 1.0),
                    emissive_color=(0.0, 0.05, 0.3),
                ),
            ),
        },
    )
    markers = VisualizationMarkers(marker_cfg)
    route_origin_w = env.unwrapped.scene.env_origins[0].clone()
    route_origin_w[2] += env.unwrapped._height_cmd[0]
    return markers, route_origin_w, path_markers


def _update_markers(env, markers, route_origin_w, trail_w: list):
    import torch

    root_position_w = env.unwrapped._robot.data.root_pos_w[0].detach().clone()
    tangent_w = env.unwrapped._straight_line_tangent_w[0]
    if playback_case.stage_id is None:
        progress_m = torch.sum((root_position_w - route_origin_w) * tangent_w)
        closest_target_w = route_origin_w + progress_m * tangent_w
    else:
        query = env.unwrapped._query_pure_rl_longitudinal_path()
        closest_target_w = env.unwrapped.scene.env_origins[0] + query.horizontal_progress_m[0] * tangent_w
        closest_target_w[2] = env.unwrapped.scene.env_origins[0, 2] + query.reference_altitude_m[0]
    translations = torch.stack([route_origin_w, closest_target_w, *trail_w], dim=0)
    marker_indices = torch.tensor(
        [0, 1] + [2] * len(trail_w),
        dtype=torch.int32,
        device=translations.device,
    )
    markers.visualize(translations=translations, marker_indices=marker_indices)


def main() -> None:
    import gymnasium as gym
    import imageio.v2 as imageio
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

    env_cfg = parse_env_cfg(playback_case.task, device="cpu", num_envs=1)
    heading_rad = math.radians(float(args_cli.heading_deg))
    flap_phase_rad = math.radians(float(args_cli.flap_phase_deg))
    env_cfg.seed = int(args_cli.seed)
    env_cfg.episode_length_s = float(args_cli.duration_s)
    env_cfg.randomize_commands = False
    env_cfg.randomize_straight_line_heading = False
    env_cfg.randomize_flap_phase_at_reset = False
    env_cfg.pure_rl_eval_heading_schedule_rad = (heading_rad,)
    env_cfg.pure_rl_eval_flap_phase_schedule_rad = (flap_phase_rad,)
    if playback_case.stage_id == "c2b":
        assert playback_case.longitudinal_task_id is not None
        assert playback_case.longitudinal_slope_deg is not None
        assert playback_case.longitudinal_entry_length_m is not None
        assert playback_case.longitudinal_slope_length_m is not None
        env_cfg.pure_rl_eval_longitudinal_task_schedule = (playback_case.longitudinal_task_id,)
        env_cfg.pure_rl_eval_longitudinal_slope_deg_schedule = (playback_case.longitudinal_slope_deg,)
        env_cfg.pure_rl_eval_entry_length_m_schedule = (playback_case.longitudinal_entry_length_m,)
        env_cfg.pure_rl_eval_slope_length_m_schedule = (playback_case.longitudinal_slope_length_m,)
    env_cfg.wind_enabled = False
    env_cfg.wind_ou_enabled = False
    env_cfg.viewer.resolution = (int(args_cli.width), int(args_cli.height))
    env_cfg.viewer.eye = (-6.0, -10.0, 4.0)
    env_cfg.viewer.lookat = (2.0, 0.0, 0.0)
    robot_asset_path = (
        _REPO_ROOT
        / "source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf"
    ).resolve()
    robot_usd_dir = (args_cli.output_dir.expanduser().resolve() / "generated_assets/flap_robot_552").resolve()
    robot_usd_dir.mkdir(parents=True, exist_ok=True)
    env_cfg.robot.spawn.asset_path = str(robot_asset_path)
    env_cfg.robot.spawn.usd_dir = str(robot_usd_dir)

    agent_cfg = load_cfg_from_registry(playback_case.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = "cpu"
    agent_cfg_dict = agent_cfg.to_dict()
    agent_cfg_dict["device"] = "cpu"

    args_cli.output_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)
    base_env = gym.make(playback_case.task, cfg=env_cfg, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg_dict.get("clip_actions"))
    runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device="cpu")
    runner.load(str(selection.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    policy_hz = 1.0 / float(env.unwrapped.step_dt)
    capture_stride = int(round(policy_hz / float(args_cli.record_fps)))
    if capture_stride <= 0 or not math.isclose(
        policy_hz / capture_stride,
        float(args_cli.record_fps),
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        raise ValueError(
            f"--record-fps must divide the {policy_hz:.6f} Hz policy rate exactly; got {args_cli.record_fps}."
        )

    markers, route_origin_w, _path_markers = _spawn_route_visuals(env, heading_rad)
    trail_w = [env.unwrapped._robot.data.root_pos_w[0].detach().clone()]
    _update_markers(env, markers, route_origin_w, trail_w)
    obs = env.get_observations()
    policy_nn.reset(torch.ones(1, dtype=torch.long, device=env.unwrapped.device))
    metric_names = (
        "cross_track_error_m",
        "height_error_m",
        "along_track_progress_m",
        "along_track_velocity_mps",
        "tilt_rad",
        "angular_rate_rad_s",
        "actual_flap_frequency_hz",
        "frequency_limit_active",
        "tail_limit_active",
        "normalized_action_delta",
    )
    if playback_case.stage_id == "c2b":
        metric_names = (*metric_names, "reached_recovery")
    episode_metrics = {name: [] for name in metric_names}
    max_steps = int(round(float(args_cli.duration_s) * policy_hz)) + 2
    writer = imageio.get_writer(
        str(mp4_path),
        fps=int(args_cli.record_fps),
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    )
    terminated = False
    time_out = False
    termination_causes = {"ground": False, "tilt": False, "cross_track": False, "height": False}
    steps = 0
    rendered_frame_count = 0
    max_rendered_frame_std = 0.0

    def render_frame():
        """Render one camera frame without advancing physics."""

        root_position_w = base_env.unwrapped._robot.data.root_pos_w[0].detach().cpu()
        eye = root_position_w + root_position_w.new_tensor((-7.0, -11.0, 4.5))
        target = root_position_w + root_position_w.new_tensor((3.0, 0.0, 0.0))
        base_env.unwrapped.sim.set_camera_view(eye.tolist(), target.tolist())
        return base_env.unwrapped.render()

    def append_rendered_frame() -> None:
        nonlocal rendered_frame_count, max_rendered_frame_std
        frame = render_frame()
        frame_std = float(frame.std())
        rendered_frame_count += 1
        max_rendered_frame_std = max(max_rendered_frame_std, frame_std)
        writer.append_data(frame)

    try:
        for _ in range(30):
            initial_frame = render_frame()
            initial_frame_std = float(initial_frame.std())
            if initial_frame_std >= 1.0:
                writer.append_data(initial_frame)
                rendered_frame_count = 1
                max_rendered_frame_std = initial_frame_std
                break
        else:
            raise RuntimeError("Playback renderer did not produce a non-blank initial frame after warmup.")
        for step in range(max_steps):
            with torch.inference_mode():
                actions = policy(obs)
            obs, _reward, dones, _extras = env.step(actions)
            step_metrics = read_pure_rl_step_metrics(env.unwrapped)
            for name in metric_names:
                episode_metrics[name].append(float(step_metrics[name][0].item()))
            if step % int(args_cli.trail_stride) == 0:
                trail_w.append(env.unwrapped._robot.data.root_pos_w[0].detach().clone())
            _update_markers(env, markers, route_origin_w, trail_w)
            if (step + 1) % capture_stride == 0:
                append_rendered_frame()
            steps = step + 1
            if bool(dones[0].item()):
                terminated = bool(env.unwrapped.reset_terminated[0].item())
                time_out = bool(env.unwrapped.reset_time_outs[0].item())
                termination_causes = {
                    "ground": bool(step_metrics["ground_termination"][0].item()),
                    "tilt": bool(step_metrics["tilt_termination"][0].item()),
                    "cross_track": bool(step_metrics["cross_track_termination"][0].item()),
                    "height": bool(step_metrics["height_termination"][0].item()),
                }
                policy_nn.reset(dones)
                break
            policy_nn.reset(dones)
    finally:
        writer.close()
        env.close()

    if not episode_metrics["tilt_rad"]:
        raise RuntimeError("Playback produced no physical policy steps.")
    if rendered_frame_count < 2 or max_rendered_frame_std < 1.0:
        raise RuntimeError(
            "Playback rendering failed the non-blank-frame gate: "
            f"frame_count={rendered_frame_count}, max_frame_std={max_rendered_frame_std:.6f}."
        )
    episode_row = summarize_pure_rl_episode(
        step_metrics=episode_metrics,
        step_dt_s=float(env_cfg.decimation * env_cfg.sim.dt),
        terminated=terminated,
        time_out=time_out,
        termination_causes=termination_causes,
    )
    case = {
        "name": "playback_fixed_heading_phase",
        "wind_enabled": False,
        "wind_xy_mps": (0.0, 0.0),
        "wind_ou_enabled": False,
        "straight_line_heading_schedule_rad": (heading_rad,),
    }
    result_row = aggregate_pure_rl_case_row(
        checkpoint=selection.checkpoint,
        ckpt_index=int(selection.selection_row.get("ckpt_index", -1)),
        case=case,
        episode_rows=[episode_row],
    )
    recovery_reached = bool(
        playback_case.stage_id == "c2b"
        and any(bool(value) for value in episode_metrics["reached_recovery"])
    )
    playback_success = playback_case_succeeded(
        stage_id=playback_case.stage_id,
        terminated=terminated,
        c1_success_gate_passed=int(result_row["success_gate_passed"]) == 1,
        recovery_reached=recovery_reached,
    )
    result_row.update(
        {
            "stage_id": playback_case.stage_id,
            "longitudinal_task": playback_case.longitudinal_task,
            "longitudinal_slope_deg": playback_case.longitudinal_slope_deg,
            "recovery_reached": recovery_reached,
            "playback_success": playback_success,
        }
    )
    required_metrics = (
        "score",
        "timeout_rate",
        "termination_rate",
        "mean_abs_cross_track_error_m",
        "mean_abs_height_error_m",
        "mean_along_track_progress_m",
    )
    if not numeric_metrics_are_finite(result_row, required_metrics):
        raise RuntimeError(f"Playback result contains non-finite metrics: {result_row}")

    subprocess.run(
        build_gif_command(
            ffmpeg_executable=ffmpeg_executable,
            input_mp4=mp4_path,
            output_gif=gif_path,
            fps=int(args_cli.gif_fps),
            width_px=int(args_cli.gif_width),
        ),
        check=True,
    )
    manifest = {
        "schema_version": PLAYBACK_SCHEMA_VERSION,
        "task": playback_case.task,
        "command": [sys.executable, *sys.argv],
        "git": _git_provenance(),
        "run_dir": str(selection.run_dir),
        "checkpoint": str(selection.checkpoint),
        "checkpoint_sha256": sha256_file(selection.checkpoint),
        "selection_row": selection.selection_row,
        "condition": {
            "seed": int(args_cli.seed),
            "heading_deg": float(args_cli.heading_deg),
            "flap_phase_deg": float(args_cli.flap_phase_deg),
            "wind_xy_mps": [0.0, 0.0],
            "stage_id": playback_case.stage_id,
            "longitudinal_task": playback_case.longitudinal_task,
            "longitudinal_slope_deg": playback_case.longitudinal_slope_deg,
            "longitudinal_entry_length_m": playback_case.longitudinal_entry_length_m,
            "longitudinal_slope_length_m": playback_case.longitudinal_slope_length_m,
        },
        "recording": {
            "steps": steps,
            "policy_hz": policy_hz,
            "mp4_fps": int(args_cli.record_fps),
            "gif_fps": int(args_cli.gif_fps),
            "resolution": [int(args_cli.width), int(args_cli.height)],
            "frame_count": rendered_frame_count,
            "max_frame_std": max_rendered_frame_std,
            "mp4": str(mp4_path),
            "gif": str(gif_path),
        },
        "result": result_row,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        "[OK] PureRL playback:",
        json.dumps(
            {
                "mp4": str(mp4_path),
                "gif": str(gif_path),
                "manifest": str(manifest_path),
                "playback_success": playback_success,
                "score": result_row["score"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not playback_success:
        raise RuntimeError(f"Playback did not pass its stage success requirement: {result_row}")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # SimulationApp.close() may terminate Kit before Python emits a pending
        # traceback, so flush diagnostic evidence before shutting Kit down.
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        simulation_app.close()
