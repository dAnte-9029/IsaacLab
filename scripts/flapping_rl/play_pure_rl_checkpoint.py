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
    PLAYBACK_SCHEMA_VERSION,
    build_gif_command,
    compute_route_visual_geometry,
    numeric_metrics_are_finite,
    resolve_successful_checkpoint,
    sha256_file,
    validate_new_playback_outputs,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a successful PureRL checkpoint with route visualization and MP4/GIF recording."
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Training run containing eval/best_checkpoint.json.")
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
selection = resolve_successful_checkpoint(args_cli.run_dir)
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
    return markers, route_origin_w


def _update_markers(env, markers, route_origin_w, trail_w: list):
    import torch

    root_position_w = env.unwrapped._robot.data.root_pos_w[0].detach().clone()
    tangent_w = env.unwrapped._straight_line_tangent_w[0]
    progress_m = torch.sum((root_position_w - route_origin_w) * tangent_w)
    closest_target_w = route_origin_w + progress_m * tangent_w
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

    env_cfg = parse_env_cfg(MEASURED_PURE_RL_TASK_ID, device="cpu", num_envs=1)
    heading_rad = math.radians(float(args_cli.heading_deg))
    flap_phase_rad = math.radians(float(args_cli.flap_phase_deg))
    env_cfg.seed = int(args_cli.seed)
    env_cfg.episode_length_s = float(args_cli.duration_s)
    env_cfg.randomize_commands = False
    env_cfg.randomize_straight_line_heading = False
    env_cfg.randomize_flap_phase_at_reset = False
    env_cfg.pure_rl_eval_heading_schedule_rad = (heading_rad,)
    env_cfg.pure_rl_eval_flap_phase_schedule_rad = (flap_phase_rad,)
    env_cfg.wind_enabled = False
    env_cfg.wind_ou_enabled = False
    env_cfg.viewer.resolution = (int(args_cli.width), int(args_cli.height))
    env_cfg.viewer.eye = (-6.0, -10.0, 4.0)
    env_cfg.viewer.lookat = (2.0, 0.0, 0.0)

    agent_cfg = load_cfg_from_registry(MEASURED_PURE_RL_TASK_ID, "rsl_rl_cfg_entry_point")
    agent_cfg.device = "cpu"
    agent_cfg_dict = agent_cfg.to_dict()
    agent_cfg_dict["device"] = "cpu"

    args_cli.output_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)
    base_env = gym.make(MEASURED_PURE_RL_TASK_ID, cfg=env_cfg, render_mode="rgb_array")
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

    markers, route_origin_w = _spawn_route_visuals(env, heading_rad)
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

    def append_rendered_frame() -> None:
        nonlocal rendered_frame_count, max_rendered_frame_std
        frame = base_env.unwrapped.render()
        frame_std = float(frame.std())
        rendered_frame_count += 1
        max_rendered_frame_std = max(max_rendered_frame_std, frame_std)
        writer.append_data(frame)

    try:
        append_rendered_frame()
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
        "task": MEASURED_PURE_RL_TASK_ID,
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
                "success_gate_passed": result_row["success_gate_passed"],
                "score": result_row["score"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if int(result_row["success_gate_passed"]) != 1:
        raise RuntimeError(f"Playback did not pass the curriculum-1 success gate: {result_row}")


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
