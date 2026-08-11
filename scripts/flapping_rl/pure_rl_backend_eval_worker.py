"""Fresh-process worker for one PureRL backend, suite, and rollout mode."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Mapping, Sequence

import numpy as np
from isaaclab.app import AppLauncher


_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from eval_suites import build_eval_cases
from pure_rl_backend_diagnostics import validate_trace_arrays
from pure_rl_eval_common import (
    PURE_RL_CURRICULUM1_EVAL_SUITE,
    aggregate_pure_rl_case_row,
    assert_pure_rl_longitudinal_reset_schedule,
    assert_pure_rl_reset_schedule,
    longitudinal_stage_for_task,
    pure_rl_backend_for_task,
    read_pure_rl_step_metrics,
    summarize_pure_rl_episode,
)
from pure_rl_longitudinal_eval import (
    build_longitudinal_evaluation_grid,
    summarize_longitudinal_evaluation,
)


@dataclass(frozen=True)
class RegisteredCase:
    """One fixed reset condition assigned to one worker environment."""

    case_id: str
    stage_id: str
    task: str
    task_id: int
    signed_slope_deg: float
    heading_rad: float
    flap_phase_rad: float
    entry_length_m: float
    slope_length_m: float
    episode_duration_s: float
    promotion_eligible: bool


def with_local_extension_root(kit_args: str, source_root: Path) -> str:
    """Add this checkout's source directory to Kit extension discovery once."""

    extension_args = f"--ext-folder {source_root.resolve()}"
    if extension_args in str(kit_args):
        return str(kit_args)
    return f"{str(kit_args).strip()} {extension_args}".strip()


def with_native_constraint_extension(kit_args: str, native_extensions_root: Path) -> str:
    """Enable the CPU-native holonomic constraint extension exactly once."""

    result = str(kit_args).strip()
    extension_args = f"--ext-folder {native_extensions_root.resolve()}"
    if extension_args not in result:
        result = f"{result} {extension_args}".strip()
    enable_arg = "--enable omni.flapping_bot.holonomic_constraint"
    if enable_arg not in result:
        result = f"{result} {enable_arg}".strip()
    return result


def policy_observation_tensor(observations):
    """Return the policy group tensor from RSL-RL TensorDict observations."""

    try:
        return observations["policy"]
    except (IndexError, KeyError, TypeError):
        return observations


def configure_generated_assets(env_cfg, output_dir: Path) -> None:
    """Keep URDF conversion products inside the evaluation output directory."""

    env_cfg.robot.spawn.usd_dir = str(output_dir / "generated_assets")


def configure_evaluation_seed(env_cfg, *, seed: int) -> None:
    """Set the explicit seed shared by matched backend evaluations."""

    if seed < 0:
        raise ValueError("evaluation seed must be nonnegative")
    env_cfg.seed = int(seed)


def resolve_registered_cases(
    task: str,
    *,
    case_ids: Sequence[str] | None = None,
) -> tuple[RegisteredCase, ...]:
    """Resolve the approved C1 or C2a fixed grid, optionally in requested order."""

    pure_rl_backend_for_task(task)
    stage_id = longitudinal_stage_for_task(task)
    if stage_id is None:
        suite_case = build_eval_cases(PURE_RL_CURRICULUM1_EVAL_SUITE)[0]
        headings = tuple(float(value) for value in suite_case["straight_line_heading_schedule_rad"])
        phases = tuple(float(value) for value in suite_case["flap_phase_schedule_rad"])
        registered = tuple(
            RegisteredCase(
                case_id=f"c1_h{index // 4}_p{index % 4}",
                stage_id="c1_straight",
                task="level",
                task_id=0,
                signed_slope_deg=0.0,
                heading_rad=heading,
                flap_phase_rad=phase,
                entry_length_m=0.0,
                slope_length_m=0.0,
                episode_duration_s=12.0,
                promotion_eligible=True,
            )
            for index, (heading, phase) in enumerate(zip(headings, phases, strict=True))
        )
    elif stage_id == "c2a":
        registered = tuple(
            RegisteredCase(
                case_id=case.case_id,
                stage_id=case.stage_id,
                task=case.task,
                task_id=case.task_id,
                signed_slope_deg=case.signed_slope_deg,
                heading_rad=case.heading_rad,
                flap_phase_rad=case.flap_phase_rad,
                entry_length_m=case.entry_length_m,
                slope_length_m=case.slope_length_m,
                episode_duration_s=case.episode_duration_s,
                promotion_eligible=case.promotion_eligible,
            )
            for case in build_longitudinal_evaluation_grid(stage_id)
        )
    else:
        raise ValueError(f"GPU qualification worker supports only C1 and C2a, got stage {stage_id!r}.")

    if case_ids is None:
        return registered
    by_id = {case.case_id: case for case in registered}
    missing = [case_id for case_id in case_ids if case_id not in by_id]
    if missing:
        raise ValueError(f"Unknown case IDs for {task}: {missing}")
    if len(set(case_ids)) != len(tuple(case_ids)):
        raise ValueError("case_ids must be unique")
    return tuple(by_id[case_id] for case_id in case_ids)


def validate_worker_request(
    *,
    task: str,
    backend_id: str,
    rollout_mode: str,
    action_dir: Path | None,
) -> None:
    """Fail closed on task/backend or closed-loop/replay mismatch."""

    expected_backend = pure_rl_backend_for_task(task)
    if backend_id != expected_backend:
        raise ValueError(
            f"Task/backend mismatch: task {task!r} requires backend {expected_backend!r}, got {backend_id!r}."
        )
    if rollout_mode not in {"closed_loop", "action_replay"}:
        raise ValueError("rollout_mode must be closed_loop or action_replay")
    if rollout_mode == "action_replay" and action_dir is None:
        raise ValueError("action_dir is required for action_replay")
    if rollout_mode == "closed_loop" and action_dir is not None:
        raise ValueError("action_dir is valid only for action_replay")


def load_action_sequence(path: Path) -> np.ndarray:
    """Load and validate one policy-rate four-channel action sequence."""

    with np.load(path, allow_pickle=False) as data:
        if "actions" not in data:
            raise ValueError(f"Action file has no 'actions' array: {path}")
        actions = np.asarray(data["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 4:
        raise ValueError(f"Action sequence must have exactly four channels: {path}")
    if actions.shape[0] == 0 or not np.all(np.isfinite(actions)):
        raise ValueError(f"Action sequence must be non-empty and finite: {path}")
    return actions


def stack_trace_samples(samples: Sequence[Mapping[str, object]]) -> dict[str, np.ndarray]:
    """Stack aligned scalar/vector samples into NPZ-ready arrays."""

    if not samples:
        raise ValueError("trace samples must not be empty")
    fields = tuple(samples[0].keys())
    expected = set(fields)
    if any(set(sample.keys()) != expected for sample in samples[1:]):
        raise ValueError("trace samples must contain the same fields")
    trace = {name: np.asarray([sample[name] for sample in samples]) for name in fields}
    validate_trace_arrays(trace)
    return trace


def physical_trace_sample_is_valid(*, done: bool) -> bool:
    """Return whether a post-step state still belongs to the active episode."""

    return not done


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--backend-id", required=True, choices=("cpu_native_authority", "gpu_implicit_candidate"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rollout-mode", choices=("closed_loop", "action_replay"), default="closed_loop")
    parser.add_argument("--action-dir", type=Path, default=None)
    parser.add_argument("--case-id", action="append", dest="case_ids", default=None)
    parser.add_argument("--capture-traces", action="store_true")
    parser.add_argument("--capture-actions", action="store_true")
    parser.add_argument("--maximum-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()
    return args


def _load_yaml_compat(path: Path) -> dict[str, object]:
    import yaml

    class _Loader(yaml.FullLoader):
        pass

    def _construct_posixpath(loader: yaml.FullLoader, node: yaml.Node) -> Path:
        return Path(*loader.construct_sequence(node))

    _Loader.add_constructor("tag:yaml.org,2002:python/object/apply:pathlib.PosixPath", _construct_posixpath)
    with path.open("r", encoding="utf-8") as stream:
        loaded = yaml.load(stream, Loader=_Loader)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping in {path}")
    return loaded


def _configure_reset_schedule(env_cfg, cases: Sequence[RegisteredCase]) -> None:
    env_cfg.randomize_commands = False
    env_cfg.randomize_straight_line_heading = False
    env_cfg.randomize_flap_phase_at_reset = False
    env_cfg.pure_rl_eval_heading_schedule_rad = tuple(case.heading_rad for case in cases)
    env_cfg.pure_rl_eval_flap_phase_schedule_rad = tuple(case.flap_phase_rad for case in cases)
    if cases[0].stage_id == "c2a":
        env_cfg.pure_rl_eval_longitudinal_task_schedule = tuple(case.task_id for case in cases)
        env_cfg.pure_rl_eval_longitudinal_slope_deg_schedule = tuple(
            case.signed_slope_deg for case in cases
        )
        env_cfg.pure_rl_eval_entry_length_m_schedule = tuple(case.entry_length_m for case in cases)
        env_cfg.pure_rl_eval_slope_length_m_schedule = tuple(case.slope_length_m for case in cases)


def _assert_reset_schedule(env, cases: Sequence[RegisteredCase]) -> None:
    unwrapped = env.unwrapped
    headings = tuple(case.heading_rad for case in cases)
    phases = tuple(case.flap_phase_rad for case in cases)
    if cases[0].stage_id == "c2a":
        path = unwrapped._pure_rl_longitudinal_path
        if path is None:
            raise RuntimeError("C2a worker did not allocate longitudinal path state")
        assert_pure_rl_longitudinal_reset_schedule(
            actual_heading_rad=unwrapped._straight_line_heading_rad,
            actual_flap_phase_rad=unwrapped._phase,
            actual_task_id=path.task_id,
            actual_signed_slope_rad=path.signed_slope_rad,
            expected_heading_schedule_rad=headings,
            expected_flap_phase_schedule_rad=phases,
            expected_task_schedule=tuple(case.task_id for case in cases),
            expected_signed_slope_deg_schedule=tuple(case.signed_slope_deg for case in cases),
        )
    else:
        assert_pure_rl_reset_schedule(
            actual_heading_rad=unwrapped._straight_line_heading_rad,
            actual_flap_phase_rad=unwrapped._phase,
            expected_heading_schedule_rad=headings,
            expected_flap_phase_schedule_rad=phases,
        )


def _checkpoint_iteration(path: Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else -1


def _tensor_value(tensor, env_id: int):
    return tensor[env_id].detach().cpu().numpy().copy()


def _trace_sample(env, *, env_id: int, time_s: float, actions, reward, metrics) -> dict[str, object]:
    unwrapped = env.unwrapped
    robot = unwrapped._robot
    local_position = robot.data.root_pos_w - unwrapped.scene.env_origins
    left_joint_id = int(unwrapped._joint_ids[unwrapped._IDX_LEFT_WING])
    right_joint_id = int(unwrapped._joint_ids[unwrapped._IDX_RIGHT_WING])
    left_position = robot.data.joint_pos[:, left_joint_id] - float(unwrapped._wing_mid_L)
    right_position = -(robot.data.joint_pos[:, right_joint_id] - float(unwrapped._wing_mid_R))
    left_velocity = robot.data.joint_vel[:, left_joint_id]
    right_velocity = -robot.data.joint_vel[:, right_joint_id]
    reward_terms = unwrapped._debug_last_pure_rl_reward_terms or {}

    sample: dict[str, object] = {
        "time_s": float(time_s),
        "actions": _tensor_value(actions, env_id),
        "done": False,
        "terminated": False,
        "time_out": False,
        "physical_state_valid": True,
        "executed_action": _tensor_value(unwrapped._debug_last_exec_action, env_id),
        "reward": float(reward[env_id].item()),
        "root_position_m": _tensor_value(local_position, env_id),
        "root_quaternion_wxyz": _tensor_value(robot.data.root_quat_w, env_id),
        "root_linear_velocity_world_mps": _tensor_value(robot.data.root_lin_vel_w, env_id),
        "root_angular_velocity_body_rad_s": _tensor_value(robot.data.root_ang_vel_b, env_id),
        "requested_frequency_hz": float(unwrapped._requested_frequency_hz[env_id].item()),
        "applied_frequency_hz": float(unwrapped._applied_frequency_hz[env_id].item()),
        "actual_tail_angle_rad": np.asarray(
            [
                unwrapped._debug_last_actual_rudder_rad[env_id].item(),
                unwrapped._debug_last_actual_left_elevon_rad[env_id].item(),
                unwrapped._debug_last_actual_right_elevon_rad[env_id].item(),
            ],
            dtype=np.float32,
        ),
        "wing_target_position_rad": float(unwrapped._q_cmd[env_id].item()),
        "wing_target_velocity_rad_s": float(unwrapped._qd_cmd[env_id].item()),
        "wing_actual_position_rad": np.asarray(
            [left_position[env_id].item(), right_position[env_id].item()], dtype=np.float32
        ),
        "wing_actual_velocity_rad_s": np.asarray(
            [left_velocity[env_id].item(), right_velocity[env_id].item()], dtype=np.float32
        ),
        "wing_tracking_error_rad": np.asarray(
            [
                left_position[env_id].item() - unwrapped._q_cmd[env_id].item(),
                right_position[env_id].item() - unwrapped._q_cmd[env_id].item(),
            ],
            dtype=np.float32,
        ),
        "wing_sync_error_rad": float((left_position[env_id] - right_position[env_id]).item()),
        "wing_force_link_n": _tensor_value(unwrapped._debug_last_wing_force_link_n, env_id),
        "wing_moment_link_nm": _tensor_value(
            unwrapped._debug_last_wing_moment_link_about_com_nm, env_id
        ),
        "wing_force_body_n": _tensor_value(unwrapped._debug_last_wing_force_b, env_id),
        "wing_moment_body_nm": _tensor_value(
            unwrapped._debug_last_wing_moment_b_about_base_com_nm, env_id
        ),
        "tail_force_body_n": _tensor_value(unwrapped._debug_last_tail_force_b, env_id),
        "tail_moment_body_nm": _tensor_value(
            unwrapped._debug_last_tail_moment_b_about_base_com_nm, env_id
        ),
        "total_aero_force_body_n": _tensor_value(unwrapped._debug_last_force_b, env_id),
        "total_aero_moment_body_nm": _tensor_value(unwrapped._debug_last_torque_b, env_id),
    }
    for name, tensor in metrics.items():
        sample[name] = _tensor_value(tensor, env_id)
    for name, tensor in reward_terms.items():
        sample[f"reward_term_{name}"] = _tensor_value(tensor, env_id)
    return sample


def _termination_cause(metrics: Mapping[str, object], env_id: int, *, timed_out: bool) -> str:
    for name in ("ground", "tilt", "cross_track", "height"):
        if bool(metrics[f"{name}_termination"][env_id].item()):
            return name
    return "timeout" if timed_out else "unknown"


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _generic_summary(rows: Sequence[Mapping[str, object]], *, backend_id: str, rollout_mode: str) -> dict[str, object]:
    return {
        "backend_id": backend_id,
        "rollout_mode": rollout_mode,
        "case_count": len(rows),
        "success_rate": sum(float(bool(row["success"])) for row in rows) / len(rows),
        "termination_rate": sum(float(bool(row["terminated"])) for row in rows) / len(rows),
        "finite_metrics": all(bool(row["finite_metrics"]) for row in rows),
        "hard_gate_passed": False,
    }


def _run_worker(args: argparse.Namespace) -> dict[str, object]:
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    action_dir = args.action_dir.expanduser().resolve() if args.action_dir is not None else None
    validate_worker_request(
        task=args.task,
        backend_id=args.backend_id,
        rollout_mode=args.rollout_mode,
        action_dir=action_dir,
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    cases = resolve_registered_cases(args.task, case_ids=args.case_ids)
    if args.rollout_mode == "action_replay" and args.case_ids is None:
        raise ValueError("action_replay requires explicit --case-id selections")
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_device_prefix = "cpu" if args.backend_id == "cpu_native_authority" else "cuda"
    if not str(args.device).startswith(expected_device_prefix):
        raise ValueError(
            f"Backend {args.backend_id} requires a {expected_device_prefix} device, got {args.device!r}."
        )

    args.kit_args = with_local_extension_root(
        str(getattr(args, "kit_args", "")),
        _SCRIPT_DIR.parents[1] / "source",
    )
    if args.backend_id == "cpu_native_authority":
        args.kit_args = with_native_constraint_extension(
            args.kit_args,
            _SCRIPT_DIR.parents[1] / "source/flapping_bot/native_extensions",
        )
    print("[pure-rl-backend-worker] launching Isaac Sim", flush=True)
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    print("[pure-rl-backend-worker] Isaac Sim ready", flush=True)
    env = None
    start_time = time.perf_counter()
    total_simulated_env_steps = 0
    try:
        import gymnasium as gym
        import torch
        from rsl_rl.runners import OnPolicyRunner

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

        print("[pure-rl-backend-worker] runtime imports ready", flush=True)
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=len(cases))
        configure_generated_assets(env_cfg, output_dir)
        configure_evaluation_seed(env_cfg, seed=int(args.seed))
        print("[pure-rl-backend-worker] environment config parsed", flush=True)
        _configure_reset_schedule(env_cfg, cases)
        if str(env_cfg.sim.device) != str(args.device):
            raise RuntimeError("parse_env_cfg did not preserve the requested simulation device")
        if bool(env_cfg.scene.replicate_physics) != (args.backend_id == "gpu_implicit_candidate"):
            raise RuntimeError("Environment physics-replication setting does not match the backend contract")

        saved_agent = checkpoint.parent / "params" / "agent.yaml"
        if saved_agent.is_file():
            agent_cfg_dict = _load_yaml_compat(saved_agent)
        else:
            agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
            agent_cfg_dict = agent_cfg.to_dict()
        agent_cfg_dict["device"] = str(args.device)

        env = gym.make(args.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg_dict.get("clip_actions", None))
        runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=str(args.device))
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic

        obs, _ = env.reset()
        policy_nn.reset(torch.ones(len(cases), dtype=torch.long, device=env.unwrapped.device))
        _assert_reset_schedule(env, cases)
        policy_observations = policy_observation_tensor(obs)
        if int(policy_observations.shape[-1]) != 555:
            raise RuntimeError(f"Expected 555 observations, got {tuple(policy_observations.shape)}")

        replay_actions: list[np.ndarray] | None = None
        if args.rollout_mode == "action_replay":
            assert action_dir is not None
            replay_actions = [load_action_sequence(action_dir / f"{case.case_id}.npz") for case in cases]

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
            "frequency_slew_hz_per_s",
            "frequency_governor_limited",
        )
        if cases[0].stage_id == "c2a":
            metric_names += (
                "lateral_normal_velocity_mps",
                "vertical_normal_velocity_mps",
                "active_slope_rad",
                "reached_recovery",
            )
        step_metrics = [{name: [] for name in metric_names} for _ in cases]
        traces: list[list[dict[str, object]]] = [[] for _ in cases]
        captured_actions: list[list[np.ndarray]] = [[] for _ in cases]
        completed = [False] * len(cases)
        final_rows: list[dict[str, object] | None] = [None] * len(cases)
        longitudinal_rows: list[dict[str, object] | None] = [None] * len(cases)
        step_index = 0
        maximum_steps = int(args.maximum_steps) if args.maximum_steps is not None else int(env.unwrapped.max_episode_length)

        while not all(completed):
            if step_index >= maximum_steps:
                raise RuntimeError(f"Worker exceeded maximum_steps={maximum_steps} before all cases completed")
            if replay_actions is None:
                with torch.inference_mode():
                    actions = policy(obs)
            else:
                policy_observations = policy_observation_tensor(obs)
                actions = torch.zeros(
                    (len(cases), 4),
                    device=env.unwrapped.device,
                    dtype=policy_observations.dtype,
                )
                for env_id, sequence in enumerate(replay_actions):
                    if not completed[env_id] and step_index < sequence.shape[0]:
                        actions[env_id] = torch.as_tensor(sequence[step_index], device=actions.device)
            if tuple(actions.shape) != (len(cases), 4):
                raise RuntimeError(f"Expected four policy actions, got {tuple(actions.shape)}")

            obs, rewards, dones, _info = env.step(actions)
            total_simulated_env_steps += len(cases)
            metrics = read_pure_rl_step_metrics(env.unwrapped)
            finite_root = torch.all(torch.isfinite(env.unwrapped._robot.data.root_state_w), dim=1)
            finite_actions = torch.all(torch.isfinite(actions), dim=1)

            for env_id, case in enumerate(cases):
                if completed[env_id]:
                    continue
                done = bool(dones[env_id].item())
                for name in metric_names:
                    step_metrics[env_id][name].append(float(metrics[name][env_id].item()))
                if args.capture_actions and replay_actions is None:
                    captured_actions[env_id].append(actions[env_id].detach().cpu().numpy().copy())
                if args.capture_traces and physical_trace_sample_is_valid(done=done):
                    traces[env_id].append(
                        _trace_sample(
                            env,
                            env_id=env_id,
                            time_s=step_index * float(env.unwrapped.step_dt),
                            actions=actions,
                            reward=rewards,
                            metrics=metrics,
                        )
                    )

                nonfinite = not bool(finite_root[env_id].item() and finite_actions[env_id].item())
                replay_exhausted = bool(
                    replay_actions is not None and step_index + 1 >= replay_actions[env_id].shape[0]
                )
                if not (done or nonfinite or replay_exhausted):
                    continue

                terminated = bool(env.unwrapped.reset_terminated[env_id].item()) if done else False
                timed_out = bool(env.unwrapped.reset_time_outs[env_id].item()) if done else replay_exhausted
                cause = "nonfinite" if nonfinite else _termination_cause(metrics, env_id, timed_out=timed_out)
                episode = summarize_pure_rl_episode(
                    step_metrics=step_metrics[env_id],
                    step_dt_s=float(env.unwrapped.step_dt),
                    terminated=terminated or nonfinite,
                    time_out=timed_out and not nonfinite,
                    termination_causes={
                        name: bool(metrics[f"{name}_termination"][env_id].item())
                        for name in ("ground", "tilt", "cross_track", "height")
                    },
                )
                recovery_reached = bool(
                    any(bool(value) for value in step_metrics[env_id].get("reached_recovery", []))
                )
                success = bool(not terminated and not nonfinite and (recovery_reached if case.stage_id == "c2a" else timed_out))
                path_error_m = float(episode["mean_abs_cross_track_error_m"]) + float(
                    episode["mean_abs_height_error_m"]
                )
                final_rows[env_id] = {
                    "case_id": case.case_id,
                    "backend_id": args.backend_id,
                    "rollout_mode": args.rollout_mode,
                    "stage_id": case.stage_id,
                    "task": case.task,
                    "signed_slope_deg": case.signed_slope_deg,
                    "heading_rad": case.heading_rad,
                    "flap_phase_rad": case.flap_phase_rad,
                    "success": success,
                    "terminated": bool(terminated or nonfinite),
                    "time_out": bool(timed_out and not nonfinite),
                    "termination_cause": cause,
                    "duration_s": float(episode["episode_duration_s"]),
                    "path_error_m": path_error_m,
                    "finite_metrics": not nonfinite,
                    "recovery_reached": recovery_reached,
                    **episode,
                }
                if case.stage_id == "c2a":
                    longitudinal_rows[env_id] = {
                        "case_id": case.case_id,
                        "stage_id": case.stage_id,
                        "task": case.task,
                        "promotion_eligible": case.promotion_eligible,
                        "terminated": bool(terminated or nonfinite),
                        "success": success,
                        "recovery_reached": recovery_reached,
                        "cross_track_error_m": step_metrics[env_id]["cross_track_error_m"],
                        "height_error_m": step_metrics[env_id]["height_error_m"],
                        "tangent_velocity_mps": step_metrics[env_id]["along_track_velocity_mps"],
                        "finite_metrics": not nonfinite,
                    }
                completed[env_id] = True
            policy_nn.reset(dones)
            step_index += 1

        per_case_rows = [row for row in final_rows if row is not None]
        if len(per_case_rows) != len(cases):
            raise RuntimeError("Worker did not produce exactly one row per registered case")

        if args.rollout_mode == "closed_loop" and args.case_ids is None and cases[0].stage_id == "c2a":
            raw_longitudinal_rows = [row for row in longitudinal_rows if row is not None]
            summary = summarize_longitudinal_evaluation(
                raw_longitudinal_rows,
                expected_cases=build_longitudinal_evaluation_grid("c2a"),
                checkpoint=str(checkpoint),
                ppo_iteration=_checkpoint_iteration(checkpoint),
            )
            summary["hard_gate_passed"] = bool(summary["promotion_gate_passed"])
        elif args.rollout_mode == "closed_loop" and args.case_ids is None and cases[0].stage_id == "c1_straight":
            suite_case = build_eval_cases(PURE_RL_CURRICULUM1_EVAL_SUITE)[0]
            episode_rows = [
                {key: value for key, value in row.items() if key in {
                    "episode_duration_s",
                    "along_track_progress_m",
                    "mean_along_track_velocity_mps",
                    "reverse_motion_fraction",
                    "mean_abs_cross_track_error_m",
                    "max_abs_cross_track_error_m",
                    "mean_abs_height_error_m",
                    "max_abs_height_error_m",
                    "mean_tilt_deg",
                    "max_tilt_deg",
                    "mean_angular_rate_rad_s",
                    "mean_actual_flap_frequency_hz",
                    "frequency_limit_fraction",
                    "tail_limit_fraction",
                    "mean_normalized_action_delta",
                    "mean_abs_frequency_slew_hz_per_s",
                    "frequency_governor_limited_fraction",
                    "terminated",
                    "time_out",
                    "ground_termination",
                    "tilt_termination",
                    "cross_track_termination",
                    "height_termination",
                }}
                for row in per_case_rows
            ]
            summary = aggregate_pure_rl_case_row(
                checkpoint=checkpoint,
                ckpt_index=_checkpoint_iteration(checkpoint),
                case=suite_case,
                episode_rows=episode_rows,
            )
            summary["hard_gate_passed"] = bool(summary["success_gate_passed"])
        else:
            summary = _generic_summary(per_case_rows, backend_id=args.backend_id, rollout_mode=args.rollout_mode)

        summary.update(
            {
                "backend_id": args.backend_id,
                "rollout_mode": args.rollout_mode,
                "task": args.task,
                "checkpoint": str(checkpoint),
                "selected_case_ids": [case.case_id for case in cases],
            }
        )
        _write_csv(output_dir / "per_case_metrics.csv", per_case_rows)
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        if args.capture_traces:
            trace_dir = output_dir / "traces"
            trace_dir.mkdir(parents=True, exist_ok=True)
            for case, samples in zip(cases, traces, strict=True):
                trace = stack_trace_samples(samples)
                np.savez_compressed(trace_dir / f"{case.case_id}.npz", **trace)
        if args.capture_actions and replay_actions is None:
            action_output_dir = output_dir / "actions"
            action_output_dir.mkdir(parents=True, exist_ok=True)
            for case, samples in zip(cases, captured_actions, strict=True):
                actions_array = np.asarray(samples, dtype=np.float32)
                if actions_array.ndim != 2 or actions_array.shape[1] != 4:
                    raise RuntimeError(f"Captured action shape is invalid for {case.case_id}: {actions_array.shape}")
                np.savez_compressed(action_output_dir / f"{case.case_id}.npz", actions=actions_array)

        elapsed_s = time.perf_counter() - start_time
        success_manifest = {
            "completed": True,
            "task": args.task,
            "backend_id": args.backend_id,
            "rollout_mode": args.rollout_mode,
            "checkpoint": str(checkpoint),
            "device": str(args.device),
            "seed": int(args.seed),
            "case_ids": [case.case_id for case in cases],
            "case_count": len(cases),
            "total_simulated_env_steps": total_simulated_env_steps,
            "wall_time_s": elapsed_s,
            "evaluation_env_steps_per_second": total_simulated_env_steps / max(elapsed_s, 1.0e-9),
            "summary_path": str(output_dir / "summary.json"),
            "per_case_metrics_path": str(output_dir / "per_case_metrics.csv"),
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(success_manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        return success_manifest
    except BaseException as error:
        failure_manifest = {
            "completed": False,
            "task": str(args.task),
            "backend_id": str(args.backend_id),
            "rollout_mode": str(args.rollout_mode),
            "seed": int(args.seed),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(failure_manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        traceback.print_exception(type(error), error, error.__traceback__, file=sys.stderr)
        sys.stderr.flush()
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        manifest = _run_worker(args)
    except BaseException as error:
        manifest = {
            "completed": False,
            "task": str(getattr(args, "task", "")),
            "backend_id": str(getattr(args, "backend_id", "")),
            "rollout_mode": str(getattr(args, "rollout_mode", "")),
            "seed": int(getattr(args, "seed", 0)),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        raise
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
