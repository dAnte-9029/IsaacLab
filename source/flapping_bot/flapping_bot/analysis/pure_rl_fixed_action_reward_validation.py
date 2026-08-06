"""Fixed-action runtime gate for the PureRL curriculum-1 reward contract."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch

from flapping_bot.direct.flapping_bot.action_contract import frequency_hz_to_normalized_action
from flapping_bot.direct.flapping_bot.pure_rl_reward import compute_pure_rl_reward_terms


Tensor = torch.Tensor

ACTION_FAMILY_NAMES: tuple[str, ...] = (
    "hold_reset_action",
    "frequency_0_hz",
    "frequency_2_5_hz",
    "frequency_5_hz",
    "rudder_plus_0_5",
    "left_elevon_plus_0_5",
    "right_elevon_plus_0_5",
    "all_tail_soft_limit_0_9",
)

REWARD_TERM_NAMES: tuple[str, ...] = (
    "path",
    "progress",
    "velocity",
    "roll",
    "angular_rate",
    "pitch_envelope_penalty",
    "flap_penalty",
    "frequency_action_delta_penalty",
    "tail_action_delta_penalty",
    "tail_action_limit_penalty",
    "total",
)

TELEMETRY_KEYS: tuple[str, ...] = (
    "PureRLReward/total",
    "PureRLReward/path",
    "PureRLReward/progress",
    "PureRLReward/velocity",
    "PureRLReward/roll",
    "PureRLReward/angular_rate",
    "PureRLPenalty/pitch_envelope",
    "PureRLPenalty/flap",
    "PureRLPenalty/frequency_action_delta",
    "PureRLPenalty/tail_action_delta",
    "PureRLPenalty/tail_action_limit",
    "PureRLTermination/ground_fraction",
    "PureRLTermination/tilt_fraction",
    "PureRLTermination/cross_track_fraction",
    "PureRLTermination/height_error_fraction",
    "PureRLTermination/terminated_fraction",
    "PureRLTermination/time_out_fraction",
)


@dataclass(frozen=True)
class PureRLFixedActionRewardResult:
    """JSON summary and aligned per-sample/per-step numeric traces."""

    summary: dict[str, object]
    traces: dict[str, np.ndarray]


def build_fixed_action_matrix(
    base_action: Tensor,
    *,
    minimum_frequency_hz: float,
    maximum_frequency_hz: float,
) -> tuple[Tensor, Tensor]:
    """Build eight repeated constant-action families from the reset action."""

    if base_action.ndim != 2 or base_action.shape[1] != 4:
        raise ValueError("base_action must have shape (N, 4).")
    if base_action.shape[0] < len(ACTION_FAMILY_NAMES):
        raise ValueError("base_action must contain at least eight environments.")
    if not bool(torch.all(torch.isfinite(base_action))):
        raise ValueError("base_action must be finite.")
    if not float(maximum_frequency_hz) > float(minimum_frequency_hz):
        raise ValueError("maximum_frequency_hz must exceed minimum_frequency_hz.")

    environment_ids = torch.arange(base_action.shape[0], device=base_action.device)
    family = torch.remainder(environment_ids, len(ACTION_FAMILY_NAMES))
    action = base_action.clone()
    for family_index, frequency_hz in ((1, 0.0), (2, 2.5), (3, 5.0)):
        mask = family == family_index
        target = torch.full(
            (int(mask.sum().item()),),
            frequency_hz,
            device=base_action.device,
            dtype=base_action.dtype,
        )
        action[mask, 0] = frequency_hz_to_normalized_action(
            target,
            minimum_frequency_hz=minimum_frequency_hz,
            maximum_frequency_hz=maximum_frequency_hz,
        )
    action[family == 4, 1] = 0.5
    action[family == 5, 2] = 0.5
    action[family == 6, 3] = 0.5
    action[family == 7, 1:4] = 0.9
    return torch.clamp(action, -1.0, 1.0), family


def summarize_fixed_action_trace(
    traces: dict[str, np.ndarray],
    *,
    expected_action_by_environment: np.ndarray,
    total_policy_steps: int,
    reconstruction_tolerance: float = 5.0e-6,
    action_tolerance: float = 1.0e-7,
    settled_frequency_tolerance_hz: float = 0.15,
) -> dict[str, object]:
    """Apply deterministic acceptance gates to one fixed-action trace."""

    required = {
        "sample_step",
        "environment_id",
        "family_index",
        "applied_action",
        "actual_frequency_hz",
        "returned_reward",
        *REWARD_TERM_NAMES,
        "step_returned_reward_mean",
        "step_telemetry_reward_total_mean",
        "step_terminated_fraction",
        "step_telemetry_terminated_fraction",
    }
    missing = sorted(required.difference(traces))
    if missing:
        raise ValueError(f"Missing trace fields: {missing}")
    if total_policy_steps < 2:
        raise ValueError("total_policy_steps must be at least two.")

    numeric_arrays = [np.asarray(value) for value in traces.values() if np.asarray(value).dtype.kind in "fiu"]
    all_finite = bool(all(np.all(np.isfinite(value)) for value in numeric_arrays))
    sample_step = np.asarray(traces["sample_step"], dtype=np.int64)
    environment_id = np.asarray(traces["environment_id"], dtype=np.int64)
    family = np.asarray(traces["family_index"], dtype=np.int64)
    if not (sample_step.shape == environment_id.shape == family.shape):
        raise ValueError("Sample indices must be aligned 1D arrays.")
    action = np.asarray(traces["applied_action"], dtype=np.float64)
    expected_action = np.asarray(expected_action_by_environment, dtype=np.float64)
    if action.shape != (sample_step.size, 4) or expected_action.ndim != 2 or expected_action.shape[1] != 4:
        raise ValueError("Action traces must have shape (samples, 4).")
    if environment_id.size and (environment_id.min() < 0 or environment_id.max() >= expected_action.shape[0]):
        raise ValueError("environment_id falls outside expected_action_by_environment.")

    action_error = np.abs(action - expected_action[environment_id])
    action_max_error = float(np.max(action_error)) if action_error.size else math.inf
    returned_reward = np.asarray(traces["returned_reward"], dtype=np.float64)
    rebuilt_reward = np.asarray(traces["total"], dtype=np.float64)
    reconstruction_error = np.abs(returned_reward - rebuilt_reward)
    reconstruction_max_error = float(np.max(reconstruction_error)) if reconstruction_error.size else math.inf

    step_reward_error = np.abs(
        np.asarray(traces["step_returned_reward_mean"], dtype=np.float64)
        - np.asarray(traces["step_telemetry_reward_total_mean"], dtype=np.float64)
    )
    step_termination_error = np.abs(
        np.asarray(traces["step_terminated_fraction"], dtype=np.float64)
        - np.asarray(traces["step_telemetry_terminated_fraction"], dtype=np.float64)
    )
    telemetry_reward_max_error = float(np.max(step_reward_error)) if step_reward_error.size else math.inf
    telemetry_termination_max_error = (
        float(np.max(step_termination_error)) if step_termination_error.size else math.inf
    )

    frequency_delta = np.asarray(traces["frequency_action_delta_penalty"], dtype=np.float64)
    tail_delta = np.asarray(traces["tail_action_delta_penalty"], dtype=np.float64)
    first_mask = sample_step == 0
    later_mask = sample_step > 0
    first_delta_max = float(np.max(frequency_delta[first_mask] + tail_delta[first_mask])) if np.any(first_mask) else 0.0
    later_delta_max = (
        float(np.max(frequency_delta[later_mask] + tail_delta[later_mask]))
        if np.any(later_mask)
        else math.inf
    )

    settled_start_step = max(1, int(total_policy_steps) - max(3, int(math.ceil(0.2 * total_policy_steps))))
    settled = sample_step >= settled_start_step
    family_summaries: list[dict[str, object]] = []
    family_counts: list[int] = []
    for family_index, family_name in enumerate(ACTION_FAMILY_NAMES):
        family_mask = family == family_index
        settled_mask = family_mask & settled
        family_counts.append(int(np.count_nonzero(family_mask)))
        family_summaries.append(
            {
                "index": family_index,
                "name": family_name,
                "sample_count": int(np.count_nonzero(family_mask)),
                "settled_sample_count": int(np.count_nonzero(settled_mask)),
                "settled_actual_frequency_hz_mean": (
                    float(np.mean(np.asarray(traces["actual_frequency_hz"])[settled_mask]))
                    if np.any(settled_mask)
                    else None
                ),
                "settled_flap_penalty_mean": (
                    float(np.mean(np.asarray(traces["flap_penalty"])[settled_mask]))
                    if np.any(settled_mask)
                    else None
                ),
                "tail_action_limit_penalty_mean": (
                    float(np.mean(np.asarray(traces["tail_action_limit_penalty"])[family_mask]))
                    if np.any(family_mask)
                    else None
                ),
            }
        )

    tracked_frequencies = np.array(
        [family_summaries[index]["settled_actual_frequency_hz_mean"] for index in (1, 2, 3)],
        dtype=np.float64,
    )
    tracked_penalties = np.array(
        [family_summaries[index]["settled_flap_penalty_mean"] for index in (1, 2, 3)],
        dtype=np.float64,
    )
    target_frequencies = np.array([0.0, 2.5, 5.0], dtype=np.float64)
    expected_penalties = np.power(target_frequencies / 5.0, 3.0)
    frequency_tracking_max_error = (
        float(np.max(np.abs(tracked_frequencies - target_frequencies)))
        if np.all(np.isfinite(tracked_frequencies))
        else math.inf
    )
    flap_penalty_max_error = (
        float(np.max(np.abs(tracked_penalties - expected_penalties)))
        if np.all(np.isfinite(tracked_penalties))
        else math.inf
    )
    ordinary_tail_limit = np.asarray(
        [family_summaries[index]["tail_action_limit_penalty_mean"] for index in range(7)],
        dtype=np.float64,
    )
    soft_limit_penalty = float(family_summaries[7]["tail_action_limit_penalty_mean"] or 0.0)

    gates = {
        "all_numeric_traces_finite": all_finite,
        "every_action_family_sampled": bool(all(count > 0 for count in family_counts)),
        "fixed_actions_preserved": action_max_error <= action_tolerance,
        "reward_reconstruction_matches_environment": reconstruction_max_error <= reconstruction_tolerance,
        "reward_total_telemetry_matches_environment_mean": telemetry_reward_max_error <= reconstruction_tolerance,
        "termination_telemetry_matches_returned_done": telemetry_termination_max_error <= reconstruction_tolerance,
        "first_transition_has_delta_penalty": first_delta_max > 1.0e-6,
        "constant_actions_have_zero_later_delta_penalty": later_delta_max <= action_tolerance,
        "settled_frequency_tracks_0_2_5_5_hz": frequency_tracking_max_error <= settled_frequency_tolerance_hz,
        "settled_cubic_flap_penalty_matches_contract": flap_penalty_max_error <= 0.05,
        "settled_flap_penalty_is_strictly_ordered": bool(np.all(np.diff(tracked_penalties) > 0.0)),
        "ordinary_actions_avoid_tail_soft_limit": bool(np.all(ordinary_tail_limit <= action_tolerance)),
        "soft_limit_family_triggers_penalty": soft_limit_penalty >= 0.20,
        "no_target_speed_reward_term": not any(
            "target_speed" in name or "speed_target" in name for name in REWARD_TERM_NAMES
        ),
    }
    return {
        "all_cases_accepted": bool(all(gates.values())),
        "gates": gates,
        "metrics": {
            "action_max_abs_error": action_max_error,
            "reward_reconstruction_max_abs_error": reconstruction_max_error,
            "reward_total_telemetry_max_abs_error": telemetry_reward_max_error,
            "termination_telemetry_max_abs_error": telemetry_termination_max_error,
            "first_step_combined_delta_penalty_max": first_delta_max,
            "later_step_combined_delta_penalty_max": later_delta_max,
            "settled_frequency_tracking_max_abs_error_hz": frequency_tracking_max_error,
            "settled_flap_penalty_max_abs_error": flap_penalty_max_error,
            "soft_limit_family_penalty_mean": soft_limit_penalty,
            "settled_start_policy_step": settled_start_step,
        },
        "action_families": family_summaries,
    }


def _configure_robot_asset(cfg: object, *, asset_path: Path | None, usd_dir: Path | None) -> None:
    if asset_path is None and usd_dir is None:
        return
    replacements: dict[str, str] = {}
    if asset_path is not None:
        replacements["asset_path"] = str(Path(asset_path).resolve())
    if usd_dir is not None:
        replacements["usd_dir"] = str(Path(usd_dir).resolve())
    cfg.robot = cfg.robot.replace(spawn=cfg.robot.spawn.replace(**replacements))


def _make_cfg(
    *,
    num_envs: int,
    seed: int,
    asset_path: Path | None,
    usd_dir: Path | None,
):
    from flapping_bot.direct.flapping_bot.straight_flight_env import (
        FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg,
    )

    cfg = FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg()
    cfg.seed = int(seed)
    cfg.scene.num_envs = int(num_envs)
    cfg.scene.env_spacing = 5.0
    cfg.sim.device = "cpu"
    cfg.sim.dt = 1.0 / 480.0
    cfg.decimation = 8
    cfg.sim.render_interval = cfg.decimation
    cfg.freeze_steps_after_reset = 0
    cfg.episode_length_s = 20.0
    cfg.teacher_guidance_enabled = False
    cfg.randomize_commands = False
    cfg.wind_enabled = False
    cfg.randomize_wind = False
    cfg.wind_ou_enabled = False
    cfg.wind_curriculum_enabled = False
    cfg.act_lpf_tau_s = 0.0
    cfg.act_rate_limit_per_s = 0.0
    _configure_robot_asset(cfg, asset_path=asset_path, usd_dir=usd_dir)
    return cfg


def _to_numpy(value: Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


def _telemetry_scalar(log: dict[str, object], key: str) -> float:
    value = log.get(key)
    if value is None:
        return math.nan
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return math.nan
        return float(value.detach().cpu().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def run_fixed_action_reward_job(
    *,
    num_envs: int = 64,
    duration_s: float = 0.75,
    seed: int = 20260806,
    asset_path: Path | None = None,
    usd_dir: Path | None = None,
) -> PureRLFixedActionRewardResult:
    """Run a free-root fixed-action reward gate after Isaac Sim has started."""

    if num_envs < 8 or num_envs % len(ACTION_FAMILY_NAMES) != 0:
        raise ValueError("num_envs must be a positive multiple of eight.")
    if duration_s < 0.6 or duration_s > 2.0:
        raise ValueError("duration_s must lie in [0.6, 2.0].")
    from isaaclab.utils.math import euler_xyz_from_quat
    from flapping_bot.direct.flapping_bot.straight_flight_env import FlappingBotStraightFlightEnv

    cfg = _make_cfg(num_envs=num_envs, seed=seed, asset_path=asset_path, usd_dir=usd_dir)
    env = FlappingBotStraightFlightEnv(cfg)
    try:
        env.reset()
        base_action = env._act_cmd.detach().clone()
        fixed_action, family = build_fixed_action_matrix(
            base_action,
            minimum_frequency_hz=float(env.cfg.min_flap_hz),
            maximum_frequency_hz=float(env.cfg.max_flap_hz),
        )
        previous_action = base_action.clone()
        active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        total_policy_steps = int(math.ceil(float(duration_s) / float(env.step_dt)))
        termination_step = np.full((env.num_envs,), -1, dtype=np.int64)
        sample_trace: dict[str, list[np.ndarray]] = {
            "sample_step": [],
            "environment_id": [],
            "family_index": [],
            "applied_action": [],
            "actual_frequency_hz": [],
            "cross_track_error_m": [],
            "height_error_m": [],
            "along_track_velocity_mps": [],
            "cross_track_velocity_mps": [],
            "vertical_velocity_mps": [],
            "roll_rad": [],
            "pitch_rad": [],
            "angular_velocity_body_rad_s": [],
            "returned_reward": [],
        }
        sample_trace.update({name: [] for name in REWARD_TERM_NAMES})
        step_trace: dict[str, list[float | int]] = {
            "step_index": [],
            "step_active_environment_count": [],
            "step_terminated_fraction": [],
            "step_truncated_fraction": [],
            "step_returned_reward_mean": [],
        }
        for key in TELEMETRY_KEYS:
            step_trace[f"telemetry__{key.replace('/', '__')}"] = []

        for policy_step in range(total_policy_steps):
            _observation, reward, terminated, truncated, extras = env.step(fixed_action)
            done = terminated | truncated
            newly_done = active & done
            if bool(torch.any(newly_done)):
                done_ids = _to_numpy(torch.nonzero(newly_done, as_tuple=False).squeeze(1)).astype(int)
                termination_step[done_ids] = policy_step
            survivors = active & ~done
            active &= ~done

            log = extras.get("log", {}) if isinstance(extras, dict) else {}
            step_trace["step_index"].append(policy_step)
            step_trace["step_active_environment_count"].append(int(active.sum().item()))
            step_trace["step_terminated_fraction"].append(float(terminated.to(torch.float32).mean().item()))
            step_trace["step_truncated_fraction"].append(float(truncated.to(torch.float32).mean().item()))
            step_trace["step_returned_reward_mean"].append(float(reward.mean().item()))
            for key in TELEMETRY_KEYS:
                step_trace[f"telemetry__{key.replace('/', '__')}"].append(
                    _telemetry_scalar(log, key)
                )

            ids = torch.nonzero(survivors, as_tuple=False).squeeze(1)
            if ids.numel() > 0:
                local_position_w = env._robot.data.root_pos_w - env.scene.env_origins
                height_error_m = local_position_w[:, 2] - env._height_cmd
                cross_track_error_m = env._straight_line_cross_track_m(local_position_w)
                ground_velocity_w = env._robot.data.root_lin_vel_w
                along_track_velocity_mps = torch.sum(
                    ground_velocity_w * env._straight_line_tangent_w, dim=1
                )
                cross_track_velocity_mps = torch.sum(
                    ground_velocity_w * env._straight_line_normal_w, dim=1
                )
                vertical_velocity_mps = ground_velocity_w[:, 2]
                roll_rad, pitch_rad, _yaw_rad = euler_xyz_from_quat(env._robot.data.root_quat_w)
                terms = compute_pure_rl_reward_terms(
                    cross_track_error_m=cross_track_error_m,
                    height_error_m=height_error_m,
                    along_track_velocity_mps=along_track_velocity_mps,
                    cross_track_velocity_mps=cross_track_velocity_mps,
                    vertical_velocity_mps=vertical_velocity_mps,
                    roll_rad=roll_rad,
                    pitch_rad=pitch_rad,
                    angular_velocity_body_rad_s=env._robot.data.root_ang_vel_b,
                    actual_flap_frequency_hz=env._freq,
                    applied_action=env._act_cmd,
                    previous_applied_action=previous_action,
                    config=env.cfg.pure_rl_reward_cfg,
                )
                sample_trace["sample_step"].append(
                    np.full((ids.numel(),), policy_step, dtype=np.int64)
                )
                sample_trace["environment_id"].append(_to_numpy(ids).astype(np.int64))
                sample_trace["family_index"].append(_to_numpy(family[ids]).astype(np.int64))
                tensors = {
                    "applied_action": env._act_cmd,
                    "actual_frequency_hz": env._freq,
                    "cross_track_error_m": cross_track_error_m,
                    "height_error_m": height_error_m,
                    "along_track_velocity_mps": along_track_velocity_mps,
                    "cross_track_velocity_mps": cross_track_velocity_mps,
                    "vertical_velocity_mps": vertical_velocity_mps,
                    "roll_rad": roll_rad,
                    "pitch_rad": pitch_rad,
                    "angular_velocity_body_rad_s": env._robot.data.root_ang_vel_b,
                    "returned_reward": reward,
                }
                tensors.update(terms.as_dict())
                for name, value in tensors.items():
                    sample_trace[name].append(_to_numpy(value[ids]))
            previous_action.copy_(fixed_action)
            if not bool(torch.any(active)):
                break

        if not sample_trace["sample_step"]:
            raise RuntimeError("All environments terminated before one analyzable sample was collected.")
        traces = {name: np.concatenate(values, axis=0) for name, values in sample_trace.items()}
        traces.update({name: np.asarray(values) for name, values in step_trace.items()})
        traces["step_telemetry_reward_total_mean"] = traces[
            "telemetry__PureRLReward__total"
        ].copy()
        traces["step_telemetry_terminated_fraction"] = traces[
            "telemetry__PureRLTermination__terminated_fraction"
        ].copy()
        traces["expected_action_by_environment"] = _to_numpy(fixed_action)
        traces["base_action_by_environment"] = _to_numpy(base_action)
        traces["termination_step_by_environment"] = termination_step
        summary = summarize_fixed_action_trace(
            traces,
            expected_action_by_environment=traces["expected_action_by_environment"],
            total_policy_steps=total_policy_steps,
        )
        telemetry_arrays = [
            traces[f"telemetry__{key.replace('/', '__')}"] for key in TELEMETRY_KEYS
        ]
        termination_cause_event_counts = {
            cause: int(
                round(
                    float(
                        np.sum(
                            traces[f"telemetry__PureRLTermination__{cause}_fraction"]
                        )
                    )
                    * env.num_envs
                )
            )
            for cause in ("ground", "tilt", "cross_track", "height_error")
        }
        summary.update(
            {
                "schema_version": 1,
                "experiment": "pure_rl_curriculum1_fixed_action_reward_gate",
                "configuration": {
                    "num_envs": int(num_envs),
                    "seed": int(seed),
                    "duration_requested_s": float(duration_s),
                    "policy_dt_s": float(env.step_dt),
                    "physics_dt_s": float(env.physics_dt),
                    "total_policy_steps_requested": total_policy_steps,
                    "total_policy_steps_completed": int(traces["step_index"].size),
                    "terminal_reset_samples_included": False,
                    "randomized_route_heading": bool(env.cfg.randomize_straight_line_heading),
                    "randomized_flap_phase": bool(env.cfg.randomize_flap_phase_at_reset),
                },
                "telemetry": {
                    "required_keys": list(TELEMETRY_KEYS),
                    "all_required_keys_present_and_finite": bool(
                        all(np.all(np.isfinite(values)) for values in telemetry_arrays)
                    ),
                },
                "termination": {
                    "terminated_environment_count": int(np.count_nonzero(termination_step >= 0)),
                    "termination_step_by_environment": termination_step.tolist(),
                    "telemetry_cause_event_counts_including_post_reset_episodes": (
                        termination_cause_event_counts
                    ),
                },
                "claim_boundary": (
                    "This gate validates reward/termination wiring under fixed actions. It does not test "
                    "policy learning, closed-loop stability, convergence, or task success."
                ),
            }
        )
        summary["gates"]["all_required_telemetry_present_and_finite"] = summary["telemetry"][
            "all_required_keys_present_and_finite"
        ]
        summary["all_cases_accepted"] = bool(all(summary["gates"].values()))
        return PureRLFixedActionRewardResult(summary=summary, traces=traces)
    finally:
        env.close()


__all__ = [
    "ACTION_FAMILY_NAMES",
    "PureRLFixedActionRewardResult",
    "build_fixed_action_matrix",
    "run_fixed_action_reward_job",
    "summarize_fixed_action_trace",
]
