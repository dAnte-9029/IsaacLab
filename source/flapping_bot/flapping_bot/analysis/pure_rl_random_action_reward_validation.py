"""Bounded random-action runtime gate for the PureRL curriculum-1 task."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch

from flapping_bot.analysis.pure_rl_fixed_action_reward_validation import (
    REWARD_TERM_NAMES,
    TELEMETRY_KEYS,
    _make_cfg,
    _telemetry_scalar,
    _to_numpy,
)
from flapping_bot.direct.flapping_bot.pure_rl_reward import compute_pure_rl_reward_terms


Tensor = torch.Tensor

RANDOM_ACTION_FAMILY_NAMES: tuple[str, ...] = (
    "iid_full_range",
    "correlated_interior_tail",
)


@dataclass(frozen=True)
class PureRLRandomActionRewardResult:
    """JSON summary and aligned numeric traces from one random-action gate."""

    summary: dict[str, object]
    traces: dict[str, np.ndarray]


class BoundedRandomActionProcess:
    """Deterministic full-range and correlated random action generator."""

    def __init__(
        self,
        base_action: Tensor,
        *,
        seed: int,
        correlated_alpha: float = 0.25,
        correlated_tail_limit: float = 0.8,
    ) -> None:
        if base_action.ndim != 2 or base_action.shape[1] != 4:
            raise ValueError("base_action must have shape (N, 4).")
        if base_action.shape[0] < 2:
            raise ValueError("base_action must contain at least two environments.")
        if base_action.device.type != "cpu":
            raise ValueError("The authoritative random-action gate is CPU-only.")
        if not bool(torch.all(torch.isfinite(base_action))):
            raise ValueError("base_action must be finite.")
        if not 0.0 < float(correlated_alpha) <= 1.0:
            raise ValueError("correlated_alpha must lie in (0, 1].")
        if not 0.0 < float(correlated_tail_limit) <= 1.0:
            raise ValueError("correlated_tail_limit must lie in (0, 1].")
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(int(seed))
        self._current = base_action.detach().clone()
        self._alpha = float(correlated_alpha)
        self._tail_limit = float(correlated_tail_limit)
        environment_ids = torch.arange(base_action.shape[0], device=base_action.device)
        self.family_index = torch.remainder(environment_ids, len(RANDOM_ACTION_FAMILY_NAMES))

    def sample(self) -> Tensor:
        """Return one policy-rate action batch and advance process state."""

        random_target = 2.0 * torch.rand(
            self._current.shape,
            generator=self._generator,
            device=self._current.device,
            dtype=self._current.dtype,
        ) - 1.0
        iid_mask = self.family_index == 0
        correlated_mask = ~iid_mask
        next_action = self._current.clone()
        next_action[iid_mask] = random_target[iid_mask]
        interior_target = random_target[correlated_mask].clone()
        interior_target[:, 1:4] *= self._tail_limit
        next_action[correlated_mask] = (
            (1.0 - self._alpha) * self._current[correlated_mask]
            + self._alpha * interior_target
        )
        self._current.copy_(torch.clamp(next_action, -1.0, 1.0))
        return self._current.clone()

    def reset(self, reset_mask: Tensor, reset_action: Tensor) -> None:
        """Synchronize process state to environment reset actions."""

        if reset_mask.shape != (self._current.shape[0],) or reset_mask.dtype != torch.bool:
            raise ValueError("reset_mask must be a boolean tensor with shape (N,).")
        if reset_action.shape != self._current.shape:
            raise ValueError("reset_action must align with the process action batch.")
        self._current[reset_mask] = reset_action[reset_mask]


def summarize_random_action_trace(
    traces: dict[str, np.ndarray],
    *,
    num_envs: int,
    reconstruction_tolerance: float = 5.0e-6,
    action_tolerance: float = 1.0e-7,
) -> dict[str, object]:
    """Apply deterministic acceptance gates to one random-action trace."""

    required = {
        "sample_step",
        "environment_id",
        "episode_index",
        "family_index",
        "returned_reward",
        *REWARD_TERM_NAMES,
        "step_requested_action",
        "step_applied_action",
        "step_action_comparison_valid",
        "step_returned_reward",
        "step_observation_max_abs",
        "step_terminated",
        "step_truncated",
        "step_telemetry_reward_total_mean",
        "step_telemetry_terminated_fraction",
        "reset_count_by_environment",
    }
    missing = sorted(required.difference(traces))
    if missing:
        raise ValueError(f"Missing trace fields: {missing}")
    if num_envs < 2:
        raise ValueError("num_envs must be at least two.")

    numeric_arrays = [
        np.asarray(value) for value in traces.values() if np.asarray(value).dtype.kind in "fiu"
    ]
    all_finite = bool(all(np.all(np.isfinite(value)) for value in numeric_arrays))
    requested = np.asarray(traces["step_requested_action"], dtype=np.float64)
    applied = np.asarray(traces["step_applied_action"], dtype=np.float64)
    if requested.ndim != 3 or requested.shape[1:] != (num_envs, 4):
        raise ValueError("step_requested_action must have shape (steps, num_envs, 4).")
    if applied.shape != requested.shape:
        raise ValueError("step_applied_action must align with requested action.")
    action_comparison_valid = np.asarray(traces["step_action_comparison_valid"], dtype=bool)
    if action_comparison_valid.shape != requested.shape[:2]:
        raise ValueError("step_action_comparison_valid must have shape (steps, num_envs).")
    action_max_abs = float(np.max(np.abs(requested)))
    action_transfer_error = np.abs(requested[:, :, 1:4] - applied[:, :, 1:4])
    action_transfer_max_error = (
        float(np.max(action_transfer_error[action_comparison_valid]))
        if np.any(action_comparison_valid)
        else math.inf
    )

    stress_action = requested[:, 0::2, :].reshape(-1, 4)
    stress_p01 = np.quantile(stress_action, 0.01, axis=0)
    stress_p99 = np.quantile(stress_action, 0.99, axis=0)
    full_range_coverage = bool(np.all(stress_p01 < -0.90) and np.all(stress_p99 > 0.90))
    correlated_tail = requested[:, 1::2, 1:4]
    correlated_tail_max_abs = float(np.max(np.abs(correlated_tail)))

    returned_reward = np.asarray(traces["returned_reward"], dtype=np.float64)
    rebuilt_reward = np.asarray(traces["total"], dtype=np.float64)
    reward_reconstruction_max_error = float(np.max(np.abs(returned_reward - rebuilt_reward)))
    step_reward = np.asarray(traces["step_returned_reward"], dtype=np.float64)
    telemetry_reward = np.asarray(traces["step_telemetry_reward_total_mean"], dtype=np.float64)
    reward_telemetry_max_error = float(
        np.max(np.abs(np.mean(step_reward, axis=1) - telemetry_reward))
    )
    terminated = np.asarray(traces["step_terminated"], dtype=bool)
    truncated = np.asarray(traces["step_truncated"], dtype=bool)
    telemetry_terminated = np.asarray(
        traces["step_telemetry_terminated_fraction"], dtype=np.float64
    )
    termination_telemetry_max_error = float(
        np.max(np.abs(np.mean(terminated, axis=1) - telemetry_terminated))
    )

    observation_max_abs = float(np.max(np.asarray(traces["step_observation_max_abs"])))
    frequency_delta = np.asarray(traces["frequency_slew_penalty"], dtype=np.float64)
    tail_delta = np.asarray(traces["tail_action_delta_penalty"], dtype=np.float64)
    delta_in_contract = bool(
        np.all((frequency_delta >= 0.0) & (frequency_delta <= 1.0 + action_tolerance))
        and np.all((tail_delta >= 0.0) & (tail_delta <= 1.0 + action_tolerance))
    )
    sample_family = np.asarray(traces["family_index"], dtype=np.int64)
    combined_delta = frequency_delta + tail_delta
    delta_mean_by_family = [
        float(np.mean(combined_delta[sample_family == family_index]))
        for family_index in range(len(RANDOM_ACTION_FAMILY_NAMES))
    ]

    reset_count = np.asarray(traces["reset_count_by_environment"], dtype=np.int64)
    episode_index = np.asarray(traces["episode_index"], dtype=np.int64)
    reset_event_count = int(np.sum(reset_count))
    environments_reset = int(np.count_nonzero(reset_count > 0))
    environments_reset_repeatedly = int(np.count_nonzero(reset_count >= 2))
    post_reset_sample_count = int(np.count_nonzero(episode_index > 0))
    timeout_event_count = int(np.count_nonzero(truncated))

    gates = {
        "all_numeric_traces_finite": all_finite,
        "requested_actions_stay_in_normalized_bounds": action_max_abs <= 1.0 + action_tolerance,
        "applied_tail_actions_match_requested_actions": action_transfer_max_error <= action_tolerance,
        "iid_family_covers_both_action_extremes": full_range_coverage,
        "correlated_tail_family_stays_inside_0_8": (
            correlated_tail_max_abs <= 0.8 + action_tolerance
        ),
        "reward_reconstruction_matches_environment": (
            reward_reconstruction_max_error <= reconstruction_tolerance
        ),
        "reward_total_telemetry_matches_environment_mean": (
            reward_telemetry_max_error <= reconstruction_tolerance
        ),
        "termination_telemetry_matches_returned_done": (
            termination_telemetry_max_error <= reconstruction_tolerance
        ),
        "physical_frequency_slew_and_tail_delta_penalties_stay_in_contract": delta_in_contract,
        "observation_safety_clip_contract_holds": observation_max_abs <= 5.0 + action_tolerance,
        "automatic_reset_exercised": reset_event_count > 0 and environments_reset > 0,
        "repeated_reset_exercised": environments_reset_repeatedly > 0,
        "post_reset_samples_collected": post_reset_sample_count > 0,
        "no_episode_timeout": timeout_event_count == 0,
    }
    return {
        "all_cases_accepted": bool(all(gates.values())),
        "gates": gates,
        "metrics": {
            "requested_action_max_abs": action_max_abs,
            "action_transfer_max_abs_error": action_transfer_max_error,
            "stress_action_p01_by_channel": stress_p01.tolist(),
            "stress_action_p99_by_channel": stress_p99.tolist(),
            "correlated_tail_action_max_abs": correlated_tail_max_abs,
            "reward_reconstruction_max_abs_error": reward_reconstruction_max_error,
            "reward_total_telemetry_max_abs_error": reward_telemetry_max_error,
            "termination_telemetry_max_abs_error": termination_telemetry_max_error,
            "observation_max_abs": observation_max_abs,
            "combined_action_delta_penalty_mean_by_family": {
                name: delta_mean_by_family[index]
                for index, name in enumerate(RANDOM_ACTION_FAMILY_NAMES)
            },
            "reset_event_count": reset_event_count,
            "environments_reset": environments_reset,
            "environments_reset_repeatedly": environments_reset_repeatedly,
            "post_reset_sample_count": post_reset_sample_count,
            "timeout_event_count": timeout_event_count,
        },
    }


def run_random_action_reward_job(
    *,
    num_envs: int = 64,
    duration_s: float = 1.5,
    seed: int = 20260806,
    asset_path: Path | None = None,
    usd_dir: Path | None = None,
) -> PureRLRandomActionRewardResult:
    """Run a free-root bounded random-action gate after Isaac Sim starts."""

    if num_envs < 2 or num_envs % len(RANDOM_ACTION_FAMILY_NAMES) != 0:
        raise ValueError("num_envs must be a positive multiple of two.")
    if duration_s < 0.5 or duration_s > 3.0:
        raise ValueError("duration_s must lie in [0.5, 3.0].")
    from isaaclab.utils.math import euler_xyz_from_quat
    from flapping_bot.direct.flapping_bot.straight_flight_env import FlappingBotStraightFlightEnv

    cfg = _make_cfg(num_envs=num_envs, seed=seed, asset_path=asset_path, usd_dir=usd_dir)
    env = FlappingBotStraightFlightEnv(cfg)
    try:
        observations, _extras = env.reset()
        process = BoundedRandomActionProcess(env._act_cmd, seed=seed + 1)
        episode_index = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        reset_count = torch.zeros_like(episode_index)
        total_policy_steps = int(math.ceil(float(duration_s) / float(env.step_dt)))
        sample_trace: dict[str, list[np.ndarray]] = {
            "sample_step": [],
            "environment_id": [],
            "episode_index": [],
            "family_index": [],
            "actual_frequency_hz": [],
            "cross_track_error_m": [],
            "height_error_m": [],
            "along_track_velocity_mps": [],
            "cross_track_velocity_mps": [],
            "vertical_velocity_mps": [],
            "roll_rad": [],
            "pitch_rad": [],
            "angular_velocity_body_rad_s": [],
            "applied_action": [],
            "previous_applied_action": [],
            "returned_reward": [],
        }
        sample_trace.update({name: [] for name in REWARD_TERM_NAMES})
        step_trace: dict[str, list[np.ndarray | float | int]] = {
            "step_index": [],
            "step_requested_action": [],
            "step_applied_action": [],
            "step_action_comparison_valid": [],
            "step_returned_reward": [],
            "step_observation_max_abs": [],
            "step_terminated": [],
            "step_truncated": [],
        }
        for key in TELEMETRY_KEYS:
            step_trace[f"telemetry__{key.replace('/', '__')}"] = []

        initial_observation = observations["policy"]
        if initial_observation.shape != (env.num_envs, 555):
            raise RuntimeError(f"Unexpected initial observation shape: {initial_observation.shape}")
        for policy_step in range(total_policy_steps):
            previous_action = env._pure_rl_previous_reward_action.detach().clone()
            requested_action = process.sample()
            observations, reward, terminated, truncated, extras = env.step(requested_action)
            done = terminated | truncated
            policy_observation = observations["policy"]
            if policy_observation.shape != (env.num_envs, 555):
                raise RuntimeError(f"Unexpected observation shape: {policy_observation.shape}")

            step_trace["step_index"].append(policy_step)
            step_trace["step_requested_action"].append(_to_numpy(requested_action))
            step_trace["step_applied_action"].append(_to_numpy(env._act_cmd))
            step_trace["step_action_comparison_valid"].append(_to_numpy(~done))
            step_trace["step_returned_reward"].append(_to_numpy(reward))
            step_trace["step_observation_max_abs"].append(
                _to_numpy(torch.max(torch.abs(policy_observation), dim=1).values)
            )
            step_trace["step_terminated"].append(_to_numpy(terminated))
            step_trace["step_truncated"].append(_to_numpy(truncated))
            log = extras.get("log", {}) if isinstance(extras, dict) else {}
            for key in TELEMETRY_KEYS:
                step_trace[f"telemetry__{key.replace('/', '__')}"].append(
                    _telemetry_scalar(log, key)
                )

            ids = torch.nonzero(~done, as_tuple=False).squeeze(1)
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
                    frequency_slew_hz_per_s=env._frequency_slew_hz_per_s,
                    applied_action=env._act_cmd,
                    previous_applied_action=previous_action,
                    config=env.cfg.pure_rl_reward_cfg,
                )
                sample_trace["sample_step"].append(
                    np.full((ids.numel(),), policy_step, dtype=np.int64)
                )
                sample_trace["environment_id"].append(_to_numpy(ids).astype(np.int64))
                sample_trace["episode_index"].append(_to_numpy(episode_index[ids]).astype(np.int64))
                sample_trace["family_index"].append(
                    _to_numpy(process.family_index[ids]).astype(np.int64)
                )
                tensors = {
                    "actual_frequency_hz": env._freq,
                    "cross_track_error_m": cross_track_error_m,
                    "height_error_m": height_error_m,
                    "along_track_velocity_mps": along_track_velocity_mps,
                    "cross_track_velocity_mps": cross_track_velocity_mps,
                    "vertical_velocity_mps": vertical_velocity_mps,
                    "roll_rad": roll_rad,
                    "pitch_rad": pitch_rad,
                    "angular_velocity_body_rad_s": env._robot.data.root_ang_vel_b,
                    "applied_action": env._act_cmd,
                    "previous_applied_action": previous_action,
                    "returned_reward": reward,
                }
                tensors.update(terms.as_dict())
                for name, value in tensors.items():
                    sample_trace[name].append(_to_numpy(value[ids]))

            reset_count += done.to(dtype=torch.long)
            episode_index += done.to(dtype=torch.long)
            process.reset(done, env._act_cmd)

        if not sample_trace["sample_step"]:
            raise RuntimeError("No non-terminal random-action sample was collected.")
        traces = {name: np.concatenate(values, axis=0) for name, values in sample_trace.items()}
        traces.update({name: np.asarray(values) for name, values in step_trace.items()})
        traces["step_telemetry_reward_total_mean"] = traces[
            "telemetry__PureRLReward__total"
        ].copy()
        traces["step_telemetry_terminated_fraction"] = traces[
            "telemetry__PureRLTermination__terminated_fraction"
        ].copy()
        traces["reset_count_by_environment"] = _to_numpy(reset_count).astype(np.int64)
        summary = summarize_random_action_trace(traces, num_envs=num_envs)
        telemetry_arrays = [
            traces[f"telemetry__{key.replace('/', '__')}"] for key in TELEMETRY_KEYS
        ]
        cause_event_counts = {
            cause: int(
                round(
                    float(
                        np.sum(
                            traces[f"telemetry__PureRLTermination__{cause}_fraction"]
                        )
                    )
                    * num_envs
                )
            )
            for cause in ("ground", "tilt", "cross_track", "height_error")
        }
        summary.update(
            {
                "schema_version": 1,
                "experiment": "pure_rl_curriculum1_bounded_random_action_gate",
                "configuration": {
                    "num_envs": int(num_envs),
                    "seed": int(seed),
                    "action_seed": int(seed + 1),
                    "duration_requested_s": float(duration_s),
                    "policy_dt_s": float(env.step_dt),
                    "physics_dt_s": float(env.physics_dt),
                    "total_policy_steps": total_policy_steps,
                    "action_families": list(RANDOM_ACTION_FAMILY_NAMES),
                    "correlated_alpha": 0.25,
                    "correlated_tail_limit": 0.8,
                },
                "telemetry": {
                    "required_keys": list(TELEMETRY_KEYS),
                    "all_required_keys_present_and_finite": bool(
                        all(np.all(np.isfinite(values)) for values in telemetry_arrays)
                    ),
                    "termination_cause_event_counts": cause_event_counts,
                },
                "claim_boundary": (
                    "This gate validates bounded random exploration, reward/termination wiring and reset "
                    "lifecycle. It does not test PPO learning, convergence or closed-loop task success."
                ),
            }
        )
        summary["gates"]["all_required_telemetry_present_and_finite"] = summary["telemetry"][
            "all_required_keys_present_and_finite"
        ]
        summary["all_cases_accepted"] = bool(all(summary["gates"].values()))
        return PureRLRandomActionRewardResult(summary=summary, traces=traces)
    finally:
        env.close()


__all__ = [
    "BoundedRandomActionProcess",
    "PureRLRandomActionRewardResult",
    "RANDOM_ACTION_FAMILY_NAMES",
    "run_random_action_reward_job",
    "summarize_random_action_trace",
]
