"""First-episode C4 diagnostics; sampled paths and scores do not certify promotion."""

from dataclasses import asdict, replace
from pathlib import Path
import json

import numpy as np
import torch


SCHEMA = "pure_rl_continuous_curve_zero_shot_v1"
METRICS = (
    "cross_track_error_m", "height_error_m", "along_track_progress_m",
    "along_track_velocity_mps", "abs_roll_rad", "active_curvature_rad_per_m",
    "active_slope_rad", "actual_flap_frequency_hz", "tail_limit_active",
    "frequency_governor_limited", "ground_termination", "tilt_termination",
    "cross_track_termination", "height_termination", "roll_limit_termination",
)


def resample_preview_points(points: torch.Tensor, spacing: float, progress: torch.Tensor,
                            speed: torch.Tensor, scale: float) -> torch.Tensor:
    """Interpolate five preview points; speed is already bounded by the caller."""
    import math
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Preview time scale must be finite and positive")
    times = points.new_tensor((.12, .24, .36, .48, .60)) * scale
    index = ((progress[:, None] + speed[:, None] * times) / spacing).clamp(0, points.shape[1] - 1)
    lower = index.long().clamp(max=points.shape[1] - 2)
    fraction = (index - lower).unsqueeze(-1)
    rows = torch.arange(points.shape[0], device=points.device)[:, None]
    return torch.lerp(points[rows, lower], points[rows, lower + 1], fraction)


def score_episode(trace: dict[str, np.ndarray], *, terminated: bool, timed_out: bool) -> dict:
    """Score one complete first episode, rejecting nonfinite or partial traces."""
    finite = bool(trace and all(v.size and np.isfinite(v).all() for v in trace.values()))
    full_horizon = bool(finite and trace["time_s"][-1] >= 20.0 - 1.0 / 60.0 - 1e-6)
    row = {"finite": finite, "survival": bool(timed_out and not terminated and full_horizon),
           "terminated": terminated, "time_out": timed_out}
    if not finite:
        return {**row, "completion": False, "success": False}
    row["completion"] = bool(np.max(trace["along_track_progress_m"]) >= 140.0)
    for axis, name in (("horizontal", "cross_track_error_m"), ("vertical", "height_error_m")):
        error = np.abs(trace[name])
        row[f"mean_{axis}_error_m"] = float(error.mean())
        row[f"p95_{axis}_error_m"] = float(np.quantile(error, .95))
    row["reverse_motion_fraction"] = float((trace["along_track_velocity_mps"] < 0).mean())
    row["p95_roll_deg"] = float(np.rad2deg(np.quantile(trace["abs_roll_rad"], .95)))
    row["roll_limit_termination"] = bool(trace["roll_limit_termination"].any())
    row["final_progress_m"] = float(trace["along_track_progress_m"][-1])
    row["duration_s"] = float(trace["time_s"][-1])
    entered = trace["along_track_progress_m"] >= 20.0
    speed = np.linalg.norm(trace["velocity_world_mps"], axis=-1)
    row["design_speed_violation_fraction_after_entry"] = (
        float(((speed[entered] < 7) | (speed[entered] > 12)).mean()) if entered.any() else None
    )
    row["mean_frequency_gap_hz"] = float(np.abs(
        trace["requested_frequency_hz"] - trace["applied_frequency_hz"]).mean())
    row["tail_limit_fraction"] = float(trace["tail_limit_active"].mean())
    row["termination_causes"] = {name: bool(trace[name + "_termination"].any())
                                 for name in ("ground", "tilt", "cross_track", "height", "roll_limit")}
    row["success"] = bool(row["survival"] and row["completion"] and all(
        row[f"{stat}_{axis}_error_m"] <= limit
        for axis in ("horizontal", "vertical") for stat, limit in (("mean", .5), ("p95", 1.5))
    ))
    return row


def summarize(rows: list[dict], expected_ids: list[str]) -> dict:
    """Report balanced case rates; missing or duplicate cases fail closed."""
    if not expected_ids:
        raise ValueError("Expected case IDs must not be empty")
    complete = (len(rows) == len(expected_ids) and len(set(expected_ids)) == len(expected_ids)
                and sorted(r["curve_id"] for r in rows) == sorted(expected_ids)
                and all(r["terminated"] or r["time_out"] for r in rows))
    result = {"schema": SCHEMA, "evaluation_complete": complete, "promotion_eligible": False,
              "cases": rows, "case_count": len(rows), "expected_case_count": len(expected_ids)}
    for key in ("survival", "completion", "success"):
        result[f"{key}_rate"] = sum(bool(r[key]) for r in rows) / len(expected_ids)
    numeric = ("mean_horizontal_error_m", "p95_horizontal_error_m", "mean_vertical_error_m",
               "p95_vertical_error_m", "reverse_motion_fraction", "p95_roll_deg", "mean_frequency_gap_hz")
    result["case_balanced_means"] = {
        key: float(np.mean([r[key] for r in rows]))
        if rows and all(key in r for r in rows) else None for key in numeric
    }
    result["roll_limit_termination_count"] = sum(bool(r.get("roll_limit_termination")) for r in rows)
    result["slices"] = {
        field: {str(value): {"count": len(group), "success_rate": sum(r["success"] for r in group) / len(group)}
                for value in sorted({r[field] for r in rows})
                if (group := [r for r in rows if r[field] == value])}
        for field in ("family", "horizontal_mirror", "vertical_mirror")
    }
    return result


def evaluate_checkpoint(checkpoint: Path, curves: Path, output: Path, root: Path, smoke_steps: int = 0,
                        preview_time_scale: float = 1.0, tail_checkpoint: Path | None = None, tail_channels: tuple[int, ...] = (1, 2, 3)) -> dict:
    """Run deterministic CPU-native inference after the caller starts Kit."""
    import torch
    import yaml
    from tensordict import TensorDict
    from flapping_bot.direct.flapping_bot.straight_flight_env import (
        FlappingBotStraightFlightEnv, FlappingBotStraightFlightDeLaurierMeasuredPureRLC4EnvCfg,
    )
    from flapping_bot.px4_like.pure_rl_split_actor import PureRLSplitActorCritic
    from flapping_bot.path_tracking.continuous_curve import load_continuous_curves, save_continuous_curves

    specs = load_continuous_curves(curves)
    if smoke_steps:
        specs = specs[:6]
    replay = output / "curves.json"
    save_continuous_curves(replay, specs)
    source = json.loads((checkpoint.parent / "curriculum_source.json").read_text())
    agent = yaml.safe_load((checkpoint.parent / "params/agent.yaml").read_text())
    policy_cfg = dict(agent["policy"])
    if (policy_cfg.pop("class_name") != "PureRLSplitActorCritic"
            or str(source["heading_canonical_observation"]).lower() != "true"
            or agent.get("empirical_normalization")
            or agent.get("clip_actions") is not None):
        raise ValueError("Unsupported checkpoint actor/observation/action contract")

    tail_source = None
    if tail_checkpoint is not None:
        tail_source = json.loads((tail_checkpoint.parent / "curriculum_source.json").read_text())
        tail_agent = yaml.safe_load((tail_checkpoint.parent / "params/agent.yaml").read_text())
        if (tail_agent["policy"] != agent["policy"]
                or str(tail_source["heading_canonical_observation"]).lower() != "true"
                or tail_agent.get("empirical_normalization")
                or tail_agent.get("clip_actions") is not None):
            raise ValueError("Tail checkpoint must share the actor/observation/action contract")

    class TraceEnv(FlappingBotStraightFlightEnv):
        def _query_pure_rl_spatial_path(self):
            query = super()._query_pure_rl_spatial_path()
            if preview_time_scale == 1.0:
                return query
            speed = (self._robot.data.root_lin_vel_w * query.tangent_world).sum(dim=1).clamp(
                self.cfg.pure_rl_preview_minimum_speed_mps, self.cfg.pure_rl_preview_maximum_speed_mps)
            preview = resample_preview_points(
                self._pure_rl_spatial_path.points_world_m, self._pure_rl_spatial_path.sample_spacing_m,
                query.progress_m, speed, preview_time_scale)
            # Keep the base cache and all geometry/reward/termination fields unchanged.
            return replace(query, preview_points_world_m=preview)

        def _get_rewards(self):
            reward = super()._get_rewards()
            # DirectRLEnv computes dones and rewards before resetting finished rows.
            self.trace_snapshot = {
                key: getattr(self, "_eval_pure_rl_" + key).detach().cpu().numpy().copy()
                for key in METRICS
            }
            values = {
                "requested_frequency_hz": self._requested_frequency_hz,
                "applied_frequency_hz": self._applied_frequency_hz,
                "phase_rad": self._phase,
                "position_world_m": self._robot.data.root_pos_w - self.scene.env_origins,
                "velocity_world_mps": self._robot.data.root_lin_vel_w,
                "quaternion_wxyz": self._robot.data.root_quat_w,
            }
            for part in ("rudder", "left_elevon", "right_elevon"):
                for kind in ("exec", "actual"):
                    values[f"{kind}_{part}_rad"] = getattr(self, f"_debug_last_{kind}_{part}_rad")
            for key, value in values.items():
                self.trace_snapshot[key] = value.detach().cpu().numpy().copy()
            return reward

    cfg = FlappingBotStraightFlightDeLaurierMeasuredPureRLC4EnvCfg()
    # Preserve the historical CLI scale contract after C4's default becomes 1.2 s.
    cfg.pure_rl_spatial_preview_times_s = (.12, .24, .36, .48, .60)
    cfg.scene.num_envs = len(specs)
    cfg.scene.env_spacing = 5.0
    cfg.seed = 0
    cfg.pure_rl_continuous_curve_file = str(replay)
    cfg.robot = cfg.robot.replace(spawn=cfg.robot.spawn.replace(
        asset_path=str(root / "source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf"),
        usd_dir=str(output / "generated_assets"),
    ))
    env = TraceEnv(cfg)
    try:
        obs, _ = env.reset()
        if obs["policy"].shape != (len(specs), 555) or env.device != "cpu":
            raise ValueError("C4 requires 555 observations and CPU-native physics")
        if env._native_holonomic.get_active_joint_count() != len(specs):
            raise RuntimeError("Missing native holonomic constraints")
        td = TensorDict(obs, batch_size=[len(specs)])
        actor = PureRLSplitActorCritic(td, {"policy": ["policy"], "critic": ["policy"]}, 4, **policy_cfg)
        actor.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state_dict"], strict=True)
        actor.eval()
        tail_actor = None
        if tail_checkpoint is not None:
            tail_actor = PureRLSplitActorCritic(td, {"policy": ["policy"], "critic": ["policy"]}, 4, **policy_cfg)
            tail_actor.load_state_dict(torch.load(tail_checkpoint, map_location="cpu", weights_only=True)["model_state_dict"], strict=True)
            tail_actor.eval()
        histories = [[] for _ in specs]
        done = np.zeros(len(specs), dtype=bool)
        terminated = done.copy()
        timed_out = done.copy()
        with torch.inference_mode():
            for step in range(smoke_steps or (env.max_episode_length + 1)):
                policy_input = obs["policy"].detach().cpu().numpy().copy()
                actions = actor.act_inference(TensorDict(obs, batch_size=[len(specs)]))
                component_actions = {}
                if tail_actor is not None:
                    tail_actions = tail_actor.act_inference(TensorDict(obs, batch_size=[len(specs)]))
                    component_actions = {
                        "frequency_actor_raw_action": actions.detach().cpu().numpy().copy(),
                        "tail_actor_raw_action": tail_actions.detach().cpu().numpy().copy(),
                    }
                    actions = combine_frequency_tail_actions(actions, tail_actions, tail_channels)
                if not torch.isfinite(actions).all():
                    raise RuntimeError("Nonfinite actor action")
                raw_actions = actions.detach().cpu().numpy().copy()
                obs, reward, term, trunc, _ = env.step(actions)
                snapshot = {**env.trace_snapshot, **component_actions, "policy_input": policy_input, "raw_action": raw_actions,
                            "time_s": np.full(len(specs), (step + 1) * env.step_dt)}
                for index in np.flatnonzero(~done):
                    histories[index].append({k: v[index] for k, v in snapshot.items()})
                    terminated[index] = bool(term[index])
                    timed_out[index] = bool(trunc[index])
                done |= term.cpu().numpy() | trunc.cpu().numpy()
                if done.all():
                    break
        rows = []
        traces_dir = output / "traces"
        traces_dir.mkdir()
        for index, spec in enumerate(specs):
            trace = {key: np.asarray([s[key] for s in histories[index]]) for key in histories[index][0]}
            np.savez_compressed(traces_dir / f"{spec.curve_id}.npz", **trace)
            rows.append({**asdict(spec), "family": spec.curve_id.split("_")[0],
                         **score_episode(trace, terminated=bool(terminated[index]), timed_out=bool(timed_out[index]))})
        result = summarize(rows, [s.curve_id for s in specs])
        result.update(smoke_only=bool(smoke_steps), checkpoint_source=source,
                      tail_checkpoint=str(tail_checkpoint) if tail_checkpoint else None,
                      tail_checkpoint_source=tail_source, replacement_tail_channels=list(tail_channels) if tail_checkpoint else [],
                      preview_times_s=[t * preview_time_scale for t in (.12, .24, .36, .48, .60)],
                      trace_timing="policy_input/raw_action at t-dt; remaining values at t before autoreset",
                      plant={"device": env.device, "dt": cfg.sim.dt, "decimation": cfg.decimation,
                             "wing_drive_variant": cfg.wing_drive_variant})
        return result
    finally:
        import omni.physx
        omni.physx.get_physx_simulation_interface().detach_stage()
        env.close()


def combine_frequency_tail_actions(frequency_actions: "torch.Tensor", tail_actions: "torch.Tensor",
                                   tail_channels: tuple[int, ...] = (1, 2, 3)) -> "torch.Tensor":
    """Combine channel 0 and channels 1..3 evaluated on the same current input."""
    import torch
    if (frequency_actions.shape != tail_actions.shape or frequency_actions.ndim != 2
            or frequency_actions.shape[1] != 4
            or frequency_actions.device != tail_actions.device
            or frequency_actions.dtype != tail_actions.dtype):
        raise ValueError("Expected matching (N, 4) action tensors on the same device/dtype")
    if not torch.isfinite(frequency_actions).all() or not torch.isfinite(tail_actions).all():
        raise ValueError("Nonfinite component action")
    if not tail_channels or len(set(tail_channels)) != len(tail_channels) or any(c not in (1, 2, 3) for c in tail_channels):
        raise ValueError("Replacement channels must be a nonempty unique subset of 1, 2, 3")
    result = frequency_actions.clone()
    result[:, list(tail_channels)] = tail_actions[:, list(tail_channels)]
    return result
