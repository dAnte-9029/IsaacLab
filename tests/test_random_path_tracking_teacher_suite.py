from pathlib import Path
from types import SimpleNamespace

from scripts.flapping_px4.run_random_path_tracking_teacher_suite import (
    analyze_trajectory_progress,
    aggregate_episode_summaries,
    build_episode_command,
    build_episode_seeds,
    summarize_episode_records,
)


def test_build_episode_seeds_uses_base_seed_and_count() -> None:
    assert build_episode_seeds(seed_start=11, episodes=4) == [11, 12, 13, 14]


def test_aggregate_episode_summaries_reports_completion_rate_and_worst_seed() -> None:
    summary = aggregate_episode_summaries(
        [
            {
                "mission_seed": 1,
                "completed_path": True,
                "mean_abs_lateral_error_m": 0.1,
                "mean_abs_height_error_m": 0.2,
                "mean_abs_align_error_deg": 1.0,
                "final_progress_ratio": 1.0,
            },
            {
                "mission_seed": 2,
                "completed_path": False,
                "mean_abs_lateral_error_m": 0.6,
                "mean_abs_height_error_m": 0.4,
                "mean_abs_align_error_deg": 3.0,
                "final_progress_ratio": 0.8,
            },
        ]
    )
    assert summary["episodes"] == 2
    assert summary["completion_rate"] == 0.5
    assert summary["worst_progress_seed"] == 2
    assert summary["mean_episode_lateral_error_m"] == 0.35


def test_aggregate_episode_summaries_reports_jump_free_completion_rate() -> None:
    summary = aggregate_episode_summaries(
        [
            {
                "mission_seed": 1,
                "completed_path": True,
                "has_progress_jump": False,
                "max_single_step_progress_jump_m": 0.2,
                "final_progress_ratio": 1.0,
            },
            {
                "mission_seed": 2,
                "completed_path": True,
                "has_progress_jump": True,
                "has_early_progress_jump": True,
                "max_single_step_progress_jump_m": 125.0,
                "final_progress_ratio": 1.0,
            },
        ]
    )

    assert summary["completion_rate"] == 1.0
    assert summary["jump_free_completion_rate"] == 0.5
    assert summary["episodes_with_progress_jump"] == 1
    assert summary["episodes_with_early_progress_jump"] == 1
    assert summary["suspicious_jump_seeds"] == [2]
    assert summary["early_jump_seeds"] == [2]
    assert summary["max_single_step_progress_jump_m"] == 125.0


def test_build_episode_command_uses_unique_portable_root() -> None:
    args = SimpleNamespace(
        task="Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0",
        num_envs=1,
        steps=3400,
        auto_extend_steps=True,
        nominal_speed_mps=7.0,
        completion_margin_s=2.0,
        height_sp=10.0,
        metrics_warmup_s=3.0,
        straight_length_m=60.0,
        turn_radius_m=20.0,
        loiter_radius_m=20.0,
        turn_sweep_deg=90.0,
        loiter_turns=1.0,
        climb_delta_m=3.0,
        path_manager_max_roll_deg=35.0,
        path_manager_max_flight_path_angle_deg=10.0,
        mission_num_segments_min=2,
        mission_num_segments_max=4,
        mission_allow_straight=True,
        mission_allow_turn=True,
        mission_allow_loiter=True,
        mission_allow_climb_on_straight=True,
        wind_x_mps=0.0,
        wind_y_mps=0.0,
        wind_ou=False,
        wind_ou_tau_s=2.0,
        wind_ou_sigma_x_mps=0.0,
        wind_ou_sigma_y_mps=0.0,
        wind_ou_clip_to_range=False,
        print_every=0,
        headless=True,
    )

    cmd = build_episode_command(
        args,
        mission_seed=12,
        mission_label="random_seed_000012",
        episodes_root=Path("logs/episodes"),
        portable_root=Path("logs/portable/random_seed_000012"),
    )

    assert "--portable-root" in cmd
    portable_root_index = cmd.index("--portable-root") + 1
    assert cmd[portable_root_index] == "logs/portable/random_seed_000012"


def test_summarize_episode_records_counts_timeouts_and_failures() -> None:
    summary = summarize_episode_records(
        [
            {"return_code": 0, "timed_out": False},
            {"return_code": 124, "timed_out": True},
            {"return_code": 1, "timed_out": False},
        ]
    )

    assert summary["episodes_attempted"] == 3
    assert summary["episodes_succeeded"] == 1
    assert summary["failed_launches"] == 2
    assert summary["timed_out_episodes"] == 1


def test_analyze_trajectory_progress_flags_large_progress_jump(tmp_path: Path) -> None:
    traj_path = tmp_path / "trajectory_env0.csv"
    traj_path.write_text(
        "\n".join(
            [
                "episode_id,step,t,progress_s",
                "0,0,0.0,0.0",
                "0,1,0.1,1.0",
                "0,2,0.2,130.0",
            ]
        )
        + "\n"
    )

    summary = analyze_trajectory_progress(traj_path)

    assert summary["trajectory_available"] is True
    assert summary["has_progress_jump"] is True
    assert summary["has_early_progress_jump"] is True
    assert summary["suspicious_jump_count"] == 1
    assert summary["first_suspicious_jump_time_s"] == 0.2
    assert summary["max_single_step_progress_jump_m"] == 129.0
