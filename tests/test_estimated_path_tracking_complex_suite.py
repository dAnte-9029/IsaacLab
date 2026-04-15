from pathlib import Path
from types import SimpleNamespace

from scripts.flapping_px4.run_estimated_path_tracking_complex_suite import (
    _configure_scenario,
    aggregate_case_summaries,
    build_case_command,
    build_complex_suite_cases,
    parse_seed_list,
)


def test_parse_seed_list_handles_csv_values() -> None:
    assert parse_seed_list("25,30,33") == [25, 30, 33]
    assert parse_seed_list(" 36 ") == [36]


def test_build_complex_suite_cases_uses_expected_default_layout() -> None:
    cases = build_complex_suite_cases()

    assert [case["name"] for case in cases] == [
        "multi_segment",
        "case02_seed025",
        "case03_seed030",
        "case04_seed033",
        "case05_seed036",
    ]
    assert cases[0]["mission_mode"] == "fixed"
    assert cases[0]["phase"] == "multi_segment"
    assert [case["mission_seed"] for case in cases[1:]] == [25, 30, 33, 36]
    assert all(int(case["steps"]) > 0 for case in cases)


def test_build_complex_suite_cases_can_focus_seed_subset_without_fixed_case() -> None:
    cases = build_complex_suite_cases(random_seeds=[30, 33], include_fixed_multi_segment=False)

    assert [case["name"] for case in cases] == ["case01_seed030", "case02_seed033"]
    assert [case["mission_seed"] for case in cases] == [30, 33]
    assert all(case["mission_mode"] == "random" for case in cases)


def test_build_case_command_uses_estimated_baseline_runtime_contract() -> None:
    args = SimpleNamespace(
        task="Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0",
        num_envs=1,
        device="cuda:0",
        teacher_state_source="estimated",
        policy_state_source="estimated",
        imu_source="synthetic",
        controller_tuning_profile="estimated_teacher",
        total_mass_kg_override=0.95,
        steps=5200,
        auto_extend_steps=True,
        nominal_speed_mps=7.0,
        completion_margin_s=3.0,
        height_sp=10.0,
        metrics_warmup_s=3.0,
        straight_length_m=80.0,
        turn_radius_m=25.0,
        loiter_radius_m=40.0,
        turn_sweep_deg=90.0,
        loiter_turns=1.5,
        climb_delta_m=3.0,
        path_manager_max_roll_deg=35.0,
        path_manager_max_flight_path_angle_deg=10.0,
        path_warmup_enabled=True,
        path_warmup_straight_length_m=25.0,
        teacher_roll_kd=None,
        teacher_max_roll_deg=None,
        teacher_inner_elevon_pitch_rate_limit_per_s=None,
        teacher_inner_elevon_roll_rate_limit_per_s=None,
        wind_x_mps=0.0,
        wind_y_mps=0.0,
        wind_ou=False,
        wind_ou_tau_s=2.0,
        wind_ou_sigma_x_mps=0.0,
        wind_ou_sigma_y_mps=0.0,
        wind_ou_clip_to_range=False,
        wind_x_range_min_mps=None,
        wind_x_range_max_mps=None,
        wind_y_range_min_mps=None,
        wind_y_range_max_mps=None,
        print_every=0,
        headless=True,
    )
    case = build_complex_suite_cases()[1]

    cmd = build_case_command(
        args,
        case=case,
        cases_root=Path("logs/cases"),
        portable_root=Path("logs/portable/case02_seed025"),
    )

    assert "--teacher_state_source" in cmd
    assert cmd[cmd.index("--teacher_state_source") + 1] == "estimated"
    assert "--policy_state_source" in cmd
    assert cmd[cmd.index("--policy_state_source") + 1] == "estimated"
    assert "--imu_source" in cmd
    assert cmd[cmd.index("--imu_source") + 1] == "synthetic"
    assert "--controller_tuning_profile" in cmd
    assert cmd[cmd.index("--controller_tuning_profile") + 1] == "estimated_teacher"
    assert "--total_mass_kg_override" in cmd
    assert cmd[cmd.index("--total_mass_kg_override") + 1] == "0.95"
    assert "--portable-root" in cmd
    assert cmd[cmd.index("--portable-root") + 1] == "logs/portable/case02_seed025"


def test_configure_scenario_assigns_nonzero_ou_gust_clip_ranges() -> None:
    args = SimpleNamespace(
        scenario="ou_gust",
        steady_crosswind_x_mps=0.0,
        steady_crosswind_y_mps=2.0,
        ou_gust_sigma_x_mps=1.0,
        ou_gust_sigma_y_mps=1.5,
        ou_gust_clip_sigma=3.0,
        wind_x_mps=0.0,
        wind_y_mps=0.0,
        wind_ou=False,
        wind_ou_clip_to_range=False,
        wind_ou_sigma_x_mps=0.0,
        wind_ou_sigma_y_mps=0.0,
        wind_x_range_min_mps=None,
        wind_x_range_max_mps=None,
        wind_y_range_min_mps=None,
        wind_y_range_max_mps=None,
        imu_source="synthetic",
    )

    configured = _configure_scenario(args)

    assert configured.wind_ou is True
    assert configured.wind_ou_clip_to_range is True
    assert configured.wind_x_range_min_mps == -3.0
    assert configured.wind_x_range_max_mps == 3.0
    assert configured.wind_y_range_min_mps == -4.5
    assert configured.wind_y_range_max_mps == 4.5


def test_aggregate_case_summaries_reports_completion_and_worst_case() -> None:
    summary = aggregate_case_summaries(
        [
            {
                "case_name": "multi_segment",
                "completed_path": True,
                "mean_abs_lateral_error_m": 0.15,
                "mean_abs_height_error_m": 0.60,
                "p95_abs_lateral_error_m": 0.40,
                "p95_abs_height_error_m": 1.55,
                "final_progress_ratio": 0.995,
            },
            {
                "case_name": "case03_seed030",
                "completed_path": True,
                "mean_abs_lateral_error_m": 0.28,
                "mean_abs_height_error_m": 0.63,
                "p95_abs_lateral_error_m": 0.68,
                "p95_abs_height_error_m": 1.49,
                "final_progress_ratio": 0.995,
            },
        ]
    )

    assert summary["cases"] == 2
    assert summary["completion_rate"] == 1.0
    assert summary["worst_mean_lateral_case"] == "case03_seed030"
    assert summary["worst_p95_lateral_case"] == "case03_seed030"
    assert summary["mean_case_lateral_error_m"] == 0.215
