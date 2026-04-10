from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.flapping_px4.run_tail_balance_grid_search import (
    _collect_rollout_metrics,
    _baseline_metrics_from_row,
    build_rollout_command,
    build_search_parser,
    build_stage1_grid,
    build_refinement_grid,
    compute_saturation_fractions,
    evaluate_hard_constraints,
    rank_candidate_rows,
    score_candidate,
)


def test_build_stage1_grid_has_expected_cartesian_order() -> None:
    pairs = build_stage1_grid(fixed_values=[0.6, 0.8, 1.0], elevon_values=[1.0, 1.4])
    assert pairs == [
        (0.6, 1.0),
        (0.6, 1.4),
        (0.8, 1.0),
        (0.8, 1.4),
        (1.0, 1.0),
        (1.0, 1.4),
    ]


def test_build_refinement_grid_deduplicates_and_clamps() -> None:
    refined = build_refinement_grid(
        top_candidates=[(0.6, 1.0)],
        fixed_step=0.05,
        elevon_step=0.1,
        fixed_bounds=(0.6, 0.7),
        elevon_bounds=(1.0, 1.1),
        existing_pairs={(0.6, 1.0)},
    )
    assert refined == [
        (0.6, 1.1),
        (0.65, 1.0),
        (0.65, 1.1),
    ]


def test_evaluate_hard_constraints_flags_straight_height_and_random_completion() -> None:
    baseline = {
        "straight_completed": True,
        "straight_height_error": 1.0,
        "random_completion_rate": 0.9,
    }
    candidate = {
        "straight_completed": False,
        "straight_height_error": 1.051,
        "random_completion_rate": 0.84,
    }

    result = evaluate_hard_constraints(candidate, baseline)

    assert result["feasible"] is False
    assert result["failed_constraints"] == [
        "straight_completion_guard",
        "straight_height_guard",
        "random_completion_guard",
    ]


def test_evaluate_hard_constraints_flags_canonical_early_failure() -> None:
    baseline = {
        "straight_completed": True,
        "straight_height_error": 1.0,
        "random_completion_rate": 0.9,
    }
    candidate = {
        "straight_completed": True,
        "straight_height_error": 1.0,
        "random_completion_rate": 0.9,
        "canonical_early_failure": True,
    }

    result = evaluate_hard_constraints(candidate, baseline)

    assert result["feasible"] is False
    assert result["failed_constraints"] == ["canonical_early_failure_guard"]


def test_score_candidate_matches_weighted_formula() -> None:
    baseline = {
        "turn_height_error": 4.0,
        "loiter_height_error": 5.0,
        "random_height_error": 3.0,
        "random_progress": 0.8,
        "straight_height_error": 2.0,
        "canonical_done_t_s": 5.0,
        "canonical_early_height_drop_m": 1.0,
    }
    candidate = {
        "turn_height_error": 3.0,
        "loiter_height_error": 4.0,
        "random_height_error": 2.4,
        "random_progress": 0.9,
        "straight_height_error": 1.8,
        "canonical_done_t_s": 6.0,
        "canonical_early_height_drop_m": 0.6,
        "canonical_terminated_frac": 1.0 / 3.0,
        "freq_sat_frac": 0.2,
        "pitch_sat_frac": 0.4,
    }

    score = score_candidate(candidate, baseline)

    expected = (
        0.35 * ((4.0 - 3.0) / 4.0)
        + 0.35 * ((5.0 - 4.0) / 5.0)
        + 0.15 * ((3.0 - 2.4) / 3.0)
        + 0.10 * ((0.9 - 0.8) / 0.8)
        + 0.05 * (((1.05 * 2.0) - 1.8) / (1.05 * 2.0))
        + 0.05 * ((1.0 - 0.6) / 1.0)
        + 0.03 * ((6.0 - 5.0) / 5.0)
        - 0.03 * (1.0 / 3.0)
        - 0.05 * (0.5 * 0.2 + 0.5 * 0.4)
    )
    assert score == pytest.approx(expected)


def test_baseline_metrics_from_row_supplies_all_score_denominators() -> None:
    baseline_row = {
        "straight_completed": False,
        "straight_height_error": 0.4786638352606032,
        "turn_height_error": 0.6332056868032009,
        "loiter_height_error": 1.0466668832112485,
        "random_completion_rate": 0.0,
        "random_height_error": 1.592173326150281,
        "random_progress": 0.979787904362661,
        "canonical_done_t_s": 11.152777777777779,
        "canonical_early_height_drop_m": 0.9953625996907552,
    }
    candidate = {
        "straight_height_error": 0.44717291308120943,
        "turn_height_error": 0.5901693880122943,
        "loiter_height_error": 0.7706635452278296,
        "random_height_error": 1.5061903737301618,
        "random_progress": 0.9799958818606917,
        "canonical_done_t_s": 11.166666666666666,
        "canonical_early_height_drop_m": 0.7984533309936523,
        "canonical_terminated_frac": 0.0,
        "freq_sat_frac": 0.8066744661249797,
        "pitch_sat_frac": 0.0,
    }

    baseline = _baseline_metrics_from_row(baseline_row)
    score = score_candidate(candidate, baseline)

    assert baseline["canonical_done_t_s"] == pytest.approx(11.152777777777779)
    assert baseline["canonical_early_height_drop_m"] == pytest.approx(0.9953625996907552)
    assert score == pytest.approx(0.11947946090709115)


def test_collect_rollout_metrics_includes_early_failure_fields(tmp_path: Path) -> None:
    traj_path = tmp_path / "trajectory_env0.csv"
    traj_path.write_text(
        "\n".join(
            [
                "t,freq_hz,action_elevon_pitch,height_error_m",
                "0.0,4.8,-0.50,-0.10",
                "0.1,4.9,-0.98,-0.25",
                "0.2,5.0,-1.00,-0.40",
                "0.3,4.7,-0.30,-0.20",
            ]
        )
        + "\n"
    )
    summary = {
        "completed_path": False,
        "final_progress_ratio": 0.4,
        "mean_abs_height_error_post_warmup_m": 1.2,
        "mean_abs_lateral_error_post_warmup_m": 2.3,
        "mean_abs_align_error_post_warmup_deg": 11.0,
        "failure_kind": "terminated",
        "steps_completed": 4,
    }

    stats = _collect_rollout_metrics(
        summary=summary,
        traj_path=traj_path,
        freq_sat_threshold_hz=4.9,
        pitch_sat_threshold=-0.98,
        metrics_warmup_s=0.25,
    )

    assert stats["failure_kind"] == "terminated"
    assert stats["steps_completed"] == 4
    assert stats["done_t_s"] == pytest.approx(0.3)
    assert stats["min_height_error_first3s"] == pytest.approx(-0.4)


def test_compute_saturation_fractions_uses_existing_trajectory_fields(tmp_path: Path) -> None:
    traj_path = tmp_path / "trajectory_env0.csv"
    traj_path.write_text(
        "\n".join(
            [
                "t,freq_hz,action_elevon_pitch",
                "0.0,4.8,-0.50",
                "0.1,4.9,-0.98",
                "0.2,5.0,-1.00",
                "0.3,4.7,-0.30",
            ]
        )
        + "\n"
    )

    stats = compute_saturation_fractions(traj_path)

    assert stats["freq_sat_frac"] == pytest.approx(0.5)
    assert stats["pitch_sat_frac"] == pytest.approx(0.5)


def test_build_rollout_command_includes_tail_balance_overrides() -> None:
    args = SimpleNamespace(
        task="Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0",
        num_envs=1,
        metrics_warmup_s=3.0,
        height_sp=10.0,
        straight_length_m=60.0,
        turn_radius_m=20.0,
        loiter_radius_m=20.0,
        turn_sweep_deg=90.0,
        loiter_turns=1.0,
        climb_delta_m=3.0,
        path_manager_max_roll_deg=35.0,
        path_manager_max_flight_path_angle_deg=10.0,
        wind_x_mps=0.0,
        wind_y_mps=0.0,
        wind_ou=False,
        wind_ou_tau_s=2.0,
        wind_ou_sigma_x_mps=0.0,
        wind_ou_sigma_y_mps=0.0,
        wind_ou_clip_to_range=False,
        nominal_speed_mps=7.0,
        completion_margin_s=2.0,
        mission_num_segments_min=2,
        mission_num_segments_max=4,
        mission_allow_straight=True,
        mission_allow_turn=True,
        mission_allow_loiter=True,
        mission_allow_climb_on_straight=True,
        print_every=0,
        headless=True,
    )

    cmd = build_rollout_command(
        args,
        phase="level_turn",
        mission_mode="canonical",
        mission_seed=None,
        mission_label="level_turn",
        steps=2400,
        out_dir=Path("logs/search"),
        tail_fixed_horizontal_effectiveness=0.7,
        tail_elevon_effectiveness=1.5,
    )

    assert "--phase" in cmd
    assert "--tail_fixed_horizontal_effectiveness" in cmd
    assert "--tail_elevon_effectiveness" in cmd
    assert cmd[cmd.index("--tail_fixed_horizontal_effectiveness") + 1] == "0.7"
    assert cmd[cmd.index("--tail_elevon_effectiveness") + 1] == "1.5"
    assert "--headless" in cmd


def test_build_search_parser_has_agreed_defaults() -> None:
    args = build_search_parser().parse_args([])

    assert args.stage1_fixed_values == [0.6, 0.7, 0.8, 0.9, 1.0]
    assert args.stage1_elevon_values == [1.0, 1.2, 1.4, 1.6, 1.8]
    assert args.random_seeds == [11, 23, 37, 53, 71, 89]
    assert args.random_steps == 2600
    assert args.top_k_refine == 4


def test_rank_candidate_rows_orders_feasible_before_infeasible() -> None:
    rows = [
        {"candidate_label": "bad", "feasible": False, "score": float("-inf")},
        {"candidate_label": "good", "feasible": True, "score": 0.2},
        {"candidate_label": "better", "feasible": True, "score": 0.3},
    ]

    ranked = rank_candidate_rows(rows)

    assert [row["candidate_label"] for row in ranked] == ["better", "good", "bad"]
