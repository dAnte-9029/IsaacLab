from __future__ import annotations

from datetime import datetime

from scripts.flapping_rl import find_periodic_trim


def _candidate_row() -> dict[str, object]:
    return {
        "rank": 1,
        "candidate_id": 7,
        "pitch_nose_up_deg": 8.0,
        "frequency_target_hz": 4.0,
        "frequency_actual_mean_hz": 4.0001,
        "elevon_pitch_deg": -18.0,
        "rudder_deg": 0.0,
        "elevon_roll_deg": 0.0,
        "action_frequency_normalized": 0.6,
        "action_rudder_normalized": 0.0,
        "action_elevon_pitch_normalized": -18.0 / 41.0,
        "action_elevon_roll_normalized": 0.0,
        "mean_force_residual_b_x_n": 0.01,
        "mean_force_residual_b_y_n": 0.02,
        "mean_force_residual_b_z_n": -0.03,
        "mean_moment_system_com_b_x_nm": 0.001,
        "mean_moment_system_com_b_y_nm": -0.002,
        "mean_moment_system_com_b_z_nm": 0.003,
        "force_score": 0.004,
        "moment_score": 0.001,
        "total_score": 0.005,
        "valid": True,
        "gate_passed": True,
    }


def test_default_parser_describes_the_approved_125_candidate_grid() -> None:
    args = find_periodic_trim.build_parser().parse_args([])

    assert len(args.pitch_deg_values) == 5
    assert len(args.frequency_hz_values) == 5
    assert len(args.elevon_pitch_deg_values) == 5
    assert args.airspeed_mps == 7.0
    assert args.measure_cycles == 8.0
    assert args.top_k == 5
    assert args.score_mode == "longitudinal"


def test_best_trim_artifact_keeps_physical_and_normalized_actions() -> None:
    best = find_periodic_trim._build_best_trim(
        best_row=_candidate_row(),
        free_flight={"height_slope_mps": -0.1},
        airspeed_mps=7.0,
    )

    assert best["candidate_id"] == 7
    assert best["condition"]["airspeed_mps"] == 7.0
    assert best["action"] == {
        "frequency_hz": 4.0,
        "rudder_deg": 0.0,
        "elevon_pitch_deg": -18.0,
        "elevon_roll_deg": 0.0,
    }
    assert best["normalized_action"][0] == 0.6
    assert best["gate_passed"] is True
    assert best["result_type"] == "fixed_root_cycle_mean_wrench_balance_candidate"
    assert best["periodic_orbit_validated"] is False


def test_report_states_fixed_root_and_open_loop_claim_boundaries() -> None:
    row = _candidate_row()
    free_flight = {
        "height_slope_mps": -0.1,
        "body_vx_slope_mps2": 0.2,
        "pitch_slope_deg_s": 0.3,
        "final_body_angular_velocity_rad_s": [0.0, 0.1, 0.0],
    }
    best = find_periodic_trim._build_best_trim(
        best_row=row,
        free_flight=free_flight,
        airspeed_mps=7.0,
    )

    report = find_periodic_trim._render_report(
        ranked_rows=[row],
        best_trim=best,
        manifest={
            "created_at": datetime(2026, 8, 5).isoformat(),
            "search": {"score_mode": "longitudinal"},
        },
    )

    assert "fixed-root" in report
    assert "open-loop drift diagnostic" in report
    assert "not a Poincare periodic-orbit solve" in report
    assert "No controller, teacher, policy, reward or PPO update" in report
