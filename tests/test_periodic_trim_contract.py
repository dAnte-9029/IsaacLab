from __future__ import annotations

import math

import pytest
import torch

from flapping_bot.direct.flapping_bot.periodic_trim import (
    LongitudinalTrimCandidate,
    build_longitudinal_trim_grid,
    build_periodic_trim_grid,
    compute_full_wrench_trim_scores,
    compute_trim_scores,
    rank_trim_rows,
    transfer_moment_to_system_com,
)


def test_build_longitudinal_trim_grid_is_deterministic() -> None:
    candidates = build_longitudinal_trim_grid(
        pitch_nose_up_deg_values=(6.0, 8.0),
        frequency_hz_values=(3.5, 4.0),
        elevon_pitch_deg_values=(-18.0, -12.0),
    )

    assert len(candidates) == 8
    assert candidates[0] == LongitudinalTrimCandidate(
        candidate_id=0,
        pitch_nose_up_deg=6.0,
        frequency_hz=3.5,
        elevon_pitch_deg=-18.0,
    )
    assert candidates[-1].candidate_id == 7
    assert candidates[-1].pitch_nose_up_deg == 8.0
    assert candidates[-1].frequency_hz == 4.0
    assert candidates[-1].elevon_pitch_deg == -12.0


def test_build_periodic_trim_grid_adds_rudder_and_roll_dimensions() -> None:
    candidates = build_periodic_trim_grid(
        pitch_nose_up_deg_values=(12.0,),
        frequency_hz_values=(2.75,),
        elevon_pitch_deg_values=(-8.0,),
        rudder_deg_values=(-5.0, 5.0),
        elevon_roll_deg_values=(-4.0, 0.0, 4.0),
    )

    assert len(candidates) == 6
    assert (candidates[0].rudder_deg, candidates[0].elevon_roll_deg) == (-5.0, -4.0)
    assert (candidates[-1].rudder_deg, candidates[-1].elevon_roll_deg) == (5.0, 4.0)


def test_build_longitudinal_trim_grid_rejects_non_finite_or_non_positive_frequency() -> None:
    with pytest.raises(ValueError, match="frequency_hz"):
        build_longitudinal_trim_grid(
            pitch_nose_up_deg_values=(8.0,),
            frequency_hz_values=(0.0,),
            elevon_pitch_deg_values=(-18.0,),
        )
    with pytest.raises(ValueError, match="finite"):
        build_longitudinal_trim_grid(
            pitch_nose_up_deg_values=(math.nan,),
            frequency_hz_values=(4.0,),
            elevon_pitch_deg_values=(-18.0,),
        )


def test_transfer_moment_to_system_com_uses_consistent_reference() -> None:
    force_b = torch.tensor([[0.0, 0.0, 10.0]])
    moment_about_base_com_b = torch.tensor([[0.0, -2.0, 0.0]])
    system_com_from_base_com_b = torch.tensor([[0.2, 0.0, 0.0]])

    transferred = transfer_moment_to_system_com(
        force_b_n=force_b,
        moment_about_base_com_b_nm=moment_about_base_com_b,
        system_com_from_base_com_b_m=system_com_from_base_com_b,
    )

    assert torch.allclose(transferred, torch.zeros_like(transferred))


def test_compute_trim_scores_uses_longitudinal_force_and_pitch_moment() -> None:
    force_residual_b = torch.tensor(
        [
            [0.3, 100.0, 0.4],
            [0.0, 0.0, 0.0],
        ]
    )
    moment_system_com_b = torch.tensor(
        [
            [100.0, 0.25, 100.0],
            [0.0, 0.0, 0.0],
        ]
    )

    scores = compute_trim_scores(
        mean_force_residual_b_n=force_residual_b,
        mean_moment_system_com_b_nm=moment_system_com_b,
        weight_n=torch.tensor([10.0, 10.0]),
        moment_reference_length_m=0.5,
    )

    assert scores.force_score.tolist() == pytest.approx([0.05, 0.0])
    assert scores.moment_score.tolist() == pytest.approx([0.05, 0.0])
    assert scores.total_score.tolist() == pytest.approx([0.10, 0.0])


def test_compute_full_wrench_scores_all_six_residual_components() -> None:
    scores = compute_full_wrench_trim_scores(
        mean_force_residual_b_n=torch.tensor([[3.0, 4.0, 0.0]]),
        mean_moment_system_com_b_nm=torch.tensor([[0.0, 0.0, 5.0]]),
        weight_n=torch.tensor([10.0]),
        moment_reference_length_m=0.5,
    )

    assert scores.force_score.tolist() == pytest.approx([0.5])
    assert scores.moment_score.tolist() == pytest.approx([1.0])
    assert scores.total_score.tolist() == pytest.approx([1.5])


def test_rank_trim_rows_puts_valid_low_score_candidates_first() -> None:
    rows = [
        {"candidate_id": 0, "valid": True, "total_score": 0.2},
        {"candidate_id": 1, "valid": False, "total_score": 0.01},
        {"candidate_id": 2, "valid": True, "total_score": 0.1},
    ]

    ranked = rank_trim_rows(rows)

    assert [row["candidate_id"] for row in ranked] == [2, 0, 1]
    assert [row["rank"] for row in ranked] == [1, 2, 3]
