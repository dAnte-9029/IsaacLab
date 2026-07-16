from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from flapping_bot.analysis.tail_unit_audit import (
    TailAuditSettings,
    WT1_LONGITUDINAL_GATE_CHECKS,
    _classify_audit_gates,
    build_production_tail_cfg,
    mix_elevon_commands,
)
from flapping_bot.physics.tail_aero import TailAeroModel
from flapping_bot.px4_like.straight_line_controller import (
    PX4LikeStraightLineController,
    PX4LikeStraightLineControllerCfg,
)
from scripts.aerodynamics import audit_tail_unit as audit_cli


def _model() -> tuple[TailAeroModel, TailAuditSettings]:
    settings = TailAuditSettings()
    return TailAeroModel(build_production_tail_cfg(settings), "cpu"), settings


def _inputs(batch: int, speed: float = 8.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    velocity = torch.zeros(batch, 3, dtype=torch.float64)
    velocity[:, 0] = speed
    rates = torch.zeros_like(velocity)
    com = torch.tensor(TailAuditSettings().base_com_pos_b_m, dtype=torch.float64).view(1, 3).expand(batch, 3)
    return velocity, rates, com


def _passing_gate_checks() -> dict[str, bool]:
    checks = {name: True for name in WT1_LONGITUDINAL_GATE_CHECKS}
    checks["rudder_lateral_yaw_sign_pair"] = True
    return checks


def test_audit_gate_is_ready_when_all_checks_pass() -> None:
    status = _classify_audit_gates(_passing_gate_checks())

    assert status["overall_status"] == "PASS WITH LIMITATIONS"
    assert status["wt1_readiness"] == "READY WITH DOCUMENTED LIMITATIONS"
    assert status["failed_checks"] == []
    assert status["longitudinal_failed_checks"] == []
    assert status["wt1_nonblocking_failed_checks"] == []


@pytest.mark.parametrize("failed_check", WT1_LONGITUDINAL_GATE_CHECKS)
def test_each_longitudinal_gate_failure_blocks_wt1(failed_check: str) -> None:
    checks = _passing_gate_checks()
    checks[failed_check] = False

    status = _classify_audit_gates(checks)

    assert status["overall_status"] == "FAIL / SUSPECTED BUG"
    assert status["wt1_readiness"] == "NOT READY"
    assert status["failed_checks"] == [failed_check]
    assert status["longitudinal_failed_checks"] == [failed_check]
    assert status["wt1_nonblocking_failed_checks"] == []


def test_lateral_only_failure_is_explicitly_nonblocking_for_wt1() -> None:
    checks = _passing_gate_checks()
    checks["rudder_lateral_yaw_sign_pair"] = False

    status = _classify_audit_gates(checks)

    assert status["overall_status"] == "FAIL / SUSPECTED BUG"
    assert status["wt1_readiness"] == "READY WITH DOCUMENTED LIMITATIONS"
    assert status["failed_checks"] == ["rudder_lateral_yaw_sign_pair"]
    assert status["longitudinal_failed_checks"] == []
    assert status["wt1_nonblocking_failed_checks"] == ["rudder_lateral_yaw_sign_pair"]


def test_missing_longitudinal_gate_definition_fails_closed() -> None:
    checks = _passing_gate_checks()
    del checks["all_outputs_finite"]

    with pytest.raises(ValueError, match="all_outputs_finite"):
        _classify_audit_gates(checks)


@pytest.mark.parametrize(("failed_check_count", "expected"), [(0, 0), (1, 2), (7, 2)])
def test_audit_cli_exit_code_tracks_failed_checks(failed_check_count: int, expected: int) -> None:
    assert audit_cli._audit_exit_code({"failed_check_count": failed_check_count}) == expected


def test_audit_cli_main_returns_two_when_audit_checks_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "failed-audit"
    output.mkdir()
    (output / "manifest.json").write_text(json.dumps({"git_commit": "test-sha"}), encoding="utf-8")
    failed_summary = {
        "overall_status": "FAIL / SUSPECTED BUG",
        "wt1_readiness": "NOT READY",
        "control_derivatives": {
            "dMy_d_symmetric_Nm_per_rad": 0.0,
            "dMx_d_differential_Nm_per_rad": 0.0,
            "dMz_d_rudder_Nm_per_rad": 0.0,
        },
        "warning_count": 0,
        "failed_check_count": 1,
    }

    def fake_run_tail_unit_audit(**_: object) -> tuple[Path, dict[str, object]]:
        return output, failed_summary

    monkeypatch.setattr(audit_cli, "run_tail_unit_audit", fake_run_tail_unit_audit)

    assert audit_cli.main(["--output-root", str(tmp_path), "--headless"]) == 2


def test_surface_results_are_finite_and_reconstruct_moment_and_aggregate() -> None:
    model, _ = _model()
    velocity, rates, com = _inputs(3)
    left = torch.deg2rad(torch.tensor([0.0, 10.0, -12.0], dtype=torch.float64))
    right = torch.deg2rad(torch.tensor([0.0, 10.0, 8.0], dtype=torch.float64))
    rudder = torch.deg2rad(torch.tensor([0.0, 5.0, -7.0], dtype=torch.float64))
    results = model.compute_surface_results(
        root_lin_vel_b=velocity,
        root_ang_vel_b=rates,
        left_elevon_rad=left,
        right_elevon_rad=right,
        rudder_rad=rudder,
        base_com_pos_b=com,
    )
    aggregate_force, aggregate_moment = model.compute_wrench(
        root_lin_vel_b=velocity,
        root_ang_vel_b=rates,
        left_elevon_rad=left,
        right_elevon_rad=right,
        rudder_rad=rudder,
        base_com_pos_b=com,
    )
    assert len(results) == 5
    for result in results:
        assert torch.isfinite(result.force_b).all()
        assert torch.isfinite(result.moment_b_about_com).all()
        independently_reconstructed = torch.linalg.cross(result.aerodynamic_center_b_from_com, result.force_b)
        torch.testing.assert_close(result.moment_b_about_com, independently_reconstructed, atol=1.0e-13, rtol=1.0e-13)
    torch.testing.assert_close(aggregate_force, sum(result.force_b for result in results), atol=1.0e-13, rtol=1.0e-13)
    torch.testing.assert_close(
        aggregate_moment,
        sum(result.moment_b_about_com for result in results),
        atol=1.0e-13,
        rtol=1.0e-13,
    )


def test_zero_input_left_right_elevon_force_symmetry() -> None:
    model, _ = _model()
    velocity, rates, _ = _inputs(1)
    zero = torch.zeros(1, dtype=torch.float64)
    results = model.compute_surface_results(
        root_lin_vel_b=velocity,
        root_ang_vel_b=rates,
        left_elevon_rad=zero,
        right_elevon_rad=zero,
        rudder_rad=zero,
        base_com_pos_b=torch.zeros_like(velocity),
    )
    left, right = results[1], results[2]
    torch.testing.assert_close(left.force_b, right.force_b, atol=1.0e-14, rtol=1.0e-14)
    torch.testing.assert_close(left.local_velocity_b, right.local_velocity_b, atol=1.0e-14, rtol=1.0e-14)
    torch.testing.assert_close(left.alpha_rad, right.alpha_rad, atol=1.0e-14, rtol=1.0e-14)
    parity = torch.tensor([[-1.0, 1.0, -1.0]], dtype=torch.float64)
    torch.testing.assert_close(left.moment_b_about_com, parity * right.moment_b_about_com, atol=1.0e-14, rtol=1.0e-14)


def test_symmetric_elevon_has_pitch_response_without_large_roll_or_yaw() -> None:
    model, _ = _model()
    velocity, rates, com = _inputs(2)
    commands = torch.deg2rad(torch.tensor([-1.0, 1.0], dtype=torch.float64))
    zero = torch.zeros(2, dtype=torch.float64)
    _, moments = model.compute_wrench(
        root_lin_vel_b=velocity,
        root_ang_vel_b=rates,
        left_elevon_rad=commands,
        right_elevon_rad=commands,
        rudder_rad=zero,
        base_com_pos_b=com,
    )
    derivative = (moments[1] - moments[0]) / math.radians(2.0)
    assert derivative[1] > 0.0
    assert abs(float(derivative[0])) < 0.02 * abs(float(derivative[1]))
    assert abs(float(derivative[2])) < 0.02 * abs(float(derivative[1]))


def test_differential_swap_reverses_primary_roll_response() -> None:
    model, _ = _model()
    velocity, rates, com = _inputs(2)
    delta = math.radians(10.0)
    left = torch.tensor([delta, -delta], dtype=torch.float64)
    right = -left
    zero = torch.zeros(2, dtype=torch.float64)
    _, moments = model.compute_wrench(
        root_lin_vel_b=velocity,
        root_ang_vel_b=rates,
        left_elevon_rad=left,
        right_elevon_rad=right,
        rudder_rad=zero,
        base_com_pos_b=com,
    )
    assert moments[0, 0] == pytest.approx(-float(moments[1, 0]), rel=1.0e-12, abs=1.0e-12)
    assert abs(float(moments[0, 0])) > 50.0 * abs(float(moments[0, 1]))


def test_rudder_reversal_reverses_lateral_and_yaw_response() -> None:
    model, _ = _model()
    velocity, rates, com = _inputs(3)
    zero = torch.zeros(3, dtype=torch.float64)
    rudder = torch.deg2rad(torch.tensor([0.0, 10.0, -10.0], dtype=torch.float64))
    force, moment = model.compute_wrench(
        root_lin_vel_b=velocity,
        root_ang_vel_b=rates,
        left_elevon_rad=zero,
        right_elevon_rad=zero,
        rudder_rad=rudder,
        base_com_pos_b=com,
    )
    positive_force_delta = force[1, 1] - force[0, 1]
    negative_force_delta = force[2, 1] - force[0, 1]
    positive_moment_delta = moment[1, 2] - moment[0, 2]
    negative_moment_delta = moment[2, 2] - moment[0, 2]
    assert positive_force_delta < 0.0
    assert positive_moment_delta > 0.0
    assert positive_force_delta == pytest.approx(-float(negative_force_delta), rel=1.0e-12, abs=1.0e-12)
    # The configured COM has a nonzero y offset, so the rudder drag term adds a
    # small even-in-deflection Mz component about COM. The primary lift response
    # must still reverse and remain close to odd.
    assert negative_moment_delta < 0.0
    assert positive_moment_delta == pytest.approx(-float(negative_moment_delta), rel=5.0e-3, abs=1.0e-12)


def test_current_controller_to_rudder_sign_is_restoring() -> None:
    """Freeze the corrected controller-to-tail proportional sign chain."""

    model, settings = _model()
    controller = PX4LikeStraightLineController(
        PX4LikeStraightLineControllerCfg(enable_tecs=False), device=torch.device("cpu")
    )
    actions, diag = controller.compute_actions(
        pos_local=torch.tensor([[0.0, 0.0, 10.0]], dtype=torch.float32),
        ground_vel_local=torch.tensor([[8.0, 0.0, 0.0]], dtype=torch.float32),
        wind_vel_local=torch.zeros((1, 2), dtype=torch.float32),
        roll=torch.zeros(1, dtype=torch.float32),
        pitch=torch.zeros(1, dtype=torch.float32),
        yaw=torch.tensor([math.radians(5.0)], dtype=torch.float32),
        ang_vel_body=torch.zeros((1, 3), dtype=torch.float32),
    )
    course_error = float(diag["course_err"][0])
    action_rudder = float(actions[0, 1])
    assert course_error > 0.0
    assert action_rudder < 0.0
    rudder_rad = math.radians(settings.rudder_limit_deg) * action_rudder

    velocity, rates, com = _inputs(2)
    zero = torch.zeros(2, dtype=torch.float64)
    _, moment = model.compute_wrench(
        root_lin_vel_b=velocity,
        root_ang_vel_b=rates,
        left_elevon_rad=zero,
        right_elevon_rad=zero,
        rudder_rad=torch.tensor([0.0, rudder_rad], dtype=torch.float64),
        base_com_pos_b=com,
    )
    rudder_moment_delta = float(moment[1, 2] - moment[0, 2])
    assert rudder_moment_delta < 0.0
    assert course_error * rudder_moment_delta < 0.0


def test_force_magnitude_increases_with_airspeed_outside_zero_speed_protection() -> None:
    model, _ = _model()
    velocity, rates, com = _inputs(3)
    velocity[:, 0] = torch.tensor([2.0, 4.0, 8.0], dtype=torch.float64)
    command = torch.full((3,), math.radians(5.0), dtype=torch.float64)
    zero = torch.zeros(3, dtype=torch.float64)
    force, _ = model.compute_wrench(
        root_lin_vel_b=velocity,
        root_ang_vel_b=rates,
        left_elevon_rad=command,
        right_elevon_rad=command,
        rudder_rad=zero,
        base_com_pos_b=com,
    )
    norms = torch.linalg.norm(force, dim=1)
    assert norms[0] < norms[1] < norms[2]
    torch.testing.assert_close(norms[1] / norms[0], torch.tensor(4.0, dtype=torch.float64), atol=1.0e-12, rtol=1.0e-12)
    torch.testing.assert_close(norms[2] / norms[1], torch.tensor(4.0, dtype=torch.float64), atol=1.0e-12, rtol=1.0e-12)


def test_environment_elevon_mix_contract() -> None:
    symmetric = torch.tensor([0.2, 0.2])
    differential = torch.tensor([0.1, -0.1])
    left, right = mix_elevon_commands(symmetric, differential)
    torch.testing.assert_close(left, torch.tensor([0.3, 0.1]))
    torch.testing.assert_close(right, torch.tensor([0.1, 0.3]))


def test_diagnostic_refactor_preserves_pre_wt0_production_vectors() -> None:
    model, _ = _model()
    velocity = torch.tensor([[8.0, 0.0, 0.0], [8.0, 1.0, -0.5]], dtype=torch.float64)
    rates = torch.tensor([[0.0, 0.0, 0.0], [0.2, -0.3, 0.4]], dtype=torch.float64)
    left = torch.deg2rad(torch.tensor([10.0, -12.0], dtype=torch.float64))
    right = torch.deg2rad(torch.tensor([10.0, 8.0], dtype=torch.float64))
    rudder = torch.deg2rad(torch.tensor([5.0, -7.0], dtype=torch.float64))
    com = torch.tensor(TailAuditSettings().base_com_pos_b_m, dtype=torch.float64).view(1, 3).expand(2, 3)
    force, moment = model.compute_wrench(
        root_lin_vel_b=velocity,
        root_ang_vel_b=rates,
        left_elevon_rad=left,
        right_elevon_rad=right,
        rudder_rad=rudder,
        base_com_pos_b=com,
    )
    expected_force = torch.tensor(
        [
            [-0.1967604061263759, -0.11610702271825153, 1.3721741967579584],
            [-0.14053703768945344, -0.13623919933175405, 0.7501791043174153],
        ],
        dtype=torch.float64,
    )
    expected_moment = torch.tensor(
        [
            [0.0052219401078892926, 0.5914338624972231, 0.04531497019799332],
            [-0.2065087162520579, 0.2840968662662336, 0.04878668557598242],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(force, expected_force, atol=1.0e-14, rtol=1.0e-14)
    torch.testing.assert_close(moment, expected_moment, atol=1.0e-14, rtol=1.0e-14)


def test_audit_cli_runs_headless_and_writes_required_artifacts(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        str(repo_root / "scripts" / "aerodynamics" / "audit_tail_unit.py"),
        "--output-root",
        str(tmp_path),
        "--headless",
        "--angle-points",
        "9",
    ]
    result = subprocess.run(command, cwd=repo_root, check=True, capture_output=True, text=True)
    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    output = run_dirs[0]
    required = {
        "manifest.json",
        "summary_metrics.json",
        "surface_geometry.csv",
        "nominal_surface_wrench.csv",
        "zero_input_symmetry.csv",
        "symmetric_elevon_sweep.csv",
        "differential_elevon_sweep.csv",
        "rudder_sweep.csv",
        "airspeed_sweep.csv",
        "incidence_sweep.csv",
        "sideslip_sweep.csv",
        "angular_rate_sweep.csv",
        "continuity_checks.csv",
        "report.md",
        "run_command.txt",
    }
    assert required <= {path.name for path in output.iterdir()}
    assert len(list((output / "figures").glob("*.png"))) == 12
    summary = json.loads((output / "summary_metrics.json").read_text(encoding="utf-8"))
    assert summary["checks"]["aggregate_surface_sum"] is True
    assert "overall status:" in result.stdout
