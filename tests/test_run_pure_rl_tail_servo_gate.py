from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/flapping_rl/run_pure_rl_tail_servo_gate.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("run_pure_rl_tail_servo_gate", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_tail_servo_gate_builds_four_fresh_backend_mode_workers(tmp_path: Path) -> None:
    gate = _load_module()

    jobs = gate.build_worker_jobs(
        script_path=MODULE_PATH,
        output_dir=tmp_path / "gate",
        cuda_device="cuda:1",
        seed=3,
        headless=True,
        python_executable="python",
    )

    assert tuple((job.mode, job.backend_id) for job in jobs) == (
        ("no_aero", "cpu_native_authority"),
        ("no_aero", "gpu_implicit_candidate"),
        ("aero", "cpu_native_authority"),
        ("aero", "gpu_implicit_candidate"),
    )
    assert jobs[0].device == "cpu"
    assert jobs[1].device == "cuda:1"
    assert all("--worker" in job.command for job in jobs)
    assert all("--headless" in job.command for job in jobs)
    assert all("--seed" in job.command and "3" in job.command for job in jobs)
    assert len({job.output_dir for job in jobs}) == 4


def test_tail_servo_worker_environment_prefers_current_worktree_package(tmp_path: Path) -> None:
    gate = _load_module()
    project_source = tmp_path / "source/flapping_bot"

    environment = gate.worker_subprocess_environment(
        {"PYTHONPATH": "/installed/source", "OTHER": "kept"},
        project_source=project_source,
    )

    assert environment["PYTHONPATH"].split(":") == [
        str(project_source.resolve()),
        "/installed/source",
    ]
    assert environment["OTHER"] == "kept"


@pytest.mark.parametrize(
    ("backend_id", "device", "mode"),
    [
        ("cpu_native_authority", "cpu", "no_aero"),
        ("gpu_implicit_candidate", "cuda:0", "aero"),
    ],
)
def test_tail_servo_worker_request_accepts_only_frozen_contract(
    backend_id: str,
    device: str,
    mode: str,
) -> None:
    gate = _load_module()

    gate.validate_worker_request(
        backend_id=backend_id,
        device=device,
        mode=mode,
        seed=0,
    )
    with pytest.raises(ValueError):
        gate.validate_worker_request(
            backend_id=backend_id,
            device=device,
            mode="free_root",
            seed=0,
        )


def test_tail_servo_command_sequence_is_zero_order_held() -> None:
    gate = _load_module()

    assert gate.tail_level_sign(0.0) == 0.0
    assert gate.tail_level_sign(0.75) == 1.0
    assert gate.tail_level_sign(1.50) == 0.0
    assert gate.tail_level_sign(2.25) == -1.0
    assert gate.tail_level_sign(3.00) == 0.0


def test_tail_servo_tensor_snapshot_does_not_alias_cpu_storage() -> None:
    gate = _load_module()
    source = torch.tensor([1.0, 2.0])

    snapshot = gate.tensor_snapshot(source)
    source.fill_(9.0)

    assert snapshot.tolist() == [1.0, 2.0]


def test_tail_servo_worker_source_locks_backend_and_experiment_contract() -> None:
    text = MODULE_PATH.read_text(encoding="utf-8")

    for expected in (
        "FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg",
        "FlappingBotStraightFlightDeLaurierMeasuredPureRLGpuPhaseMatchedEnvCfg",
        "_PHYSICS_DT_S = 1.0 / 480.0",
        "_POLICY_DECIMATION = 8",
        "_TAIL_AMPLITUDE_DEG = 20.0",
        "_AIRSPEED_MPS = 8.0",
        '_PROJECT_SOURCE = _REPO_ROOT / "source/flapping_bot"',
        "sys.path.insert(0, str(_PROJECT_SOURCE))",
        'tail_aero_deflection_source = "actual_joint_position"',
        'tail_aero_deflection_source) != "actual_joint_position"',
        '"tail_command_rad"',
        '"tail_actual_rad"',
        '"tail_all_actual_rad"',
        '"air_velocity_body_mps"',
        '"wind_world_mps"',
        '"root_linear_velocity_body_mps"',
        '"root_angular_velocity_body_rad_s"',
        '"base_com_position_body_m"',
        '"tail_moment_pre_sim_body_nm"',
        '"tail_moment_after_write_body_nm"',
        '"tail_moment_after_sim_body_nm"',
        '"tail_moment_after_scene_update_body_nm"',
        '"tail_moment_after_recompute_body_nm"',
        '"tail_moment_body_nm"',
        '"tail_recomputed_moment_body_nm"',
    ):
        assert expected in text
