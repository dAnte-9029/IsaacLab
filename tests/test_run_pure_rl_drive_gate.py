from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/flapping_rl/run_pure_rl_drive_gate.py"
SPEC = importlib.util.spec_from_file_location("run_pure_rl_drive_gate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
drive_gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = drive_gate
SPEC.loader.exec_module(drive_gate)


def test_drive_gate_builds_four_fresh_backend_frequency_workers(tmp_path: Path) -> None:
    jobs = drive_gate.build_worker_jobs(
        script_path=MODULE_PATH,
        output_dir=tmp_path,
        cuda_device="cuda:1",
        seed=3,
        headless=True,
    )

    assert tuple((job.backend_id, job.frequency_hz) for job in jobs) == (
        ("cpu_native_authority", 4.0),
        ("cpu_native_authority", 5.0),
        ("gpu_implicit_candidate", 4.0),
        ("gpu_implicit_candidate", 5.0),
    )
    assert all("--worker" in job.command for job in jobs)
    assert all("--headless" in job.command for job in jobs)
    assert all("--seed" in job.command and "3" in job.command for job in jobs)
    assert jobs[0].device == "cpu"
    assert jobs[1].device == "cpu"
    assert jobs[2].device == "cuda:1"
    assert jobs[3].device == "cuda:1"
    assert len({job.output_dir for job in jobs}) == 4


def test_worker_environment_prefers_the_current_worktree_package(tmp_path: Path) -> None:
    project_source = tmp_path / "source/flapping_bot"

    environment = drive_gate.worker_subprocess_environment(
        {"PYTHONPATH": "/installed/source", "OTHER": "kept"},
        project_source=project_source,
    )

    assert environment["PYTHONPATH"].split(":") == [
        str(project_source.resolve()),
        "/installed/source",
    ]
    assert environment["OTHER"] == "kept"


def test_drive_gate_selects_phase_matched_gpu_candidate_explicitly(tmp_path: Path) -> None:
    jobs = drive_gate.build_worker_jobs(
        script_path=MODULE_PATH,
        output_dir=tmp_path,
        cuda_device="cuda:0",
        seed=0,
        headless=True,
        gpu_candidate="phase_matched",
    )

    gpu_jobs = [job for job in jobs if job.backend_id == "gpu_implicit_candidate"]
    assert all("--gpu-candidate" in job.command for job in gpu_jobs)
    assert all("phase_matched" in job.command for job in gpu_jobs)


def test_prescribed_target_is_evaluated_at_post_step_state_time() -> None:
    amplitude_rad = math.radians(29.4)
    assert drive_gate.prescribed_target_position_rad(
        time_s=1.0 / 480.0,
        frequency_hz=4.0,
        amplitude_rad=amplitude_rad,
    ) == pytest.approx(
        amplitude_rad * math.sin(2.0 * math.pi * 4.0 / 480.0)
    )


@pytest.mark.parametrize(
    ("backend_id", "device", "frequency_hz"),
    [
        ("cpu_native_authority", "cpu", 4.0),
        ("gpu_implicit_candidate", "cuda:0", 5.0),
    ],
)
def test_drive_gate_worker_request_accepts_only_frozen_contract(
    backend_id: str,
    device: str,
    frequency_hz: float,
) -> None:
    drive_gate.validate_worker_request(
        backend_id=backend_id,
        device=device,
        frequency_hz=frequency_hz,
        seed=0,
    )

    with pytest.raises(ValueError):
        drive_gate.validate_worker_request(
            backend_id=backend_id,
            device=device,
            frequency_hz=3.0,
            seed=0,
        )
