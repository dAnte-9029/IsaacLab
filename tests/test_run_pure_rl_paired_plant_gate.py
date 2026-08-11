from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/flapping_rl/run_pure_rl_paired_plant_gate.py"
)
SPEC = importlib.util.spec_from_file_location("run_pure_rl_paired_plant_gate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
paired_gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = paired_gate
SPEC.loader.exec_module(paired_gate)


def test_paired_gate_builds_cpu_capture_then_gpu_replay_for_each_stage(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model_1500.pt"
    output_dir = tmp_path / "paired"

    jobs = paired_gate.build_worker_jobs(
        worker_script=tmp_path / "worker.py",
        checkpoint=checkpoint,
        output_dir=output_dir,
        cuda_device="cuda:1",
        seed=0,
        headless=True,
        python_executable="python",
    )

    assert tuple((job.stage_id, job.backend_id) for job in jobs) == (
        ("c1_straight", "cpu_native_authority"),
        ("c1_straight", "gpu_implicit_candidate"),
        ("c2a", "cpu_native_authority"),
        ("c2a", "gpu_implicit_candidate"),
    )
    assert jobs[0].case_ids == ("c1_h0_p0", "c1_h2_p2")
    assert jobs[2].case_ids == (
        "c2a_slope_+00.0_h0_p0_promotion",
        "c2a_slope_-02.0_h0_p0_promotion",
        "c2a_slope_+02.0_h0_p0_promotion",
    )
    assert "--capture-actions" in jobs[0].command
    assert "--capture-actions" in jobs[2].command
    assert "--rollout-mode" in jobs[1].command
    assert "action_replay" in jobs[1].command
    assert str(jobs[0].output_dir / "actions") in jobs[1].command
    assert str(jobs[2].output_dir / "actions") in jobs[3].command
    assert all("--capture-traces" in job.command for job in jobs)
    assert all("--headless" in job.command for job in jobs)
    assert jobs[0].device == "cpu"
    assert jobs[1].device == "cuda:1"


def test_worker_environment_prefers_both_worktree_python_packages(tmp_path: Path) -> None:
    project_source = tmp_path / "source/flapping_bot"
    task_source = tmp_path / "source/isaaclab_tasks"

    environment = paired_gate.worker_subprocess_environment(
        {"PYTHONPATH": "/installed/source", "OTHER": "kept"},
        python_roots=(project_source, task_source),
    )

    assert environment["PYTHONPATH"].split(":") == [
        str(project_source.resolve()),
        str(task_source.resolve()),
        "/installed/source",
    ]
    assert environment["OTHER"] == "kept"


def test_paired_gate_selects_phase_matched_gpu_tasks_explicitly(tmp_path: Path) -> None:
    jobs = paired_gate.build_worker_jobs(
        worker_script=tmp_path / "worker.py",
        checkpoint=tmp_path / "model_1500.pt",
        output_dir=tmp_path / "paired",
        cuda_device="cuda:0",
        seed=0,
        headless=True,
        gpu_candidate="phase_matched",
        python_executable="python",
    )

    gpu_commands = [job.command for job in jobs if job.backend_id == "gpu_implicit_candidate"]
    assert paired_gate.GPU_PHASE_MATCHED_C1_TASK in gpu_commands[0]
    assert paired_gate.GPU_PHASE_MATCHED_C2A_TASK in gpu_commands[1]
