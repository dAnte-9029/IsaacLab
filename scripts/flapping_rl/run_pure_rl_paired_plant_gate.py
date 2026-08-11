"""Run CPU action capture and matched GPU action replay for the paired-plant gate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Mapping, Sequence

import numpy as np


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from pure_rl_backend_diagnostics import compare_paired_plant_traces


CPU_C1_TASK = "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-Direct-v0"
GPU_C1_TASK = "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-GpuImplicit-Direct-v0"
CPU_C2A_TASK = "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C2a-Direct-v0"
GPU_C2A_TASK = "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C2a-GpuImplicit-Direct-v0"
GPU_PHASE_MATCHED_C1_TASK = (
    "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-GpuPhaseMatched-Direct-v0"
)
GPU_PHASE_MATCHED_C2A_TASK = (
    "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C2a-GpuPhaseMatched-Direct-v0"
)
GPU_CANDIDATE_TASKS = {
    "baseline": (GPU_C1_TASK, GPU_C2A_TASK),
    "phase_matched": (GPU_PHASE_MATCHED_C1_TASK, GPU_PHASE_MATCHED_C2A_TASK),
}

C1_CASE_IDS = ("c1_h0_p0", "c1_h2_p2")
C2A_CASE_IDS = (
    "c2a_slope_+00.0_h0_p0_promotion",
    "c2a_slope_-02.0_h0_p0_promotion",
    "c2a_slope_+02.0_h0_p0_promotion",
)


@dataclass(frozen=True)
class WorkerJob:
    """One fresh-process backend evaluation in the paired-plant sequence."""

    stage_id: str
    backend_id: str
    device: str
    case_ids: tuple[str, ...]
    output_dir: Path
    command: tuple[str, ...]


def _worker_command(
    *,
    python_executable: str,
    worker_script: Path,
    task: str,
    backend_id: str,
    checkpoint: Path,
    output_dir: Path,
    device: str,
    case_ids: Sequence[str],
    seed: int,
    action_dir: Path | None,
    headless: bool,
) -> tuple[str, ...]:
    rollout_mode = "closed_loop" if action_dir is None else "action_replay"
    command = [
        python_executable,
        str(worker_script.resolve()),
        "--task",
        task,
        "--backend-id",
        backend_id,
        "--checkpoint",
        str(checkpoint.resolve()),
        "--output-dir",
        str(output_dir),
        "--rollout-mode",
        rollout_mode,
        "--seed",
        str(seed),
        "--device",
        device,
        "--capture-traces",
    ]
    if action_dir is None:
        command.append("--capture-actions")
    else:
        command.extend(("--action-dir", str(action_dir)))
    for case_id in case_ids:
        command.extend(("--case-id", case_id))
    if headless:
        command.append("--headless")
    return tuple(command)


def build_worker_jobs(
    *,
    worker_script: Path,
    checkpoint: Path,
    output_dir: Path,
    cuda_device: str,
    seed: int,
    headless: bool,
    gpu_candidate: str = "baseline",
    python_executable: str | None = None,
) -> tuple[WorkerJob, ...]:
    """Build CPU capture then GPU replay jobs for C1 and C2a."""

    if seed < 0:
        raise ValueError("paired-plant seed must be nonnegative")
    try:
        gpu_c1_task, gpu_c2a_task = GPU_CANDIDATE_TASKS[str(gpu_candidate)]
    except KeyError as error:
        raise ValueError(f"unknown GPU candidate: {gpu_candidate!r}") from error
    executable = python_executable or sys.executable
    jobs: list[WorkerJob] = []
    stage_specs = (
        ("c1_straight", CPU_C1_TASK, gpu_c1_task, C1_CASE_IDS),
        ("c2a", CPU_C2A_TASK, gpu_c2a_task, C2A_CASE_IDS),
    )
    for stage_id, cpu_task, gpu_task, case_ids in stage_specs:
        cpu_output = output_dir / f"{stage_id}_cpu_capture"
        gpu_output = output_dir / f"{stage_id}_gpu_replay"
        jobs.append(
            WorkerJob(
                stage_id=stage_id,
                backend_id="cpu_native_authority",
                device="cpu",
                case_ids=case_ids,
                output_dir=cpu_output,
                command=_worker_command(
                    python_executable=executable,
                    worker_script=worker_script,
                    task=cpu_task,
                    backend_id="cpu_native_authority",
                    checkpoint=checkpoint,
                    output_dir=cpu_output,
                    device="cpu",
                    case_ids=case_ids,
                    seed=seed,
                    action_dir=None,
                    headless=headless,
                ),
            )
        )
        jobs.append(
            WorkerJob(
                stage_id=stage_id,
                backend_id="gpu_implicit_candidate",
                device=str(cuda_device),
                case_ids=case_ids,
                output_dir=gpu_output,
                command=_worker_command(
                    python_executable=executable,
                    worker_script=worker_script,
                    task=gpu_task,
                    backend_id="gpu_implicit_candidate",
                    checkpoint=checkpoint,
                    output_dir=gpu_output,
                    device=str(cuda_device),
                    case_ids=case_ids,
                    seed=seed,
                    action_dir=cpu_output / "actions",
                    headless=headless,
                ),
            )
        )
    return tuple(jobs)


def worker_subprocess_environment(
    base_environment: Mapping[str, str],
    *,
    python_roots: Sequence[Path],
) -> dict[str, str]:
    """Prefer this worktree's project and task packages in child processes."""

    environment = dict(base_environment)
    entries = [str(path.resolve()) for path in python_roots]
    existing = environment.get("PYTHONPATH", "")
    if existing:
        entries.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(entries)
    return environment


def _load_json(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return loaded


def _load_trace(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def compare_worker_artifacts(
    *,
    jobs: Sequence[WorkerJob],
    output_dir: Path,
) -> tuple[dict[str, object], ...]:
    """Compare every CPU trace to the stage-matched GPU replay trace."""

    by_stage_backend = {(job.stage_id, job.backend_id): job for job in jobs}
    rows: list[dict[str, object]] = []
    comparison_dir = output_dir / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    for stage_id, case_ids in (("c1_straight", C1_CASE_IDS), ("c2a", C2A_CASE_IDS)):
        cpu_job = by_stage_backend[(stage_id, "cpu_native_authority")]
        gpu_job = by_stage_backend[(stage_id, "gpu_implicit_candidate")]
        for case_id in case_ids:
            cpu_trace_path = cpu_job.output_dir / "traces" / f"{case_id}.npz"
            gpu_trace_path = gpu_job.output_dir / "traces" / f"{case_id}.npz"
            comparison = compare_paired_plant_traces(
                _load_trace(cpu_trace_path),
                _load_trace(gpu_trace_path),
            )
            row = {"stage_id": stage_id, "case_id": case_id, **comparison}
            rows.append(row)
            (comparison_dir / f"{case_id}.json").write_text(
                json.dumps(row, indent=2) + "\n",
                encoding="utf-8",
            )
    return tuple(rows)


def _run_job(job: WorkerJob, *, environment: Mapping[str, str], output_dir: Path) -> dict[str, object]:
    print(
        f"[paired-plant] starting {job.stage_id} {job.backend_id} on {job.device}",
        flush=True,
    )
    log_path = output_dir / f"{job.stage_id}_{job.backend_id}.log"
    with log_path.open("w", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            job.command,
            cwd=_REPO_ROOT,
            env=dict(environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_stream.write(line)
            log_stream.flush()
            print(line, end="", flush=True)
        return_code = process.wait()
    manifest_path = job.output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Worker produced no manifest: {job.stage_id} {job.backend_id}")
    manifest = _load_json(manifest_path)
    if return_code != 0 or not bool(manifest.get("completed")):
        raise RuntimeError(
            f"Worker failed: {job.stage_id} {job.backend_id}; "
            f"return_code={return_code}; manifest={manifest}"
        )
    print(f"[paired-plant] completed {job.stage_id} {job.backend_id}", flush=True)
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cuda-device", default="cuda:0")
    parser.add_argument(
        "--gpu-candidate",
        choices=tuple(GPU_CANDIDATE_TASKS),
        default="baseline",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    jobs = build_worker_jobs(
        worker_script=_SCRIPT_DIR / "pure_rl_backend_eval_worker.py",
        checkpoint=checkpoint,
        output_dir=output_dir,
        cuda_device=str(args.cuda_device),
        seed=int(args.seed),
        headless=bool(args.headless),
        gpu_candidate=str(args.gpu_candidate),
    )
    environment = worker_subprocess_environment(
        os.environ,
        python_roots=(
            _REPO_ROOT / "source/flapping_bot",
            _REPO_ROOT / "source/isaaclab_tasks",
        ),
    )
    manifests: list[dict[str, object]] = []
    try:
        for job in jobs:
            manifests.append(_run_job(job, environment=environment, output_dir=output_dir))
        comparisons = compare_worker_artifacts(jobs=jobs, output_dir=output_dir)
        gate_passed = all(bool(row["gate_passed"]) for row in comparisons)
        summary = {
            "completed": True,
            "paired_plant_gate_passed": gate_passed,
            "checkpoint": str(checkpoint),
            "seed": int(args.seed),
            "gpu_candidate": str(args.gpu_candidate),
            "case_count": len(comparisons),
            "case_results": list(comparisons),
            "worker_manifests": manifests,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "completed": True,
            "paired_plant_gate_passed": gate_passed,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(checkpoint),
            "seed": int(args.seed),
            "gpu_candidate": str(args.gpu_candidate),
            "case_ids": [row["case_id"] for row in comparisons],
            "summary_path": str(output_dir / "summary.json"),
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(manifest, indent=2), flush=True)
    except BaseException as error:
        failure_manifest = {
            "completed": False,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(checkpoint),
            "seed": int(args.seed),
            "gpu_candidate": str(args.gpu_candidate),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(failure_manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        traceback.print_exception(type(error), error, error.__traceback__, file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
