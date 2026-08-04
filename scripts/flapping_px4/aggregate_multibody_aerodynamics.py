"""Aggregate isolated multibody aerodynamic worker results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from flapping_bot.analysis.multibody_aero_validation import (
    compare_fixed_mode_cases,
    summarize_time_step_convergence,
)

_FIXED_CONVERGENCE_SIGNALS = (
    "q_actual_rad",
    "qdot_actual_rad_s",
    "qdd_physx_rad_s2",
    "qdd_aero_input_rad_s2",
    "aero_hinge_power_w",
    "wing_force_b_x_n",
    "wing_force_b_z_n",
    "wing_moment_b_y_nm",
)
_FIXED_COMPARISON_SIGNALS = (
    "q_actual_rad",
    "wing_force_b_x_n",
    "wing_force_b_z_n",
    "wing_moment_b_y_nm",
)


def _summarize_convergence_by_ablation(
    cases: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    ablation_keys = sorted(
        {
            (
                str(case.get("coupling_mode", "")),
                str(case.get("acceleration_source", "")),
                str(case.get("wing_link_load_mode", "")),
            )
            for case in cases
        }
    )
    for coupling_mode, acceleration_source, wing_link_load_mode in ablation_keys:
        group = [
            case
            for case in cases
            if str(case.get("coupling_mode", "")) == coupling_mode
            and str(case.get("acceleration_source", "")) == acceleration_source
            and str(case.get("wing_link_load_mode", "")) == wing_link_load_mode
            and bool(case.get("stable", True))
        ]
        if len({float(case["dt_s"]) for case in group}) < 2:
            continue
        signal_names = [
            signal_name
            for signal_name in _FIXED_CONVERGENCE_SIGNALS
            if all(signal_name in case.get("harmonics", {}) for case in group)
        ]
        for row in summarize_time_step_convergence(
            group,
            signal_names=signal_names,
        ):
            rows.append(
                {
                    **row,
                    "coupling_mode": coupling_mode,
                    "acceleration_source": acceleration_source,
                    "wing_link_load_mode": wing_link_load_mode,
                }
            )
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate one-environment-per-process aerodynamic validation workers."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Directory containing worker subdirectories with summary.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New aggregate output directory.",
    )
    return parser.parse_args()


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = _parse_args()
    summary_paths = sorted(args.input_root.glob("*/summary.json"))
    if not summary_paths:
        raise FileNotFoundError(
            f"No worker summary.json files found under {args.input_root}"
        )
    if args.output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {args.output_dir}"
        )
    workers = [json.loads(path.read_text(encoding="utf-8")) for path in summary_paths]
    if any(worker.get("schema_version") != "physx_multibody_aero_worker_v1" for worker in workers):
        raise ValueError("Every input must use physx_multibody_aero_worker_v1.")
    commits = {
        str(worker.get("provenance", {}).get("git_commit", "")) for worker in workers
    }
    if len(commits) != 1:
        raise ValueError(f"Worker commits do not match: {sorted(commits)}")

    free_cases = [
        case for worker in workers for case in worker.get("free_cases", [])
    ]
    fixed_cases = [
        case for worker in workers for case in worker.get("fixed_cases", [])
    ]
    canonical_fixed_cases = [
        case
        for case in fixed_cases
        if case.get("coupling_mode") == "actual_per_wing_link"
        and bool(case.get("stable", True))
        and float(case["airspeed_mps"]) == 8.0
        and float(case["angle_of_attack_deg"]) == 0.0
    ]
    convergence = _summarize_convergence_by_ablation(canonical_fixed_cases)
    canonical_baseline_cases = [
        case
        for case in fixed_cases
        if case.get("coupling_mode") == "commanded_base_equivalent"
        and bool(case.get("stable", True))
        and float(case["airspeed_mps"]) == 8.0
        and float(case["angle_of_attack_deg"]) == 0.0
    ]
    baseline_convergence = _summarize_convergence_by_ablation(
        canonical_baseline_cases
    )
    reference_dt_s = min(
        (float(case["dt_s"]) for case in fixed_cases),
        default=float("nan"),
    )
    reference_fixed_cases = [
        case
        for case in fixed_cases
        if float(case["dt_s"]) == reference_dt_s
    ]
    fixed_comparison = compare_fixed_mode_cases(
        reference_fixed_cases,
        actual_mode="actual_per_wing_link",
        baseline_mode="commanded_base_equivalent",
        signal_names=_FIXED_COMPARISON_SIGNALS,
    )
    result = {
        "schema_version": "physx_multibody_aero_benchmark_v1",
        "worker_summaries": [str(path) for path in summary_paths],
        "worker_git_commit": next(iter(commits)),
        "free_cases": free_cases,
        "fixed_cases": fixed_cases,
        "time_step_convergence": convergence,
        "baseline_time_step_convergence": baseline_convergence,
        "fixed_comparison_reference_dt_s": reference_dt_s,
        "fixed_mode_comparison": fixed_comparison,
        "validation_gates": {
            "free_case_count": len(free_cases),
            "free_stable_count": sum(
                bool(case.get("stable", True)) for case in free_cases
            ),
            "actual_fixed_case_count": sum(
                case.get("coupling_mode") == "actual_per_wing_link"
                for case in fixed_cases
            ),
            "actual_fixed_stable_count": len(canonical_fixed_cases),
            "baseline_fixed_case_count": sum(
                case.get("coupling_mode") == "commanded_base_equivalent"
                for case in fixed_cases
            ),
            "baseline_fixed_stable_count": len(canonical_baseline_cases),
            "actual_time_step_convergence_available": bool(convergence),
            "actual_stable_count_by_acceleration_source": {
                source: sum(
                    case.get("coupling_mode") == "actual_per_wing_link"
                    and case.get("acceleration_source") == source
                    and bool(case.get("stable", True))
                    for case in fixed_cases
                )
                for source in sorted(
                    {
                        str(case.get("acceleration_source", ""))
                        for case in fixed_cases
                        if case.get("coupling_mode") == "actual_per_wing_link"
                    }
                )
            },
        },
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_rows(
        args.output_dir / "time_step_convergence.csv",
        list(convergence),
    )
    _write_rows(
        args.output_dir / "baseline_time_step_convergence.csv",
        list(baseline_convergence),
    )
    _write_rows(
        args.output_dir / "fixed_mode_comparison.csv",
        list(fixed_comparison),
    )
    print(
        f"Aggregated {len(workers)} workers: "
        f"{len(free_cases)} free cases, {len(fixed_cases)} fixed cases."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
