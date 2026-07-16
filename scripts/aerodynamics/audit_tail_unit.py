#!/usr/bin/env python3
"""Run the pure-Python WT0 tail aerodynamic unit audit."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shlex
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "source" / "flapping_bot"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from flapping_bot.analysis.tail_unit_audit import TailAuditSettings, run_tail_unit_audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit the production five-surface tail model without running Isaac Sim.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/aerodynamics/tail_unit_audit"),
        help="Root under which a timestamped SHA-qualified run directory is created.",
    )
    parser.add_argument("--headless", action="store_true", help="Accepted for explicit headless reproduction; plots always use Agg.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nominal-airspeed", type=float, default=8.0)
    parser.add_argument("--density", type=float, default=1.225)
    parser.add_argument(
        "--angle-points",
        type=int,
        default=None,
        help="Override angular sweep resolution for smoke tests; production default is at least 71 points.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = TailAuditSettings(
        seed=int(args.seed),
        nominal_airspeed_mps=float(args.nominal_airspeed),
        air_density_kg_m3=float(args.density),
    )
    if args.angle_points is not None:
        if args.angle_points < 5 or args.angle_points % 2 == 0:
            raise ValueError("--angle-points must be an odd integer >= 5 so zero is sampled.")
        count = int(args.angle_points)
        settings = replace(
            settings,
            symmetric_elevon_sweep_deg=(-35.0, 35.0, count),
            differential_elevon_sweep_deg=(-35.0, 35.0, count),
            rudder_sweep_deg=(-25.0, 25.0, count),
            incidence_sweep_deg=(-20.0, 20.0, count),
            sideslip_sweep_deg=(-20.0, 20.0, count),
            angular_rate_sweep_rad_s=(-2.0, 2.0, count),
            symmetry_elevon_sweep_deg=(-35.0, 35.0, count),
        )
    command = shlex.join(["python", str(Path(__file__).relative_to(REPO_ROOT)), *(argv if argv is not None else sys.argv[1:])])
    output_dir, summary = run_tail_unit_audit(
        repo_root=REPO_ROOT,
        output_root=args.output_root,
        settings=settings,
        run_command=command,
    )
    derivatives = summary["control_derivatives"]
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    print(f"output directory: {output_dir}")
    print(f"git commit: {manifest['git_commit']}")
    print(f"overall status: {summary['overall_status']}")
    print(f"WT1 readiness: {summary['wt1_readiness']}")
    print(
        "key control derivatives: "
        f"dMy/dsym={derivatives['dMy_d_symmetric_Nm_per_rad']:.6g} N m/rad, "
        f"dMx/ddiff={derivatives['dMx_d_differential_Nm_per_rad']:.6g} N m/rad, "
        f"dMz/drud={derivatives['dMz_d_rudder_Nm_per_rad']:.6g} N m/rad"
    )
    print(f"number of warnings: {summary['warning_count']}")
    print(f"number of failed checks: {summary['failed_check_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
