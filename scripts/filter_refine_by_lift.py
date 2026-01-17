#!/usr/bin/env python3
"""Filter auto-sweep refine_summary.csv by lift >= mg.

Example:
  ./isaaclab.sh -p scripts/filter_refine_by_lift.py \
    --refine-csv outputs/auto_sweep_run1/refine_summary.csv \
    --mass-kg 0.6 \
    --out outputs/auto_sweep_run1/refine_lift_ge_mg.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def _f(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description="Filter refine_summary.csv by mean_lift_z_N >= mg.")
    p.add_argument("--refine-csv", type=Path, required=True, help="Input refine_summary.csv from auto_sweep_windtunnel.py.")
    p.add_argument("--mass-kg", type=float, default=None, help="Mass (kg) used to compute mg threshold.")
    p.add_argument("--mg", type=float, default=None, help="Direct mg threshold (N). Overrides --mass-kg.")
    p.add_argument(
        "--margin-n",
        type=float,
        default=0.0,
        help="Extra margin above mg (N). Condition becomes mean_lift_z_N >= mg + margin_n.",
    )
    p.add_argument("--out", type=Path, default=None, help="Optional output CSV path for the filtered rows.")
    p.add_argument("--top", type=int, default=30, help="Print top N rows (sorted by score).")
    args = p.parse_args()

    if args.mg is not None:
        mg = float(args.mg)
    elif args.mass_kg is not None:
        mg = float(args.mass_kg) * 9.81
    else:
        raise SystemExit("Provide either --mg or --mass-kg.")

    rows = list(csv.DictReader(args.refine_csv.open("r", newline="")))
    if not rows:
        raise SystemExit(f"Empty refine CSV: {args.refine_csv}")

    thr = mg + float(args.margin_n)
    kept = []
    for r in rows:
        lift = _f(r.get("mean_lift_z_N", "nan"))
        if math.isnan(lift):
            continue
        if lift >= thr:
            kept.append(r)

    # Sort by score (lower is better in our sweep script).
    kept.sort(key=lambda r: _f(r.get("score", "nan")))

    print(f"[OK] file: {args.refine_csv}")
    print(f"[OK] mg threshold: {mg:.6f} N  (margin={float(args.margin_n):.6f} N, keep if lift >= {thr:.6f} N)")
    print(f"[OK] kept rows: {len(kept)} / {len(rows)}")

    header = ["V_star", "mean_thrust_N", "mean_lift_z_N", "f_hz", "eta_amp_deg", "eta_phase_deg", "body_pitch_deg", "score", "csv"]
    print(",".join(header))
    for r in kept[: max(0, int(args.top))]:
        print(",".join(str(r.get(k, "")) for k in header))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(kept)
        print(f"[OK] wrote: {args.out}")


if __name__ == "__main__":
    main()

