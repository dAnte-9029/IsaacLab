#!/usr/bin/env python3
"""Summarize AOA (angle-of-attack magnitude) statistics from a wind-tunnel CSV.

Requires the CSV to include columns written by `isaac_wind_tunnel_flappingbot_v50.py`:
  - aoa_mean_deg_{L,R}
  - aoa_max_deg_{L,R}
  - aoa_frac_pre/mid/post_{L,R}
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def _mean(xs: list[float]) -> float:
    ys = [x for x in xs if not (math.isnan(x) or math.isinf(x))]
    return float(sum(ys) / len(ys)) if ys else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize pre/mid/post-stall AOA fractions from wind-tunnel CSV.")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--start-step", type=int, default=240)
    p.add_argument("--end-step", type=int, default=None)
    args = p.parse_args()

    need = [
        "aoa_mean_deg_L",
        "aoa_mean_deg_R",
        "aoa_max_deg_L",
        "aoa_max_deg_R",
        "aoa_frac_pre_L",
        "aoa_frac_pre_R",
        "aoa_frac_mid_L",
        "aoa_frac_mid_R",
        "aoa_frac_post_L",
        "aoa_frac_post_R",
    ]

    cols: dict[str, list[float]] = {k: [] for k in need}
    with args.csv.open("r", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError(f"CSV has no header: {args.csv}")
        names = set(r.fieldnames)
        missing = [k for k in need if k not in names]
        if missing:
            raise ValueError(
                f"CSV is missing AOA columns (re-run wind script after updating it): {missing}\n"
                f"file: {args.csv}"
            )
        for row in r:
            for k in need:
                cols[k].append(float(row[k]))

    n = len(cols[need[0]])
    i0 = max(0, int(args.start_step))
    i1 = int(args.end_step) if args.end_step is not None else n
    i1 = max(i0, min(i1, n))

    def sl(k: str) -> list[float]:
        return cols[k][i0:i1]

    print(f"[OK] file: {args.csv}")
    print(f"[OK] samples used: {i1 - i0}  (steps {i0}..{i1-1})")
    print(f"[OK] mean aoa (deg) L/R: {_mean(sl('aoa_mean_deg_L')):.3f} / {_mean(sl('aoa_mean_deg_R')):.3f}")
    print(f"[OK] max  aoa (deg) L/R: {_mean(sl('aoa_max_deg_L')):.3f} / {_mean(sl('aoa_max_deg_R')):.3f}  (mean of per-step maxima)")
    print(f"[OK] frac_pre  L/R: {_mean(sl('aoa_frac_pre_L')):.3f} / {_mean(sl('aoa_frac_pre_R')):.3f}")
    print(f"[OK] frac_mid  L/R: {_mean(sl('aoa_frac_mid_L')):.3f} / {_mean(sl('aoa_frac_mid_R')):.3f}")
    print(f"[OK] frac_post L/R: {_mean(sl('aoa_frac_post_L')):.3f} / {_mean(sl('aoa_frac_post_R')):.3f}")


if __name__ == "__main__":
    main()
