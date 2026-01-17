#!/usr/bin/env python3
"""Check whether wing twist has the expected sign during downstroke.

This script is a small post-processing helper for CSV outputs from:
  `scripts/isaac_wind_tunnel_flappingbot_v50.py`

It does not try to infer "trailing-edge up" from geometry; it only answers:
  - During the chosen downstroke half-cycle (based on wing_qd sign), what are eta_tip_L/R?
  - Are L/R twists consistent (same sign)?
  - Are they mostly positive/negative?
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def _sign(x: float, eps: float = 1e-9) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def _finite(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))


def main() -> None:
    p = argparse.ArgumentParser(description="Check eta_tip sign during downstroke from wind-tunnel CSV.")
    p.add_argument("--csv", type=Path, required=True, help="CSV from isaac_wind_tunnel_flappingbot_v50.py.")
    p.add_argument(
        "--downstroke",
        type=str,
        choices=("qd_neg", "qd_pos"),
        default="qd_neg",
        help="How to define downstroke from wing_qd sign: qd_neg => wing_qd<0, qd_pos => wing_qd>0.",
    )
    p.add_argument("--skip-steps", type=int, default=0, help="Skip this many initial rows (for transients).")
    p.add_argument("--max-rows", type=int, default=20, help="Print first N matched downstroke rows (0 disables).")
    args = p.parse_args()

    required = {"t", "wing_qd", "eta_tip_L", "eta_tip_R"}
    rows: list[tuple[float, float, float, float]] = []

    with args.csv.open("r", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError(f"CSV has no header: {args.csv}")
        missing = required.difference(set(r.fieldnames))
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")
        for i, row in enumerate(r):
            if i < int(args.skip_steps):
                continue
            t = float(row["t"])
            qd = float(row["wing_qd"])
            eL = float(row["eta_tip_L"])
            eR = float(row["eta_tip_R"])
            if not (_finite(t) and _finite(qd) and _finite(eL) and _finite(eR)):
                continue
            is_down = (qd < 0.0) if args.downstroke == "qd_neg" else (qd > 0.0)
            if is_down:
                rows.append((t, qd, eL, eR))

    if not rows:
        print("[WARN] No rows matched the downstroke selector. Try --downstroke qd_pos or reduce --skip-steps.")
        return

    def _mean(xs: list[float]) -> float:
        return sum(xs) / float(len(xs)) if xs else float("nan")

    qds = [qd for _, qd, _, _ in rows]
    eLs = [eL for _, _, eL, _ in rows]
    eRs = [eR for _, _, _, eR in rows]

    same_sign = 0
    opp_sign = 0
    eL_pos = 0
    eL_neg = 0
    eR_pos = 0
    eR_neg = 0
    for _, _, eL, eR in rows:
        sL = _sign(eL)
        sR = _sign(eR)
        if sL != 0 and sR != 0:
            if sL == sR:
                same_sign += 1
            else:
                opp_sign += 1
        if sL > 0:
            eL_pos += 1
        elif sL < 0:
            eL_neg += 1
        if sR > 0:
            eR_pos += 1
        elif sR < 0:
            eR_neg += 1

    print(f"[OK] file: {args.csv}")
    print(f"[OK] downstroke selector: {args.downstroke} (skip_steps={int(args.skip_steps)})")
    print(f"[OK] downstroke samples: {len(rows)}")
    print(f"[OK] mean wing_qd: {_mean(qds):+.6f} rad/s")
    print(f"[OK] mean eta_tip_L: {_mean(eLs):+.6f} rad  (pos={eL_pos}, neg={eL_neg})")
    print(f"[OK] mean eta_tip_R: {_mean(eRs):+.6f} rad  (pos={eR_pos}, neg={eR_neg})")
    if same_sign + opp_sign > 0:
        print(f"[OK] L/R sign consistency (nonzero only): same={same_sign}, opposite={opp_sign}")

    if int(args.max_rows) > 0:
        print("")
        print("t(s)    wing_qd(rad/s)  eta_L(rad)  eta_R(rad)")
        for j, (t, qd, eL, eR) in enumerate(rows[: int(args.max_rows)]):
            print(f"{t:6.3f}  {qd:+13.6f}  {eL:+10.6f}  {eR:+10.6f}")


if __name__ == "__main__":
    main()

