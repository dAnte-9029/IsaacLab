#!/usr/bin/env python3
"""Summarize mean thrust/lift over multiple wind-tunnel CSV runs.

This is a lightweight helper for parameter sweeps (phase/amp/pitch), so you can quickly rank runs by
mean thrust and mean lift without opening plots.

Example:
  python scripts/summarize_thrust_lift_sweep.py outputs/sweep_phase_*.csv --start-step 240
"""

from __future__ import annotations

import argparse
import csv as csvlib
import math
from pathlib import Path


def _read_csv(path: Path) -> dict[str, list[float]]:
    try:
        import pandas as pd  # type: ignore

        df = pd.read_csv(path)
        out: dict[str, list[float]] = {}
        for c in df.columns:
            out[str(c)] = df[c].astype(float).tolist()
        return out
    except Exception:
        out: dict[str, list[float]] = {}
        with path.open("r", newline="") as f:
            reader = csvlib.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {path}")
            for name in reader.fieldnames:
                out[name] = []
            for row in reader:
                for name in reader.fieldnames:
                    v = row.get(name, "")
                    try:
                        out[name].append(float(v))
                    except Exception:
                        out[name].append(float("nan"))
        return out


def _unit(vx: float, vy: float, vz: float, eps: float = 1e-9) -> tuple[float, float, float]:
    n = math.sqrt(vx * vx + vy * vy + vz * vz)
    if n < eps:
        return 1.0, 0.0, 0.0
    return vx / n, vy / n, vz / n


def _dot(ax: float, ay: float, az: float, bx: float, by: float, bz: float) -> float:
    return ax * bx + ay * by + az * bz


def _mean(xs: list[float]) -> float:
    ys = [x for x in xs if not (math.isnan(x) or math.isinf(x))]
    return sum(ys) / float(len(ys)) if ys else float("nan")


def _slice_indices(data_len: int, *, start_step: int, end_step: int | None) -> slice:
    if start_step < 0:
        start_step = 0
    if end_step is None or end_step > data_len:
        end_step = data_len
    return slice(start_step, end_step, 1)


def _summarize_one(path: Path, *, start_step: int, end_step: int | None) -> tuple[float, float, float]:
    data = _read_csv(path)
    n = len(data.get("t", []))
    if n == 0:
        raise ValueError(f"Empty CSV (no 't' rows): {path}")
    sl = _slice_indices(n, start_step=start_step, end_step=end_step)

    def _get_total(name: str) -> list[float]:
        kT = f"{name}_T"
        if kT in data:
            return data[kT][sl]
        kL = f"{name}_L"
        kR = f"{name}_R"
        if kL in data and kR in data:
            return [(a + b) for a, b in zip(data[kL], data[kR])][sl]
        raise ValueError(f"Missing columns for total: {kT} or ({kL},{kR})")

    Fx = _get_total("F_world_x")
    Fy = _get_total("F_world_y")
    Fz = _get_total("F_world_z")

    # Forward direction from airflow if available: e_forward = -normalize(v_air_w).
    if all(k in data for k in ("v_air_wx", "v_air_wy", "v_air_wz")):
        vax = _mean(data["v_air_wx"][sl])
        vay = _mean(data["v_air_wy"][sl])
        vaz = _mean(data["v_air_wz"][sl])
        efx, efy, efz = _unit(-vax, -vay, -vaz)
    else:
        efx, efy, efz = 1.0, 0.0, 0.0

    # Relative-to-flow lift axis: project world up onto plane orthogonal to e_forward.
    upx, upy, upz = 0.0, 0.0, 1.0
    proj = _dot(upx, upy, upz, efx, efy, efz)
    elx, ely, elz = _unit(upx - proj * efx, upy - proj * efy, upz - proj * efz)

    thrust = [(_dot(Fx[i], Fy[i], Fz[i], efx, efy, efz)) for i in range(len(Fx))]
    lift_z = Fz
    lift_rel = [(_dot(Fx[i], Fy[i], Fz[i], elx, ely, elz)) for i in range(len(Fx))]
    return _mean(thrust), _mean(lift_z), _mean(lift_rel)


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize mean thrust/lift over multiple wind-tunnel CSVs.")
    p.add_argument("csvs", nargs="+", type=Path, help="Input CSV files (can use shell globs).")
    p.add_argument("--start-step", type=int, default=0, help="Start at this step index.")
    p.add_argument("--end-step", type=int, default=None, help="Stop before this step index (default: end).")
    p.add_argument("--out", type=Path, default=None, help="Optional output CSV summary path.")
    args = p.parse_args()

    rows: list[tuple[str, float, float, float]] = []
    for path in args.csvs:
        tmean, lzmean, lrmean = _summarize_one(path, start_step=int(args.start_step), end_step=args.end_step)
        rows.append((str(path), float(tmean), float(lzmean), float(lrmean)))

    rows_sorted = sorted(rows, key=lambda r: (-(r[1] if not math.isnan(r[1]) else -1e30)))

    print("file,mean_thrust_N,mean_lift_z_N,mean_lift_rel_N")
    for f, tmean, lzmean, lrmean in rows_sorted:
        print(f"{f},{tmean:+.6f},{lzmean:+.6f},{lrmean:+.6f}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            w = csvlib.writer(f)
            w.writerow(["file", "mean_thrust_N", "mean_lift_z_N", "mean_lift_rel_N"])
            for row in rows_sorted:
                w.writerow(list(row))
        print(f"[OK] wrote: {args.out}")


if __name__ == "__main__":
    main()

