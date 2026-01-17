#!/usr/bin/env python3
"""Plot and summarize thrust/lift from wind-tunnel CSV.

Input CSV is produced by:
  `scripts/isaac_wind_tunnel_flappingbot_v50.py`

Definitions:
  - Airflow in world is v_air_w = airspeed * flow_dir_world (direction air flows towards).
  - Forward-flight direction is e_forward = normalize(-v_air_w).
  - Thrust is the projection of total force onto e_forward:  T = F_world_T · e_forward.
  - Lift (vertical) is the projection onto world +Z:           Lz = F_world_T · [0,0,1].
  - Lift (relative-to-flow) is the projection onto the component of world +Z orthogonal to e_forward.
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


def _slice_indices(data_len: int, *, start_step: int, end_step: int | None) -> slice:
    if start_step < 0:
        start_step = 0
    if end_step is None or end_step > data_len:
        end_step = data_len
    return slice(start_step, end_step, 1)


def _mean(xs: list[float]) -> float:
    ys = [x for x in xs if not (math.isnan(x) or math.isinf(x))]
    return sum(ys) / float(len(ys)) if ys else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description="Plot & summarize thrust/lift from wind-tunnel CSV.")
    p.add_argument("--csv", type=Path, required=True, help="Input CSV from isaac_wind_tunnel_flappingbot_v50.py.")
    p.add_argument("--out", type=Path, default=None, help="Output PNG path.")
    p.add_argument("--start-step", type=int, default=0, help="Start at this step index.")
    p.add_argument("--end-step", type=int, default=None, help="Stop before this step index (default: end).")
    p.add_argument("--mass-kg", type=float, default=None, help="Optional vehicle mass for mg comparison.")
    p.add_argument(
        "--body-cda",
        type=float,
        default=None,
        help="Optional body drag model using CdA (m^2): D=0.5*rho*CdA*V^2, net_thrust = thrust - D.",
    )
    p.add_argument("--rho", type=float, default=1.225, help="Air density (kg/m^3) used for --body-cda.")
    p.add_argument("--show", action="store_true", help="Show plot window (requires display).")
    args = p.parse_args()

    data = _read_csv(args.csv)
    n = len(data.get("t", []))
    if n == 0:
        raise ValueError("Empty CSV (no 't' rows).")

    sl = _slice_indices(n, start_step=int(args.start_step), end_step=args.end_step)
    t = data["t"][sl]

    def _get_total(name: str) -> list[float]:
        kT = f"{name}_T"
        if kT in data:
            return data[kT][sl]
        # Fallback to sum of L/R if totals not present.
        kL = f"{name}_L"
        kR = f"{name}_R"
        if kL in data and kR in data:
            return [(a + b) for a, b in zip(data[kL], data[kR])][sl]
        raise ValueError(f"Missing columns for total: {kT} or ({kL},{kR})")

    Fx = _get_total("F_world_x")
    Fy = _get_total("F_world_y")
    Fz = _get_total("F_world_z")

    # Forward direction from airflow: e_forward = -normalize(v_air_w).
    if all(k in data for k in ("v_air_wx", "v_air_wy", "v_air_wz")):
        vax = _mean(data["v_air_wx"][sl])
        vay = _mean(data["v_air_wy"][sl])
        vaz = _mean(data["v_air_wz"][sl])
        efx, efy, efz = _unit(-vax, -vay, -vaz)
    else:
        # Default to +X if airflow columns are missing.
        efx, efy, efz = 1.0, 0.0, 0.0

    # "Relative-to-flow lift" axis: project world up onto plane orthogonal to e_forward.
    upx, upy, upz = 0.0, 0.0, 1.0
    proj = _dot(upx, upy, upz, efx, efy, efz)
    elx, ely, elz = upx - proj * efx, upy - proj * efy, upz - proj * efz
    elx, ely, elz = _unit(elx, ely, elz)

    thrust = [(_dot(Fx[i], Fy[i], Fz[i], efx, efy, efz)) for i in range(len(t))]
    lift_z = Fz
    lift_rel = [(_dot(Fx[i], Fy[i], Fz[i], elx, ely, elz)) for i in range(len(t))]

    # Optional quadratic body drag model (vector direction is -e_forward, so scalar net thrust subtracts D).
    V_mean = _mean(data.get("airspeed", [float("nan")])[sl]) if "airspeed" in data else float("nan")
    drag_D = None
    thrust_net = None
    if args.body_cda is not None:
        cda = float(args.body_cda)
        drag_D = 0.5 * float(args.rho) * cda * (float(V_mean) ** 2)
        thrust_net = [v - drag_D for v in thrust]

    print(f"[OK] file: {args.csv}")
    print(f"[OK] e_forward(world)=({efx:+.3f},{efy:+.3f},{efz:+.3f})  thrust = F·e_forward")
    print(f"[OK] mean thrust:   {_mean(thrust):+.6f} N")
    if drag_D is not None and thrust_net is not None:
        print(f"[OK] body drag D:   {drag_D:+.6f} N  (CdA={float(args.body_cda):g} m^2, rho={float(args.rho):g}, V≈{float(V_mean):.3f} m/s)")
        print(f"[OK] mean net_T:   {_mean(thrust_net):+.6f} N  (thrust - D)")
        if float(V_mean) > 1e-6:
            cda_req = (2.0 * _mean(thrust)) / (float(args.rho) * (float(V_mean) ** 2))
            print(f"[OK] CdA to make mean net_T=0: {cda_req:.6f} m^2")
    print(f"[OK] mean lift_z:   {_mean(lift_z):+.6f} N  (world +Z)")
    print(f"[OK] mean lift_rel: {_mean(lift_rel):+.6f} N  (up ⟂ flow)")
    if args.mass_kg is not None:
        mg = float(args.mass_kg) * 9.81
        print(f"[OK] mg:          {mg:+.6f} N  (mass={float(args.mass_kg):g} kg)")

    out = args.out
    if out is None:
        out = Path("outputs") / f"{args.csv.stem}_thrust_lift.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt  # type: ignore

    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    fig.suptitle(f"Thrust/Lift (world): {args.csv}")

    ax[0].plot(t, thrust, label="thrust = F·e_forward")
    if thrust_net is not None:
        ax[0].plot(t, thrust_net, label="net thrust = thrust - D", alpha=0.85)
    ax[0].axhline(_mean(thrust), color="k", linewidth=1.0, alpha=0.3, linestyle="--")
    if thrust_net is not None:
        ax[0].axhline(_mean(thrust_net), color="k", linewidth=1.0, alpha=0.3, linestyle=":")
    ax[0].set_ylabel("N")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(loc="best")

    ax[1].plot(t, lift_z, label="lift_z = Fz (world)")
    ax[1].axhline(_mean(lift_z), color="k", linewidth=1.0, alpha=0.3, linestyle="--")
    if args.mass_kg is not None:
        ax[1].axhline(float(args.mass_kg) * 9.81, color="r", linewidth=1.2, alpha=0.5, linestyle=":")
    ax[1].set_ylabel("N")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend(loc="best")

    ax[2].plot(t, lift_rel, label="lift_rel = F·(up ⟂ flow)")
    ax[2].axhline(_mean(lift_rel), color="k", linewidth=1.0, alpha=0.3, linestyle="--")
    ax[2].set_xlabel("t (s)")
    ax[2].set_ylabel("N")
    ax[2].grid(True, alpha=0.3)
    ax[2].legend(loc="best")

    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"[OK] wrote: {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
