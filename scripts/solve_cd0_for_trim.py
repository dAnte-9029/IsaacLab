#!/usr/bin/env python3
"""Solve for profile CD0 such that at lift-trim speed V*(CD0), mean thrust is zero.

We solve the nested problem:
  1) For a given CD0, find V* such that mean_lift_z(V,CD0)=mg (bisection in V).
  2) Adjust CD0 so that mean_thrust(V*(CD0),CD0)=0 (bisection in CD0).

This script does NOT import Isaac Sim. It repeatedly invokes:
  ./isaaclab.sh -p scripts/isaac_wind_tunnel_flappingbot_v50.py ...
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shlex
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path


def _unit(vx: float, vy: float, vz: float, eps: float = 1e-9) -> tuple[float, float, float]:
    n = math.sqrt(vx * vx + vy * vy + vz * vz)
    if n < eps:
        return 1.0, 0.0, 0.0
    return vx / n, vy / n, vz / n


def _mean(xs: list[float]) -> float:
    ys = [x for x in xs if not (math.isnan(x) or math.isinf(x))]
    return float(sum(ys) / len(ys)) if ys else float("nan")


def _read_csv_cols(path: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        for k in r.fieldnames:
            out[k] = []
        for row in r:
            for k in r.fieldnames:
                try:
                    out[k].append(float(row.get(k, "")))
                except Exception:
                    out[k].append(float("nan"))
    return out


def _slice(xs: list[float], start_step: int, end_step: int | None) -> list[float]:
    if start_step < 0:
        start_step = 0
    if end_step is None:
        return xs[start_step:]
    return xs[start_step:end_step]


@dataclass(frozen=True)
class Metrics:
    cd0: float
    V: float
    mean_lift_z: float
    mean_thrust: float
    csv_path: Path


def _compute_metrics(csv_path: Path, *, cd0: float, start_step: int, end_step: int | None) -> Metrics:
    data = _read_csv_cols(csv_path)
    if "t" not in data or not data["t"]:
        raise ValueError(f"Empty CSV: {csv_path}")

    def _get_total(name: str) -> list[float]:
        kT = f"{name}_T"
        if kT in data:
            return _slice(data[kT], start_step, end_step)
        kL = f"{name}_L"
        kR = f"{name}_R"
        if kL in data and kR in data:
            return _slice([(a + b) for a, b in zip(data[kL], data[kR])], start_step, end_step)
        raise ValueError(f"Missing columns for total: {kT} or ({kL},{kR})")

    Fx = _get_total("F_world_x")
    Fy = _get_total("F_world_y")
    Fz = _get_total("F_world_z")
    lift_z = _mean(Fz)

    if all(k in data for k in ("v_air_wx", "v_air_wy", "v_air_wz")):
        vax = _mean(_slice(data["v_air_wx"], start_step, end_step))
        vay = _mean(_slice(data["v_air_wy"], start_step, end_step))
        vaz = _mean(_slice(data["v_air_wz"], start_step, end_step))
        efx, efy, efz = _unit(-vax, -vay, -vaz)
    else:
        efx, efy, efz = 1.0, 0.0, 0.0

    thrust_series = [(Fx[i] * efx + Fy[i] * efy + Fz[i] * efz) for i in range(len(Fx))]
    thrust = _mean(thrust_series)

    V = float(_mean(_slice(data.get("airspeed", []), start_step, end_step)))
    return Metrics(cd0=float(cd0), V=V, mean_lift_z=lift_z, mean_thrust=thrust, csv_path=csv_path)


def _fmt_tag(x: float, *, nd: int = 3) -> str:
    s = f"{x:+.{nd}f}"
    return s.replace("+", "").replace("-", "m").replace(".", "p")


def _run_windtunnel(
    *,
    repo_root: Path,
    wind_script: Path,
    out_csv: Path,
    steps: int,
    V: float,
    f_hz: float,
    body_pitch_deg: float,
    flow_dir_world: tuple[float, float, float],
    wing_geom_csv: Path,
    N: int,
    dhat: float,
    device: str | None,
    sim_device: str | None,
    extra_args: list[str],
    timeout_s: float | None,
) -> None:
    cmd: list[str] = ["./isaaclab.sh", "-p", str(wind_script)]
    cmd += ["--headless", "--livestream", "0", "--force-exit"]
    if device is not None:
        cmd += ["--device", device]
    if sim_device is not None:
        cmd += ["--sim-device", sim_device]

    cmd += ["--steps", str(int(steps))]
    cmd += ["--csv", str(out_csv)]
    cmd += ["--f_hz", str(float(f_hz))]
    cmd += ["--body_pitch_deg", str(float(body_pitch_deg))]
    cmd += ["--airspeed", str(float(V))]
    cmd += ["--flow_dir_world", str(flow_dir_world[0]), str(flow_dir_world[1]), str(flow_dir_world[2])]
    cmd += ["--wing-geom-csv", str(wing_geom_csv)]
    cmd += ["--N", str(int(N))]
    cmd += ["--dhat", str(float(dhat))]
    cmd += ["--print-every", "0"]
    cmd += ["--no-auto-geom-from-urdf"]
    cmd += ["--drive-mode", "kinematic"]
    cmd += extra_args

    env = os.environ.copy()
    env.pop("LIVESTREAM", None)
    env.pop("ENABLE_CAMERAS", None)
    env.pop("PUBLIC_IP", None)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(cmd, cwd=str(repo_root), env=env, start_new_session=True)
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            pass
        try:
            proc.wait(timeout=10.0)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
        raise TimeoutError(f"Wind-tunnel run timed out after {timeout_s}s: {out_csv}") from exc
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    if not out_csv.exists():
        raise FileNotFoundError(f"Wind-tunnel run exited cleanly but did not write CSV: {out_csv}")


def main() -> None:
    p = argparse.ArgumentParser(description="Solve CD0 so that at lift-trim V*, mean thrust is zero.")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/cd0_trim"), help="Directory for run CSVs and summary.")
    p.add_argument(
        "--wind-script",
        type=Path,
        default=Path("scripts/isaac_wind_tunnel_flappingbot_v50.py"),
        help="Wind-tunnel script to run via isaaclab.sh.",
    )
    p.add_argument("--device", type=str, default=None, help="Kit device passed to AppLauncher (e.g. cuda:0, cuda:1).")
    p.add_argument("--sim-device", type=str, default=None, help="PhysX device override (e.g. cpu).")
    p.add_argument("--run-timeout-s", type=float, default=1800.0, help="Per-run timeout seconds (0 disables).")
    p.add_argument("--rerun-existing", action=argparse.BooleanOptionalAction, default=False, help="Re-run even if CSV exists.")

    p.add_argument("--steps", type=int, default=2400, help="Simulation steps per run.")
    p.add_argument("--start-step", type=int, default=240, help="Discard first N steps when averaging.")
    p.add_argument("--end-step", type=int, default=None, help="Stop before this step (default: end).")
    p.add_argument("--mass-kg", type=float, required=True, help="Vehicle mass for mg (kg).")

    # Lift trim solve (inner)
    p.add_argument("--V-min", type=float, default=0.0)
    p.add_argument("--V-max", type=float, default=10.0)
    p.add_argument("--V-max-limit", type=float, default=30.0)
    p.add_argument("--V-expand-factor", type=float, default=1.5)
    p.add_argument("--V-iters", type=int, default=8)
    p.add_argument("--V-tol", type=float, default=0.05)

    # CD0 solve (outer)
    p.add_argument("--cd0-min", type=float, default=0.0)
    p.add_argument("--cd0-max", type=float, default=0.15)
    p.add_argument("--cd0-max-limit", type=float, default=1.0)
    p.add_argument("--cd0-expand-factor", type=float, default=1.5)
    p.add_argument("--cd0-iters", type=int, default=6)
    p.add_argument("--cd0-tol", type=float, default=0.005)

    # Invariants / scenario
    p.add_argument("--f-hz", type=float, required=True)
    p.add_argument("--pitch-deg", type=float, required=True)
    p.add_argument("--flow-dir-world", type=float, nargs=3, default=(-1.0, 0.0, 0.0))
    p.add_argument("--wing-geom-csv", type=Path, required=True)
    p.add_argument("--N", type=int, default=80)
    p.add_argument("--dhat", type=float, default=0.0)

    # Profile drag gating
    p.add_argument("--alpha1-deg", type=float, default=15.0)
    p.add_argument("--alpha2-deg", type=float, default=35.0)

    p.add_argument("--extra-args", type=str, default="", help="Extra args appended to wind script (shell-style string).")
    args = p.parse_args()

    if float(args.alpha2_deg) <= float(args.alpha1_deg):
        raise ValueError("--alpha2-deg must be > --alpha1-deg.")

    mg = float(args.mass_kg) * 9.81
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    base_extra = shlex.split(str(args.extra_args)) if str(args.extra_args).strip() else []
    timeout_s = None if float(args.run_timeout_s) <= 0.0 else float(args.run_timeout_s)

    def eval_at(cd0: float, V: float) -> Metrics:
        tag = f"cd0{_fmt_tag(cd0, nd=3)}_V{_fmt_tag(V, nd=3)}"
        out_csv = out_dir / f"{tag}.csv"
        extra = list(base_extra)
        extra += ["--profile-cd0", str(float(cd0)), "--profile-alpha1-deg", str(float(args.alpha1_deg)), "--profile-alpha2-deg", str(float(args.alpha2_deg))]
        if out_csv.exists() and not bool(args.rerun_existing):
            return _compute_metrics(out_csv, cd0=cd0, start_step=int(args.start_step), end_step=args.end_step)
        _run_windtunnel(
            repo_root=repo_root,
            wind_script=args.wind_script,
            out_csv=out_csv,
            steps=int(args.steps),
            V=float(V),
            f_hz=float(args.f_hz),
            body_pitch_deg=float(args.pitch_deg),
            flow_dir_world=(float(args.flow_dir_world[0]), float(args.flow_dir_world[1]), float(args.flow_dir_world[2])),
            wing_geom_csv=args.wing_geom_csv,
            N=int(args.N),
            dhat=float(args.dhat),
            device=args.device,
            sim_device=args.sim_device,
            extra_args=extra,
            timeout_s=timeout_s,
        )
        return _compute_metrics(out_csv, cd0=cd0, start_step=int(args.start_step), end_step=args.end_step)

    def lift_resid(m: Metrics) -> float:
        return float(m.mean_lift_z - mg)

    def solve_V_star(cd0: float) -> Metrics:
        V_lo = float(args.V_min)
        V_hi = float(args.V_max)
        m_lo = eval_at(cd0, V_lo)
        m_hi = eval_at(cd0, V_hi)

        # Expand V_hi if both are below mg.
        while lift_resid(m_lo) < 0.0 and lift_resid(m_hi) < 0.0 and V_hi < float(args.V_max_limit):
            V_hi = min(float(args.V_max_limit), max(V_hi + 0.5, V_hi * float(args.V_expand_factor)))
            m_hi = eval_at(cd0, V_hi)

        if lift_resid(m_lo) * lift_resid(m_hi) > 0.0:
            raise RuntimeError(
                f"Could not bracket lift trim for cd0={cd0:g}. "
                f"g_lo={lift_resid(m_lo):+.3f}, g_hi={lift_resid(m_hi):+.3f}. "
                "Increase --V-max or change kinematics."
            )

        m_star = m_hi
        for _ in range(int(args.V_iters)):
            if abs(V_hi - V_lo) <= float(args.V_tol):
                break
            V_mid = 0.5 * (V_lo + V_hi)
            m_mid = eval_at(cd0, V_mid)
            gm = lift_resid(m_mid)
            if gm == 0.0:
                return m_mid
            if lift_resid(m_lo) * gm <= 0.0:
                V_hi = V_mid
                m_hi = m_mid
                m_star = m_mid
            else:
                V_lo = V_mid
                m_lo = m_mid
                m_star = m_mid
        return m_star

    def thrust_at_lift_trim(cd0: float) -> tuple[float, Metrics]:
        m_star = solve_V_star(cd0)
        return float(m_star.mean_thrust), m_star

    # Bracket CD0.
    cd0_lo = float(args.cd0_min)
    cd0_hi = float(args.cd0_max)
    t_lo, m_lo = thrust_at_lift_trim(cd0_lo)
    t_hi, m_hi = thrust_at_lift_trim(cd0_hi)

    print(f"[OK] mg={mg:.6f} N (mass={float(args.mass_kg):g} kg)")
    print(f"[OK] alpha1/alpha2={float(args.alpha1_deg):g}/{float(args.alpha2_deg):g} deg")
    print(f"[OK] cd0_lo={cd0_lo:g} -> V*={m_lo.V:.6f}  lift={m_lo.mean_lift_z:.6f}  thrust={t_lo:+.6f}")
    print(f"[OK] cd0_hi={cd0_hi:g} -> V*={m_hi.V:.6f}  lift={m_hi.mean_lift_z:.6f}  thrust={t_hi:+.6f}")

    # Expand cd0_hi until sign change or limit.
    while t_lo > 0.0 and t_hi > 0.0 and cd0_hi < float(args.cd0_max_limit):
        cd0_hi = min(float(args.cd0_max_limit), max(cd0_hi + 0.02, cd0_hi * float(args.cd0_expand_factor)))
        t_hi, m_hi = thrust_at_lift_trim(cd0_hi)
        print(f"[OK] expand cd0_hi -> {cd0_hi:g}  V*={m_hi.V:.6f}  thrust={t_hi:+.6f}")

    if t_lo * t_hi > 0.0:
        raise RuntimeError(
            "Could not bracket thrust=0 at lift trim within CD0 limits. "
            f"t_lo={t_lo:+.3f}, t_hi={t_hi:+.3f} (cd0_hi={cd0_hi:g})."
        )

    cd0_star = cd0_hi
    m_star = m_hi
    t_star = t_hi
    for i in range(int(args.cd0_iters)):
        if abs(cd0_hi - cd0_lo) <= float(args.cd0_tol):
            break
        cd0_mid = 0.5 * (cd0_lo + cd0_hi)
        t_mid, m_mid = thrust_at_lift_trim(cd0_mid)
        print(f"[it {i+1:02d}] cd0_mid={cd0_mid:.6f} -> V*={m_mid.V:.6f}  thrust={t_mid:+.6f}")
        if t_mid == 0.0:
            cd0_lo = cd0_hi = cd0_mid
            cd0_star, m_star, t_star = cd0_mid, m_mid, t_mid
            break
        if t_lo * t_mid <= 0.0:
            cd0_hi = cd0_mid
            t_hi = t_mid
            m_hi = m_mid
            cd0_star, m_star, t_star = cd0_mid, m_mid, t_mid
        else:
            cd0_lo = cd0_mid
            t_lo = t_mid
            m_lo = m_mid

    out_summary = out_dir / "cd0_trim_summary.csv"
    with out_summary.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cd0_star", "V_star", "mean_lift_z_N", "mean_thrust_N", "mg_N", "csv"])
        w.writerow([f"{cd0_star:.6f}", f"{m_star.V:.6f}", f"{m_star.mean_lift_z:.6f}", f"{t_star:.6f}", f"{mg:.6f}", str(m_star.csv_path)])

    print(f"[OK] cd0* (net thrust=0 @ lift-trim): {cd0_star:.6f}")
    print(f"[OK] V*(cd0*): {m_star.V:.6f} m/s")
    print(f"[OK] mean lift_z: {m_star.mean_lift_z:.6f} N  (mg={mg:.6f})")
    print(f"[OK] mean thrust: {t_star:+.6f} N")
    print(f"[OK] wrote: {out_summary}")


if __name__ == "__main__":
    main()

