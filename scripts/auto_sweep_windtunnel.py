#!/usr/bin/env python3
"""One-shot coarse-to-fine sweep for wind-tunnel flapping QSM runs.

This script does NOT import Isaac Sim. It orchestrates multiple runs by invoking:
  ./isaaclab.sh -p scripts/isaac_wind_tunnel_flappingbot_v50.py ...

Workflow:
  1) Coarse sweep over (f_hz, eta_amp, eta_phase, body_pitch) at a few V samples.
     Keep candidates that bracket thrust sign change (existence of self-propel speed).
  2) For each candidate, refine V* via bisection on mean thrust (T=0), and evaluate lift at V*.
  3) Rank candidates by |L(V*)-mg| (if mass provided) and optionally torque penalty.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import shlex
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _parse_list_or_range(s: str, *, as_int: bool = False) -> list[float]:
    """Parse either 'a,b,c' or 'start:end:step' (end inclusive-ish by stepping)."""
    s = s.strip()
    if ":" in s:
        parts = [p.strip() for p in s.split(":")]
        if len(parts) not in (2, 3):
            raise ValueError(f"Invalid range syntax: {s!r}. Use start:end[:step].")
        start = float(parts[0])
        end = float(parts[1])
        step = float(parts[2]) if len(parts) == 3 else 1.0
        if step == 0:
            raise ValueError(f"Invalid range syntax: step=0 in {s!r}.")
        xs: list[float] = []
        # include end by stepping with tolerance
        x = start
        tol = abs(step) * 1e-6 + 1e-12
        if step > 0:
            while x <= end + tol:
                xs.append(x)
                x += step
        else:
            while x >= end - tol:
                xs.append(x)
                x += step
        return [int(round(v)) if as_int else float(v) for v in xs]
    if not s:
        return []
    xs = [p.strip() for p in s.split(",") if p.strip()]
    out = [float(x) for x in xs]
    return [int(round(v)) if as_int else float(v) for v in out]


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


def _read_csv(path: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
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


def _slice(data: list[float], start_step: int, end_step: int | None) -> list[float]:
    if start_step < 0:
        start_step = 0
    if end_step is None:
        return data[start_step:]
    return data[start_step:end_step]


@dataclass(frozen=True)
class Params:
    f_hz: float
    eta_amp_deg: float
    eta_phase_deg: float
    body_pitch_deg: float


@dataclass
class Eval:
    V: float
    thrust: float
    lift_z: float
    tau_norm: float | None = None
    csv_path: Path | None = None


def _compute_metrics(csv_path: Path, *, start_step: int, end_step: int | None, torque_penalty: bool) -> Eval:
    data = _read_csv(csv_path)
    n = len(data.get("t", []))
    if n == 0:
        raise ValueError(f"Empty CSV (no rows): {csv_path}")

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

    # Forward direction from airflow: e_forward = -normalize(v_air_w).
    if all(k in data for k in ("v_air_wx", "v_air_wy", "v_air_wz")):
        vax = _mean(_slice(data["v_air_wx"], start_step, end_step))
        vay = _mean(_slice(data["v_air_wy"], start_step, end_step))
        vaz = _mean(_slice(data["v_air_wz"], start_step, end_step))
        efx, efy, efz = _unit(-vax, -vay, -vaz)
    else:
        efx, efy, efz = 1.0, 0.0, 0.0

    thrust_series = [(_dot(Fx[i], Fy[i], Fz[i], efx, efy, efz)) for i in range(len(Fx))]
    thrust = _mean(thrust_series)
    lift_z = _mean(Fz)

    tau_norm = None
    if torque_penalty:
        # Prefer torque about base if available.
        keys = (
            ("tau_world_x_T_about_base", "tau_world_y_T_about_base", "tau_world_z_T_about_base"),
            ("tau_world_x_T", "tau_world_y_T", "tau_world_z_T"),
        )
        tx = ty = tz = None
        for kx, ky, kz in keys:
            if kx in data and ky in data and kz in data:
                tx = _mean(_slice(data[kx], start_step, end_step))
                ty = _mean(_slice(data[ky], start_step, end_step))
                tz = _mean(_slice(data[kz], start_step, end_step))
                break
        if tx is not None and ty is not None and tz is not None:
            tau_norm = math.sqrt(tx * tx + ty * ty + tz * tz)

    return Eval(V=float("nan"), thrust=thrust, lift_z=lift_z, tau_norm=tau_norm, csv_path=csv_path)


def _fmt_tag(x: float, *, nd: int = 3) -> str:
    # e.g. 2.0 -> 2p000, -5 -> m5p000
    s = f"{x:+.{nd}f}"
    s = s.replace("+", "").replace("-", "m").replace(".", "p")
    return s


def _run_windtunnel(
    *,
    repo_root: Path,
    wind_script: Path,
    out_csv: Path,
    steps: int,
    V: float,
    params: Params,
    device: str | None,
    sim_device: str | None,
    flow_dir_world: tuple[float, float, float],
    airspeed: float,
    wing_geom_csv: Path,
    dhat: float,
    N: int,
    twist_limit_deg: float,
    twist_sign: float,
    twist_prescribed: str,
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
    cmd += ["--f_hz", str(params.f_hz)]
    cmd += ["--body_pitch_deg", str(params.body_pitch_deg)]
    cmd += ["--airspeed", str(airspeed)]
    cmd += ["--flow_dir_world", str(flow_dir_world[0]), str(flow_dir_world[1]), str(flow_dir_world[2])]
    cmd += ["--wing-geom-csv", str(wing_geom_csv)]
    cmd += ["--dhat", str(dhat)]
    cmd += ["--N", str(int(N))]
    cmd += ["--print-every", "0"]
    cmd += ["--no-auto-geom-from-urdf"]
    cmd += ["--drive-mode", "kinematic"]
    cmd += ["--twist-mode", "prescribed"]
    cmd += ["--twist-prescribed", str(twist_prescribed)]
    cmd += ["--twist-sign", str(twist_sign)]
    cmd += ["--twist-eta-limit-deg", str(twist_limit_deg)]
    cmd += ["--twist-eta-amp-deg", str(params.eta_amp_deg)]
    cmd += ["--twist-eta-phase-deg", str(params.eta_phase_deg)]
    cmd += extra_args

    env = os.environ.copy()
    # Ensure these don't accidentally trigger rendering/streaming modes.
    env.pop("LIVESTREAM", None)
    env.pop("ENABLE_CAMERAS", None)
    env.pop("PUBLIC_IP", None)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # Run each IsaacSim invocation in its own process group so we can terminate the whole tree on timeout.
    proc = subprocess.Popen(cmd, cwd=str(repo_root), env=env, start_new_session=True)
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        # Try a graceful stop first, then a hard kill.
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


def _find_brackets(samples: list[Eval]) -> list[tuple[Eval, Eval]]:
    # assumes samples sorted by V
    brackets: list[tuple[Eval, Eval]] = []
    for a, b in zip(samples, samples[1:]):
        if math.isnan(a.thrust) or math.isnan(b.thrust):
            continue
        if a.thrust == 0.0:
            continue
        if (a.thrust > 0.0 and b.thrust < 0.0) or (a.thrust < 0.0 and b.thrust > 0.0):
            brackets.append((a, b))
    return brackets


def main() -> None:
    p = argparse.ArgumentParser(description="Coarse-to-fine sweep: find self-propel speed then lift balance.")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/auto_sweep"), help="Output directory.")
    p.add_argument(
        "--wind-script",
        type=Path,
        default=Path("scripts/isaac_wind_tunnel_flappingbot_v50.py"),
        help="Wind-tunnel script to run via isaaclab.sh.",
    )
    p.add_argument("--device", type=str, default=None, help="Kit device passed to AppLauncher (e.g. cuda:0, cuda:1).")
    p.add_argument("--sim-device", type=str, default=None, help="PhysX device override (e.g. cpu).")
    p.add_argument("--steps", type=int, default=2400, help="Simulation steps per run.")
    p.add_argument("--start-step", type=int, default=240, help="Discard first N steps when averaging.")
    p.add_argument("--end-step", type=int, default=None, help="Stop before this step (default: end).")
    p.add_argument("--mass-kg", type=float, default=None, help="If provided, use mg in ranking (lift match).")
    p.add_argument("--thrust-eps", type=float, default=0.2, help="Thrust sign-change threshold for coarse bracketing (N).")
    p.add_argument("--lift-min-frac-mg", type=float, default=0.6, help="Coarse filter: require lift >= frac*mg at some V.")
    p.add_argument("--top-k", type=int, default=12, help="How many coarse candidates to refine.")
    p.add_argument("--bisection-iters", type=int, default=7, help="Bisection iterations for V* solve.")
    p.add_argument("--torque-penalty", action="store_true", help="Include mean torque norm in ranking (soft penalty).")
    p.add_argument(
        "--run-timeout-s",
        type=float,
        default=1800.0,
        help="Per-run timeout in seconds (kills the IsaacSim process tree). Use 0 to disable.",
    )
    p.add_argument(
        "--continue-on-failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If a single run fails, record NaNs and continue instead of aborting the whole sweep.",
    )
    p.add_argument(
        "--rerun-existing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, re-run simulations even if the expected per-run CSV already exists in --out-dir.",
    )

    # Fixed / shared args (user-declared invariants).
    p.add_argument("--flow-dir-world", type=float, nargs=3, default=(-1.0, 0.0, 0.0))
    p.add_argument("--wing-geom-csv", type=Path, required=True)
    p.add_argument("--dhat", type=float, default=0.0)
    p.add_argument("--N", type=int, default=80)
    p.add_argument("--twist-prescribed", type=str, default="sin", choices=("sin", "sign_qd", "qd_scaled"))
    p.add_argument("--twist-sign", type=float, default=-1.0)
    p.add_argument("--twist-eta-limit-deg", type=float, default=20.0)
    p.add_argument("--extra-args", type=str, default="", help="Extra args appended to wind script (shell-style string).")

    # Sweep grids (user-declared variables).
    p.add_argument("--f-hz", type=str, default="2:6:1", help="List or range for f_hz, e.g. '2,3,4' or '2:6:1'.")
    p.add_argument("--eta-amp-deg", type=str, default="5,10,15,20", help="List or range for eta_amp (deg).")
    p.add_argument("--eta-phase-deg", type=str, default="0:315:45", help="List or range for eta_phase (deg).")
    p.add_argument("--pitch-deg", type=str, default="5,10,15,20", help="List or range for body_pitch_deg (deg).")
    p.add_argument(
        "--V-samples",
        type=str,
        default="0,3,6,9",
        help="Coarse V samples (m/s) used to bracket thrust sign, e.g. '0,3,6,9' or '0:10:2'.",
    )
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    f_grid = _parse_list_or_range(str(args.f_hz))
    amp_grid = _parse_list_or_range(str(args.eta_amp_deg))
    phase_grid = _parse_list_or_range(str(args.eta_phase_deg))
    pitch_grid = _parse_list_or_range(str(args.pitch_deg))
    V_samples = sorted(_parse_list_or_range(str(args.V_samples)))

    if not (f_grid and amp_grid and phase_grid and pitch_grid and V_samples):
        raise ValueError("Empty sweep grid. Provide non-empty --f-hz/--eta-amp-deg/--eta-phase-deg/--pitch-deg/--V-samples.")

    mg = None
    if args.mass_kg is not None:
        mg = float(args.mass_kg) * 9.81

    extra_args = shlex.split(str(args.extra_args)) if str(args.extra_args).strip() else []

    # Cache: (params, V) -> Eval
    cache: dict[tuple[Params, float], Eval] = {}

    def eval_at(params: Params, V: float) -> Eval:
        key = (params, float(V))
        if key in cache:
            return cache[key]
        tag = (
            f"f{_fmt_tag(params.f_hz)}_amp{_fmt_tag(params.eta_amp_deg)}_ph{_fmt_tag(params.eta_phase_deg)}_"
            f"pitch{_fmt_tag(params.body_pitch_deg)}_V{_fmt_tag(V)}"
        )
        out_csv = out_dir / f"{tag}.csv"
        timeout_s = None if float(args.run_timeout_s) <= 0.0 else float(args.run_timeout_s)
        try:
            if out_csv.exists() and not bool(args.rerun_existing):
                # Reuse previous result (resume capability). If the CSV is corrupt/partial, fall back to re-run.
                try:
                    m = _compute_metrics(
                        out_csv,
                        start_step=int(args.start_step),
                        end_step=args.end_step,
                        torque_penalty=bool(args.torque_penalty),
                    )
                except Exception:
                    _run_windtunnel(
                        repo_root=repo_root,
                        wind_script=args.wind_script,
                        out_csv=out_csv,
                        steps=int(args.steps),
                        V=float(V),
                        params=params,
                        device=args.device,
                        sim_device=args.sim_device,
                        flow_dir_world=(
                            float(args.flow_dir_world[0]),
                            float(args.flow_dir_world[1]),
                            float(args.flow_dir_world[2]),
                        ),
                        airspeed=float(V),
                        wing_geom_csv=args.wing_geom_csv,
                        dhat=float(args.dhat),
                        N=int(args.N),
                        twist_limit_deg=float(args.twist_eta_limit_deg),
                        twist_sign=float(args.twist_sign),
                        twist_prescribed=str(args.twist_prescribed),
                        extra_args=extra_args,
                        timeout_s=timeout_s,
                    )
                    m = _compute_metrics(
                        out_csv,
                        start_step=int(args.start_step),
                        end_step=args.end_step,
                        torque_penalty=bool(args.torque_penalty),
                    )
            else:
                _run_windtunnel(
                    repo_root=repo_root,
                    wind_script=args.wind_script,
                    out_csv=out_csv,
                    steps=int(args.steps),
                    V=float(V),
                    params=params,
                    device=args.device,
                    sim_device=args.sim_device,
                    flow_dir_world=(
                        float(args.flow_dir_world[0]),
                        float(args.flow_dir_world[1]),
                        float(args.flow_dir_world[2]),
                    ),
                    airspeed=float(V),
                    wing_geom_csv=args.wing_geom_csv,
                    dhat=float(args.dhat),
                    N=int(args.N),
                    twist_limit_deg=float(args.twist_eta_limit_deg),
                    twist_sign=float(args.twist_sign),
                    twist_prescribed=str(args.twist_prescribed),
                    extra_args=extra_args,
                    timeout_s=timeout_s,
                )
                m = _compute_metrics(
                    out_csv,
                    start_step=int(args.start_step),
                    end_step=args.end_step,
                    torque_penalty=bool(args.torque_penalty),
                )
            m.V = float(V)
        except Exception as exc:
            if not bool(args.continue_on_failure):
                raise
            print(f"[WARN] run failed: V={V:g} params={params} err={exc}")
            m = Eval(V=float(V), thrust=float('nan'), lift_z=float('nan'), tau_norm=float('nan') if args.torque_penalty else None, csv_path=None)

        cache[key] = m
        return m

    # ---- Phase 1: Coarse sweep
    coarse_rows: list[dict[str, float | str]] = []
    coarse_candidates: list[tuple[Params, tuple[Eval, Eval], float]] = []  # (params, bracket, coarse_score)

    thrust_eps = float(args.thrust_eps)
    for f_hz, eta_amp_deg, eta_phase_deg, body_pitch_deg in itertools.product(f_grid, amp_grid, phase_grid, pitch_grid):
        params = Params(f_hz=float(f_hz), eta_amp_deg=float(eta_amp_deg), eta_phase_deg=float(eta_phase_deg), body_pitch_deg=float(body_pitch_deg))
        samples: list[Eval] = []
        for V in V_samples:
            samples.append(eval_at(params, float(V)))
        samples.sort(key=lambda e: e.V)

        brackets = _find_brackets(samples)
        valid_thrust = [abs(e.thrust) for e in samples if not math.isnan(e.thrust)]
        valid_lift = [e.lift_z for e in samples if not math.isnan(e.lift_z)]
        # If all runs failed for this parameter set, record NaNs and skip.
        if not valid_thrust or not valid_lift:
            coarse_rows.append(
                {
                    "f_hz": params.f_hz,
                    "eta_amp_deg": params.eta_amp_deg,
                    "eta_phase_deg": params.eta_phase_deg,
                    "body_pitch_deg": params.body_pitch_deg,
                    "min_abs_thrust_N": float("nan"),
                    "max_lift_z_N": float("nan"),
                    "has_bracket": 0.0,
                    "lift_ok": 0.0,
                }
            )
            continue
        min_abs_thrust = min(valid_thrust)
        max_lift = max(valid_lift)

        has_bracket = False
        chosen_bracket = None
        for a, b in brackets:
            if abs(a.thrust) >= thrust_eps and abs(b.thrust) >= thrust_eps:
                has_bracket = True
                chosen_bracket = (a, b)
                break
        if not has_bracket and brackets:
            has_bracket = True
            chosen_bracket = brackets[0]

        lift_ok = True
        if mg is not None:
            lift_ok = max_lift >= float(args.lift_min_frac_mg) * mg

        coarse_rows.append(
            {
                "f_hz": params.f_hz,
                "eta_amp_deg": params.eta_amp_deg,
                "eta_phase_deg": params.eta_phase_deg,
                "body_pitch_deg": params.body_pitch_deg,
                "min_abs_thrust_N": float(min_abs_thrust),
                "max_lift_z_N": float(max_lift),
                "has_bracket": 1.0 if has_bracket else 0.0,
                "lift_ok": 1.0 if lift_ok else 0.0,
            }
        )

        if has_bracket and lift_ok and chosen_bracket is not None:
            # Coarse score: prefer closer-to-zero thrust and higher lift (if mass provided).
            score = float(min_abs_thrust)
            if mg is not None:
                # Penalize if even max lift is far below mg.
                score += max(0.0, (mg - max_lift)) * 0.5
            coarse_candidates.append((params, chosen_bracket, score))

    coarse_out = out_dir / "coarse_summary.csv"
    with coarse_out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(coarse_rows[0].keys()))
        w.writeheader()
        w.writerows(coarse_rows)
    print(f"[OK] wrote: {coarse_out}")

    coarse_candidates.sort(key=lambda x: x[2])
    coarse_candidates = coarse_candidates[: max(1, int(args.top_k))]
    print(f"[OK] coarse candidates: {len(coarse_candidates)}")

    # ---- Phase 2: Refine V* for candidates
    refine_rows: list[dict[str, float | str]] = []

    def rank_score(thrust0: float, lift0: float, tau0: float | None) -> float:
        score = abs(thrust0) * 0.5  # should be near 0 by construction
        if mg is not None:
            score += abs(lift0 - mg)
        else:
            score += -lift0 * 0.1  # if no mass, prefer larger lift
        if tau0 is not None:
            score += tau0 * 0.2
        return score

    best = None
    best_score = float("inf")

    for params, (a0, b0), _ in coarse_candidates:
        # Ensure bracket order: a has positive thrust, b negative (swap if needed).
        a = a0
        b = b0
        if a.thrust < 0.0 and b.thrust > 0.0:
            a, b = b, a
        if not (a.thrust > 0.0 and b.thrust < 0.0):
            # If it doesn't strictly bracket, skip.
            continue

        lo = a.V
        hi = b.V
        flo = a.thrust
        fhi = b.thrust

        for _ in range(max(1, int(args.bisection_iters))):
            mid = 0.5 * (lo + hi)
            em = eval_at(params, mid)
            fm = em.thrust
            if fm > 0.0:
                lo, flo = mid, fm
            else:
                hi, fhi = mid, fm

        V_star = 0.5 * (lo + hi)
        e_star = eval_at(params, V_star)
        s = rank_score(e_star.thrust, e_star.lift_z, e_star.tau_norm)
        refine_rows.append(
            {
                "f_hz": params.f_hz,
                "eta_amp_deg": params.eta_amp_deg,
                "eta_phase_deg": params.eta_phase_deg,
                "body_pitch_deg": params.body_pitch_deg,
                "V_star": float(V_star),
                "mean_thrust_N": float(e_star.thrust),
                "mean_lift_z_N": float(e_star.lift_z),
                "mean_tau_norm": float(e_star.tau_norm) if e_star.tau_norm is not None else float("nan"),
                "score": float(s),
                "csv": str(e_star.csv_path) if e_star.csv_path is not None else "",
            }
        )
        if s < best_score:
            best_score = s
            best = (params, V_star, e_star)

    refine_out = out_dir / "refine_summary.csv"
    if refine_rows:
        with refine_out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(refine_rows[0].keys()))
            w.writeheader()
            w.writerows(sorted(refine_rows, key=lambda r: float(r["score"])))  # type: ignore[arg-type]
        print(f"[OK] wrote: {refine_out}")
    else:
        print("[WARN] no refined candidates (check coarse_summary.csv and adjust sweep ranges).")

    if best is None:
        raise SystemExit(2)

    params, V_star, e_star = best
    print("[OK] best:")
    print(f"  f_hz={params.f_hz:g}  eta_amp_deg={params.eta_amp_deg:g}  eta_phase_deg={params.eta_phase_deg:g}  pitch_deg={params.body_pitch_deg:g}")
    print(f"  V*={V_star:.3f} m/s  mean_thrust={e_star.thrust:+.6f} N  mean_lift_z={e_star.lift_z:+.6f} N")
    if mg is not None:
        print(f"  mg={mg:+.6f} N  lift-mg={e_star.lift_z - mg:+.6f} N")
    if e_star.tau_norm is not None:
        print(f"  mean_tau_norm={e_star.tau_norm:+.6f} N·m")
    print(f"  csv={e_star.csv_path}")


if __name__ == "__main__":
    main()
