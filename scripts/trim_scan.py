#!/usr/bin/env python3
"""Grid scan of (f_hz, body_pitch_deg, airspeed) for trim search.

Each run writes to:
  <out_dir>/f{f}_pitch{p}_U{U}/windtunnel_v50.csv

Runs are distributed across the listed GPU devices (one worker per device).
"""

from __future__ import annotations

import argparse
import csv
import math
import shlex
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_range_list(s: str) -> list[float]:
    s = str(s).strip()
    if ":" in s:
        parts = s.split(":")
        if len(parts) != 3:
            raise ValueError("Range must be 'start:stop:step'.")
        start, stop, step = (float(p) for p in parts)
        if step <= 0:
            raise ValueError("Range step must be > 0.")
        out: list[float] = []
        v = start
        while v <= stop + 1e-9:
            out.append(float(v))
            v += step
        return out
    vals = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(float(tok))
    if not vals:
        raise ValueError("Empty list.")
    return vals


def _fmt_tag(x: float, *, nd: int = 3) -> str:
    # 2.5 -> 2p500, -5 -> m5p000
    s = f"{x:+.{nd}f}"
    return s.replace("+", "").replace("-", "m").replace(".", "p")


def _safe_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _is_finite(x: float) -> bool:
    return math.isfinite(x) and not math.isnan(x)


def _time_mean(t: list[float], x: list[float]) -> float:
    if len(t) < 2:
        return float("nan")
    total = 0.0
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        if not _is_finite(dt) or dt <= 0.0:
            continue
        total += 0.5 * (x[i] + x[i + 1]) * dt
    span = t[-1] - t[0]
    return total / span if span > 0.0 else float("nan")


def _load_series(csv_path: Path, *, start_step: int, end_step: int | None) -> dict[str, list[float]]:
    cols: dict[str, list[float]] = {}
    with csv_path.open("r", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        for name in r.fieldnames:
            cols[str(name)] = []
        for row in r:
            step_s = row.get("step", "")
            if step_s == "":
                continue
            step = int(float(step_s))
            if step < start_step:
                continue
            if end_step is not None and step > end_step:
                continue
            for name in cols:
                cols[name].append(_safe_float(row.get(name, "")))
    return cols


@dataclass(frozen=True)
class Metrics:
    mean_fx: float
    mean_fz: float
    mean_pin: float
    mean_sep: float


def _compute_metrics(csv_path: Path, *, start_step: int, end_step: int | None) -> Metrics:
    cols = _load_series(csv_path, start_step=start_step, end_step=end_step)
    t = cols.get("t", [])
    if not t:
        raise ValueError(f"CSV missing 't' column or empty after slicing: {csv_path}")
    fx = cols.get("F_world_x_T", [])
    fz = cols.get("F_world_z_T", [])
    pin = cols.get("power_in_T", [])
    del_sep_L = cols.get("del_sep_ratio_L", [])
    del_sep_R = cols.get("del_sep_ratio_R", [])
    aoa_post_L = cols.get("aoa_frac_post_L", [])
    aoa_post_R = cols.get("aoa_frac_post_R", [])
    use_del = bool(del_sep_L and del_sep_R and any(_is_finite(v) for v in del_sep_L + del_sep_R))
    if use_del:
        sep_ratio = [(del_sep_L[i] + del_sep_R[i]) * 0.5 for i in range(len(t))]
    else:
        sep_ratio = [(aoa_post_L[i] + aoa_post_R[i]) * 0.5 for i in range(len(t))]
    return Metrics(
        mean_fx=_time_mean(t, fx),
        mean_fz=_time_mean(t, fz),
        mean_pin=_time_mean(t, pin),
        mean_sep=_time_mean(t, sep_ratio),
    )


def _run_one(
    *,
    isaaclab_sh: Path,
    wind_script: Path,
    run_dir: Path,
    f_hz: float,
    body_pitch_deg: float,
    airspeed: float,
    steps: int,
    start_step: int,
    end_step: int | None,
    device: str | None,
    sim_device: str | None,
    flow_dir_world: tuple[float, float, float],
    wing_geom_csv: Path,
    N: int,
    aero_model: str,
    extra_args: list[str],
    rerun: bool,
) -> tuple[Path, Metrics]:
    run_dir.mkdir(parents=True, exist_ok=True)
    out_csv = run_dir / "windtunnel_v50.csv"
    if not out_csv.exists() or rerun:
        cmd: list[str] = [str(isaaclab_sh), "-p", str(wind_script)]
        cmd += ["--headless", "--livestream", "0", "--force-exit"]
        if device:
            cmd += ["--device", str(device)]
        if sim_device:
            cmd += ["--sim-device", str(sim_device)]
        cmd += ["--print-every", "0"]
        cmd += ["--steps", str(int(steps))]
        cmd += ["--csv", str(out_csv)]
        cmd += ["--f_hz", str(float(f_hz))]
        cmd += ["--body_pitch_deg", str(float(body_pitch_deg))]
        cmd += ["--airspeed", str(float(airspeed))]
        cmd += ["--flow_dir_world", str(flow_dir_world[0]), str(flow_dir_world[1]), str(flow_dir_world[2])]
        cmd += ["--wing-geom-csv", str(wing_geom_csv)]
        cmd += ["--N", str(int(N))]
        cmd += ["--aero-model", str(aero_model)]
        cmd += ["--no-auto-geom-from-urdf"]
        if extra_args:
            cmd += list(extra_args)
        subprocess.run(cmd, check=True)
    return out_csv, _compute_metrics(out_csv, start_step=start_step, end_step=end_step)


def main() -> None:
    p = argparse.ArgumentParser(description="Grid scan for trim (f_hz, pitch, airspeed).")
    p.add_argument("--out-dir", type=Path, default=Path("outputs_DeLaurier/trim_scan"))
    p.add_argument("--wind-script", type=Path, default=Path("scripts/isaac_wind_tunnel_flappingbot_v50.py"))
    p.add_argument(
        "--isaaclab-sh",
        type=Path,
        default=REPO_ROOT / "isaaclab.sh",
        help="Path to isaaclab.sh",
    )
    p.add_argument("--f-hz", type=str, default="2:4:0.5", help="Frequency grid (Hz).")
    p.add_argument("--pitch-deg", type=str, default="0:15:5", help="Body pitch grid (deg).")
    p.add_argument("--U", type=str, default="5:9:1", help="Airspeed grid (m/s).")
    p.add_argument("--steps", type=int, default=480, help="Simulation steps per run.")
    p.add_argument("--start-step", type=int, default=0, help="Start step for averaging.")
    p.add_argument("--end-step", type=int, default=None, help="End step (inclusive) for averaging.")
    p.add_argument(
        "--devices",
        type=str,
        default="cuda:0,cuda:1",
        help="Comma-separated device list (one worker per device).",
    )
    p.add_argument(
        "--sim-devices",
        type=str,
        default="",
        help="Optional comma-separated sim devices; if empty, reuse --devices.",
    )
    p.add_argument("--flow-dir-world", type=float, nargs=3, default=(-1.0, 0.0, 0.0), metavar=("DX", "DY", "DZ"))
    p.add_argument(
        "--wing-geom-csv",
        type=Path,
        default=Path("outputs_DeLaurier/right_wing_te_fit_poly5_gap50.csv"),
    )
    p.add_argument("--N", type=int, default=80)
    p.add_argument("--aero-model", type=str, choices=("wang2016", "delaurier1993"), default="delaurier1993")
    p.add_argument("--extra-args", type=str, default="", help="Extra args forwarded to wind script.")
    p.add_argument("--rerun", action="store_true", help="Rerun even if CSV exists.")
    p.add_argument("--trim-mass-kg", type=float, default=None, help="If set, use mg as lift target (kg).")
    p.add_argument("--trim-fz-target", type=float, default=None, help="Optional lift target (N). Overrides mass.")
    p.add_argument("--trim-fx-tol", type=float, default=0.5, help="Trim filter: |Fx| <= tol (N).")
    p.add_argument("--trim-fz-tol", type=float, default=1.0, help="Trim filter: |Fz-target| <= tol (N).")
    p.add_argument("--trim-top", type=int, default=20, help="Top-N ranked trim points to save.")
    args = p.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "trim_scan_summary.csv"

    f_grid = _parse_range_list(args.f_hz)
    pitch_grid = _parse_range_list(args.pitch_deg)
    U_grid = _parse_range_list(args.U)
    extra_args = shlex.split(str(args.extra_args)) if str(args.extra_args).strip() else []

    devices = [d.strip() for d in str(args.devices).split(",") if d.strip()]
    if not devices:
        devices = [""]
    sim_devices = [d.strip() for d in str(args.sim_devices).split(",") if d.strip()]
    if sim_devices and len(sim_devices) != len(devices):
        raise ValueError("--sim-devices must match --devices length when provided.")
    if not sim_devices:
        sim_devices = devices

    jobs: list[tuple[float, float, float]] = []
    for f_hz in f_grid:
        for pitch in pitch_grid:
            for U in U_grid:
                jobs.append((f_hz, pitch, U))

    job_groups: list[list[tuple[float, float, float]]] = [[] for _ in devices]
    for idx, job in enumerate(jobs):
        job_groups[idx % len(devices)].append(job)

    lock = threading.Lock()
    results: list[list[float | str]] = []

    def worker(dev: str, sim_dev: str, jobs_local: list[tuple[float, float, float]]) -> None:
        for f_hz, pitch, U in jobs_local:
            tag = f"f{_fmt_tag(f_hz)}_pitch{_fmt_tag(pitch)}_U{_fmt_tag(U)}"
            run_dir = out_dir / tag
            try:
                out_csv, metrics = _run_one(
                    isaaclab_sh=args.isaaclab_sh,
                    wind_script=args.wind_script,
                    run_dir=run_dir,
                    f_hz=f_hz,
                    body_pitch_deg=pitch,
                    airspeed=U,
                    steps=int(args.steps),
                    start_step=int(args.start_step),
                    end_step=args.end_step,
                    device=dev if dev else None,
                    sim_device=sim_dev if sim_dev else None,
                    flow_dir_world=tuple(float(x) for x in args.flow_dir_world),
                    wing_geom_csv=args.wing_geom_csv,
                    N=int(args.N),
                    aero_model=str(args.aero_model),
                    extra_args=extra_args,
                    rerun=bool(args.rerun),
                )
                row = [
                    f_hz,
                    pitch,
                    U,
                    metrics.mean_fx,
                    metrics.mean_fz,
                    metrics.mean_pin,
                    metrics.mean_sep,
                    str(out_csv),
                ]
                with lock:
                    results.append(row)
                print(f"[OK] {tag} Fx={metrics.mean_fx:+.4f} Fz={metrics.mean_fz:+.4f} Pin={metrics.mean_pin:+.4f}")
            except Exception as exc:
                row = [f_hz, pitch, U, float("nan"), float("nan"), float("nan"), float("nan"), ""]
                with lock:
                    results.append(row)
                print(f"[ERR] {tag} {exc}")

    threads = []
    for dev, sim_dev, jobs_local in zip(devices, sim_devices, job_groups):
        t = threading.Thread(target=worker, args=(dev, sim_dev, jobs_local), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    results.sort(key=lambda r: (float(r[0]), float(r[1]), float(r[2])))
    with summary_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["f_hz", "pitch_deg", "airspeed", "mean_Fx", "mean_Fz", "mean_Pin", "mean_sep_ratio", "csv"])
        w.writerows(results)
    print(f"[OK] summary CSV: {summary_csv}")

    # ---- Trim filtering / ranking.
    fz_target = args.trim_fz_target
    if fz_target is None and args.trim_mass_kg is not None:
        fz_target = float(args.trim_mass_kg) * 9.81

    ranked_rows: list[list[float | str]] = []
    candidate_rows: list[list[float | str]] = []
    for row in results:
        f_hz, pitch, U, fx, fz, pin, sep, csv_path = row
        fx_abs = abs(float(fx)) if _is_finite(float(fx)) else float("nan")
        if fz_target is None or not _is_finite(float(fz)):
            fz_err = float("nan")
        else:
            fz_err = abs(float(fz) - float(fz_target))
        if fz_target is None or not _is_finite(float(fz_err)):
            score = fx_abs
        else:
            score = fx_abs + fz_err
        ranked_rows.append(
            [f_hz, pitch, U, fx, fz, pin, sep, fx_abs, fz_target if fz_target is not None else "", fz_err, score, csv_path]
        )
        if _is_finite(fx_abs) and fx_abs <= float(args.trim_fx_tol):
            if fz_target is None or (_is_finite(fz_err) and fz_err <= float(args.trim_fz_tol)):
                candidate_rows.append(
                    [
                        f_hz,
                        pitch,
                        U,
                        fx,
                        fz,
                        pin,
                        sep,
                        fx_abs,
                        fz_target if fz_target is not None else "",
                        fz_err,
                        score,
                        csv_path,
                    ]
                )

    ranked_rows.sort(key=lambda r: float(r[10]) if _is_finite(float(r[10])) else float("inf"))
    top_n = max(int(args.trim_top), 0)
    ranked_csv = out_dir / "trim_scan_ranked.csv"
    with ranked_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "f_hz",
                "pitch_deg",
                "airspeed",
                "mean_Fx",
                "mean_Fz",
                "mean_Pin",
                "mean_sep_ratio",
                "abs_Fx",
                "Fz_target",
                "abs_Fz_err",
                "score",
                "csv",
            ]
        )
        w.writerows(ranked_rows[:top_n] if top_n else ranked_rows)
    print(f"[OK] ranked CSV: {ranked_csv}")

    candidates_csv = out_dir / "trim_scan_candidates.csv"
    with candidates_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "f_hz",
                "pitch_deg",
                "airspeed",
                "mean_Fx",
                "mean_Fz",
                "mean_Pin",
                "mean_sep_ratio",
                "abs_Fx",
                "Fz_target",
                "abs_Fz_err",
                "score",
                "csv",
            ]
        )
        w.writerows(candidate_rows)
    print(f"[OK] candidates CSV: {candidates_csv}")


if __name__ == "__main__":
    main()
