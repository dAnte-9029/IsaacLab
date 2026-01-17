#!/usr/bin/env python3
"""Scan downstroke ratio delta for phase-warped flapping and summarize mean metrics."""

from __future__ import annotations

import argparse
import csv
import math
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from flapping_bot.flapping_bot.physics.phase_warp import PhaseWarp


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
        raise ValueError("Empty delta list.")
    return vals


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


def _max_finite(xs: list[float]) -> float:
    vals = [v for v in xs if _is_finite(v)]
    return max(vals) if vals else float("nan")


def _normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = v
    n = math.sqrt(x * x + y * y + z * z)
    if n <= 1e-12:
        raise ValueError("flow_dir_world must be non-zero.")
    return (x / n, y / n, z / n)


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


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
class Sample:
    delta: float
    f_eff: float
    f_down: float
    f_up: float
    mean_thrust: float
    mean_lift: float
    mean_pin: float
    mean_eta: float
    sep_ratio: float
    peak_power: float
    peak_normal: float
    csv: Path


def _run_once(
    *,
    isaaclab_sh: Path,
    wind_script: Path,
    out_csv: Path,
    steps: int,
    f_hz: float,
    f_ref_hz: float,
    delta: float,
    smoothness: float,
    mode: str,
    body_pitch_deg: float,
    V: float,
    flow_dir_world: tuple[float, float, float],
    wing_geom_csv: Path,
    N: int,
    aero_model: str,
    start_step: int,
    end_step: int | None,
    extra_args: list[str],
    rerun: bool,
) -> Sample:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists() and not rerun:
        pass
    else:
        cmd = [
            str(isaaclab_sh),
            "-p",
            str(wind_script),
            "--headless",
            "--livestream",
            "0",
            "--force-exit",
            "--print-every",
            "0",
            "--steps",
            str(int(steps)),
            "--csv",
            str(out_csv),
            "--f_hz",
            str(float(f_hz)),
            "--aero-model",
            str(aero_model),
            "--body_pitch_deg",
            str(float(body_pitch_deg)),
            "--airspeed",
            str(float(V)),
            "--flow_dir_world",
            str(flow_dir_world[0]),
            str(flow_dir_world[1]),
            str(flow_dir_world[2]),
            "--wing-geom-csv",
            str(wing_geom_csv),
            "--N",
            str(int(N)),
            "--enable-modulation",
            "--downstroke-ratio",
            str(float(delta)),
            "--modulation-mode",
            str(mode),
            "--modulation-smoothness",
            str(float(smoothness)),
            "--modulation-f-ref-hz",
            str(float(f_ref_hz)),
            "--no-auto-geom-from-urdf",
            "--profile-cd0",
            "0.0",
        ]
        if extra_args:
            cmd += list(extra_args)
        subprocess.run(cmd, check=True)

    cols = _load_series(out_csv, start_step=int(start_step), end_step=end_step)
    t = cols.get("t", [])
    Fx = cols.get("F_world_x_T", [])
    Fy = cols.get("F_world_y_T", [])
    Fz = cols.get("F_world_z_T", [])
    pin = cols.get("power_in_T", [])
    aoa_post_L = cols.get("aoa_frac_post_L", [])
    aoa_post_R = cols.get("aoa_frac_post_R", [])
    del_sep_L = cols.get("del_sep_ratio_L", [])
    del_sep_R = cols.get("del_sep_ratio_R", [])
    ny_L = cols.get("F_wang_y_L", [])
    ny_R = cols.get("F_wang_y_R", [])

    if not t:
        raise ValueError(f"CSV missing 't' column or empty after slicing: {out_csv}")

    flow_hat = _normalize(flow_dir_world)
    e_forward = (-flow_hat[0], -flow_hat[1], -flow_hat[2])
    thrust = [_dot((Fx[i], Fy[i], Fz[i]), e_forward) for i in range(len(t))]
    lift = Fz
    use_del = bool(del_sep_L and del_sep_R and any(_is_finite(v) for v in del_sep_L + del_sep_R))
    if use_del:
        sep_ratio = [(del_sep_L[i] + del_sep_R[i]) * 0.5 for i in range(len(t))]
    else:
        sep_ratio = [(aoa_post_L[i] + aoa_post_R[i]) * 0.5 for i in range(len(t))]
    normal_total = [abs(ny_L[i] + ny_R[i]) for i in range(len(t))]

    mean_thrust = _time_mean(t, thrust)
    mean_lift = _time_mean(t, lift)
    mean_pin = _time_mean(t, pin)
    mean_eta = (mean_thrust * float(V) / mean_pin) if mean_pin > 1e-9 else float("nan")
    mean_sep = _time_mean(t, sep_ratio)
    peak_power = _max_finite(pin)
    peak_normal = _max_finite(normal_total)

    warp = PhaseWarp(
        f_hz=float(f_hz),
        delta=float(delta),
        smoothness=float(smoothness),
        mode=str(mode),
        f_ref_hz=float(f_ref_hz),
    )
    return Sample(
        delta=float(delta),
        f_eff=warp.f_eff,
        f_down=warp.f_down,
        f_up=warp.f_up,
        mean_thrust=mean_thrust,
        mean_lift=mean_lift,
        mean_pin=mean_pin,
        mean_eta=mean_eta,
        sep_ratio=mean_sep,
        peak_power=peak_power,
        peak_normal=peak_normal,
        csv=out_csv,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Scan downstroke ratio delta and summarize flapping performance.")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs_DeLaurier/modulation_scan"))
    ap.add_argument("--wind-script", type=Path, default=Path("scripts/isaac_wind_tunnel_flappingbot_v50.py"))
    ap.add_argument(
        "--isaaclab-sh",
        type=Path,
        default=(Path(__file__).resolve().parents[1] / "isaaclab.sh"),
        help="Path to isaaclab.sh",
    )
    ap.add_argument("--delta", type=str, default="0.3:0.7:0.02", help="Delta range list, e.g. 0.3:0.7:0.02.")
    ap.add_argument("--smoothness", type=float, default=0.0, help="Smoothness width (fraction of cycle).")
    ap.add_argument("--mode", type=str, choices=("fixed_f", "fixed_peak_qd"), default="fixed_f")
    ap.add_argument("--aero-model", type=str, choices=("wang2016", "delaurier1993"), default="delaurier1993")
    ap.add_argument("--f-hz", type=float, default=4.5, help="Flapping frequency (Hz) for fixed_f mode.")
    ap.add_argument("--f-ref-hz", type=float, default=4.5, help="Reference frequency for fixed_peak_qd mode (Hz).")
    ap.add_argument("--U", type=float, default=6.9, help="Forward airspeed (m/s).")
    ap.add_argument("--body-pitch-deg", type=float, default=10.0, help="Body pitch (deg).")
    ap.add_argument(
        "--flow-dir-world",
        type=float,
        nargs=3,
        default=(-1.0, 0.0, 0.0),
        metavar=("DX", "DY", "DZ"),
        help="Wind direction in world (unitless). Air flows towards this vector.",
    )
    ap.add_argument("--wing-geom-csv", type=Path, default=Path("outputs/right_wing_te_fit_poly5/right_wing_te_fit_poly5.csv"))
    ap.add_argument("--N", type=int, default=80, help="Spanwise strip count for BEM discretization.")
    ap.add_argument("--steps", type=int, default=2400, help="Simulation steps per run.")
    ap.add_argument("--start-step", type=int, default=240, help="Start step for averaging.")
    ap.add_argument("--end-step", type=int, default=None, help="End step (inclusive) for averaging.")
    ap.add_argument("--extra-args", type=str, default="", help="Extra args forwarded to wind script.")
    ap.add_argument("--rerun", action="store_true", help="Rerun even if CSV exists.")
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "modulation_scan.csv"

    deltas = _parse_range_list(args.delta)
    extra_args = shlex.split(str(args.extra_args)) if str(args.extra_args).strip() else []

    samples: list[Sample] = []
    for delta in deltas:
        tag = f"d{str(delta).replace('.', 'p')}"
        csv_path = out_dir / f"mod_{tag}.csv"
        sample = _run_once(
            isaaclab_sh=args.isaaclab_sh,
            wind_script=args.wind_script,
            out_csv=csv_path,
            steps=int(args.steps),
            f_hz=float(args.f_hz),
            f_ref_hz=float(args.f_ref_hz),
            delta=float(delta),
            smoothness=float(args.smoothness),
            mode=str(args.mode),
            body_pitch_deg=float(args.body_pitch_deg),
            V=float(args.U),
            flow_dir_world=tuple(float(x) for x in args.flow_dir_world),
            wing_geom_csv=args.wing_geom_csv,
            N=int(args.N),
            aero_model=str(args.aero_model),
            start_step=int(args.start_step),
            end_step=args.end_step,
            extra_args=extra_args,
            rerun=bool(args.rerun),
        )
        samples.append(sample)
        print(
            f"[delta {sample.delta:.3f}] T={sample.mean_thrust:+.4f} "
            f"L={sample.mean_lift:+.4f} Pin={sample.mean_pin:+.4f} eta={sample.mean_eta:+.4f}"
        )

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "delta",
                "f_eff_hz",
                "f_down_hz",
                "f_up_hz",
                "mean_thrust_N",
                "mean_lift_N",
                "mean_pin_W",
                "mean_eta",
                "sep_ratio",
                "peak_power_W",
                "peak_normal_N",
                "csv",
            ]
        )
        for s in samples:
            w.writerow(
                [
                    f"{s.delta:.6f}",
                    f"{s.f_eff:.6f}",
                    f"{s.f_down:.6f}",
                    f"{s.f_up:.6f}",
                    f"{s.mean_thrust:.6f}",
                    f"{s.mean_lift:.6f}",
                    f"{s.mean_pin:.6f}",
                    f"{s.mean_eta:.6f}",
                    f"{s.sep_ratio:.6f}",
                    f"{s.peak_power:.6f}",
                    f"{s.peak_normal:.6f}",
                    str(s.csv),
                ]
            )

    try:
        import matplotlib.pyplot as plt  # type: ignore

        xs = [s.delta for s in samples]
        tbar = [s.mean_thrust for s in samples]
        pbar = [s.mean_pin for s in samples]
        eta = [s.mean_eta for s in samples]
        sep = [s.sep_ratio for s in samples]

        def _plot(y: list[float], title: str, ylabel: str, name: str) -> None:
            fig, ax = plt.subplots(1, 1, figsize=(7, 4))
            ax.plot(xs, y, marker="o")
            ax.set_xlabel("downstroke ratio delta")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / name, dpi=160)

        _plot(tbar, "Mean Thrust vs delta", "T_bar (N)", "mod_Tbar.png")
        _plot(pbar, "Mean Input Power vs delta", "P_in_bar (W)", "mod_Pinbar.png")
        _plot(eta, "Propulsive Efficiency vs delta", "eta_bar", "mod_eta.png")
        _plot(sep, "Separated-Flow Ratio vs delta", "sep_ratio", "mod_sep_ratio.png")
        print(f"[OK] summary CSV: {out_csv}")
        print(f"[OK] plots saved to: {out_dir}")
    except Exception as exc:
        print("[WARN] matplotlib unavailable or plot error:", repr(exc))
        print(f"[OK] summary CSV: {out_csv}")


if __name__ == "__main__":
    main()
