#!/usr/bin/env python3
"""Scan downstroke ratio delta for phase-warped flapping and summarize mean metrics."""

from __future__ import annotations

import argparse
import csv
import math
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

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


def _minmax(vals: list[float]) -> tuple[float, float]:
    finite = [v for v in vals if _is_finite(v)]
    if not finite:
        return (0.0, 1.0)
    vmin = min(finite)
    vmax = max(finite)
    if abs(vmax - vmin) < 1e-9:
        vmin -= 1.0
        vmax += 1.0
    return (vmin, vmax)


def _format_tick(v: float) -> str:
    if not _is_finite(v):
        return ""
    av = abs(v)
    if av >= 1000.0:
        return f"{v:.0f}"
    if av >= 100.0:
        return f"{v:.1f}"
    if av >= 10.0:
        return f"{v:.2f}"
    if av >= 1.0:
        return f"{v:.3f}"
    if av >= 0.1:
        return f"{v:.4f}"
    return f"{v:.3g}"


def _write_svg_line(
    path: Path,
    x: list[float],
    y: list[float],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    color: str = "#2a5caa",
) -> None:
    if len(x) < 2 or len(y) < 2:
        return
    xmin, xmax = _minmax(x)
    ymin, ymax = _minmax(y)
    width = 700.0
    height = 380.0
    margin = 50.0
    tick_size = 5.0
    n_ticks = 5

    def to_px(xv: float, yv: float) -> tuple[float, float]:
        px = margin + (xv - xmin) / (xmax - xmin) * (width - 2 * margin)
        py = margin + (ymax - yv) / (ymax - ymin) * (height - 2 * margin)
        return px, py

    pts = []
    for xv, yv in zip(x, y):
        if not (_is_finite(xv) and _is_finite(yv)):
            continue
        px, py = to_px(xv, yv)
        pts.append(f"{px:.1f},{py:.1f}")
    if not pts:
        return
    pts_str = " ".join(pts)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}">\n')
        f.write('<rect width="100%" height="100%" fill="#ffffff"/>\n')
        f.write(f'<text x="{margin:.0f}" y="{margin - 12:.0f}" font-size="14" '
                f'font-family="sans-serif">{title}</text>\n')
        # Axes.
        f.write(f'<line x1="{margin:.1f}" y1="{height - margin:.1f}" '
                f'x2="{width - margin:.1f}" y2="{height - margin:.1f}" '
                f'stroke="#333" stroke-width="1"/>\n')
        f.write(f'<line x1="{margin:.1f}" y1="{margin:.1f}" '
                f'x2="{margin:.1f}" y2="{height - margin:.1f}" '
                f'stroke="#333" stroke-width="1"/>\n')
        # Ticks and labels.
        for i in range(n_ticks):
            xt = xmin + (xmax - xmin) * i / (n_ticks - 1)
            px, _ = to_px(xt, ymin)
            f.write(
                f'<line x1="{px:.1f}" y1="{height - margin:.1f}" '
                f'x2="{px:.1f}" y2="{height - margin + tick_size:.1f}" '
                f'stroke="#333" stroke-width="1"/>\n'
            )
            f.write(
                f'<text x="{px:.1f}" y="{height - margin + 18:.1f}" font-size="10" '
                f'font-family="sans-serif" text-anchor="middle">{_format_tick(xt)}</text>\n'
            )
        for i in range(n_ticks):
            yt = ymin + (ymax - ymin) * i / (n_ticks - 1)
            _, py = to_px(xmin, yt)
            f.write(
                f'<line x1="{margin - tick_size:.1f}" y1="{py:.1f}" '
                f'x2="{margin:.1f}" y2="{py:.1f}" '
                f'stroke="#333" stroke-width="1"/>\n'
            )
            f.write(
                f'<text x="{margin - tick_size - 2:.1f}" y="{py + 3:.1f}" font-size="10" '
                f'font-family="sans-serif" text-anchor="end">{_format_tick(yt)}</text>\n'
            )
        # Line.
        f.write(f'<polyline points="{pts_str}" fill="none" stroke="{color}" stroke-width="1.5"/>\n')
        # Labels.
        f.write(f'<text x="{width / 2:.1f}" y="{height - 8:.1f}" font-size="11" '
                f'font-family="sans-serif">{xlabel}</text>\n')
        f.write(f'<text x="{8:.1f}" y="{height / 2:.1f}" font-size="11" '
                f'font-family="sans-serif" transform="rotate(-90 8,{height / 2:.1f})">{ylabel}</text>\n')
        f.write("</svg>\n")


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
    amp: float
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
    mean_nc: float
    mean_na: float
    mean_fx_suction: float
    mean_fx_camber: float
    mean_fx_friction: float
    mean_fx_total: float
    mean_k: float
    mean_alpha_prime: float
    mean_alpha_le: float
    share_na: float
    share_fx_suction: float
    share_fx_camber: float
    share_fx_friction: float
    csv: Path


def _run_once(
    *,
    isaaclab_sh: Path,
    wind_script: Path,
    out_csv: Path,
    steps: int,
    f_hz: float,
    f_ref_hz: float,
    amp: float,
    delta: float,
    smoothness: float,
    mode: str,
    profile: str,
    ff_amp: float,
    ff_shift_deg: float,
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
    plot_per_run: bool,
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
            "--modulation-profile",
            str(profile),
            "--downstroke-ratio",
            str(float(delta)),
            "--modulation-mode",
            str(mode),
            "--modulation-smoothness",
            str(float(smoothness)),
            "--modulation-f-ref-hz",
            str(float(f_ref_hz)),
            "--modulation-ff-amp",
            str(float(ff_amp)),
            "--modulation-ff-shift-deg",
            str(float(ff_shift_deg)),
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

    # Thrust is reported along +X world (user convention).
    thrust = Fx
    lift = Fz
    use_del = bool(del_sep_L and del_sep_R and any(_is_finite(v) for v in del_sep_L + del_sep_R))
    if use_del:
        sep_ratio = [(del_sep_L[i] + del_sep_R[i]) * 0.5 for i in range(len(t))]
    else:
        sep_ratio = [(aoa_post_L[i] + aoa_post_R[i]) * 0.5 for i in range(len(t))]
    normal_total = [abs(ny_L[i] + ny_R[i]) for i in range(len(t))]

    def _sum_lr(name_l: str, name_r: str) -> list[float]:
        if name_l in cols and name_r in cols:
            return [cols[name_l][i] + cols[name_r][i] for i in range(len(t))]
        return [float("nan")] * len(t)

    def _mean_lr(name_l: str, name_r: str) -> list[float]:
        if name_l in cols and name_r in cols:
            return [(cols[name_l][i] + cols[name_r][i]) * 0.5 for i in range(len(t))]
        return [float("nan")] * len(t)

    mean_thrust = _time_mean(t, thrust)
    mean_lift = _time_mean(t, lift)
    mean_pin = _time_mean(t, pin)
    mean_eta = (mean_thrust * float(V) / mean_pin) if mean_pin > 1e-9 else float("nan")
    mean_sep = _time_mean(t, sep_ratio)
    peak_power = _max_finite(pin)
    peak_normal = _max_finite(normal_total)

    del_nc = _sum_lr("del_Nc_L", "del_Nc_R")
    del_na = _sum_lr("del_Na_L", "del_Na_R")
    del_fx_suction = _sum_lr("del_Fx_suction_L", "del_Fx_suction_R")
    del_fx_camber = _sum_lr("del_Fx_camber_L", "del_Fx_camber_R")
    del_fx_friction = _sum_lr("del_Fx_friction_L", "del_Fx_friction_R")
    del_fx_total = _sum_lr("del_Fx_total_L", "del_Fx_total_R")
    del_k = _mean_lr("del_k_mean_L", "del_k_mean_R")
    del_alpha_prime = _mean_lr("del_alpha_prime_mean_L", "del_alpha_prime_mean_R")
    del_alpha_le = _mean_lr("del_alpha_le_mean_L", "del_alpha_le_mean_R")

    mean_nc = _time_mean(t, del_nc)
    mean_na = _time_mean(t, del_na)
    mean_fx_suction = _time_mean(t, del_fx_suction)
    mean_fx_camber = _time_mean(t, del_fx_camber)
    mean_fx_friction = _time_mean(t, del_fx_friction)
    mean_fx_total = _time_mean(t, del_fx_total)
    mean_k = _time_mean(t, del_k)
    mean_alpha_prime = _time_mean(t, del_alpha_prime)
    mean_alpha_le = _time_mean(t, del_alpha_le)

    denom_na = mean_nc + mean_na
    share_na = (mean_na / denom_na) if _is_finite(denom_na) and abs(denom_na) > 1e-9 else float("nan")
    fx_abs = abs(mean_fx_suction) + abs(mean_fx_camber) + abs(mean_fx_friction)
    share_fx_suction = abs(mean_fx_suction) / fx_abs if fx_abs > 1e-9 else float("nan")
    share_fx_camber = abs(mean_fx_camber) / fx_abs if fx_abs > 1e-9 else float("nan")
    share_fx_friction = abs(mean_fx_friction) / fx_abs if fx_abs > 1e-9 else float("nan")

    if plot_per_run:
        run_dir = out_csv.parent
        _write_svg_line(
            run_dir / "series_thrust.svg",
            t,
            thrust,
            title=f"Thrust vs time (amp={amp:.2f}, delta={delta:.3f})",
            xlabel="t (s)",
            ylabel="Thrust (N)",
        )
        _write_svg_line(
            run_dir / "series_lift.svg",
            t,
            lift,
            title=f"Lift vs time (amp={amp:.2f}, delta={delta:.3f})",
            xlabel="t (s)",
            ylabel="Lift (N)",
        )
        _write_svg_line(
            run_dir / "series_power.svg",
            t,
            pin,
            title=f"Input Power vs time (amp={amp:.2f}, delta={delta:.3f})",
            xlabel="t (s)",
            ylabel="Power (W)",
        )
        _write_svg_line(
            run_dir / "series_sep_ratio.svg",
            t,
            sep_ratio,
            title=f"Separation Ratio vs time (amp={amp:.2f}, delta={delta:.3f})",
            xlabel="t (s)",
            ylabel="sep_ratio",
            color="#b04a2f",
        )

    if str(profile) == "phase_warp":
        warp = PhaseWarp(
            f_hz=float(f_hz),
            delta=float(delta),
            smoothness=float(smoothness),
            mode=str(mode),
            f_ref_hz=float(f_ref_hz),
        )
        f_eff = warp.f_eff
        f_down = warp.f_down
        f_up = warp.f_up
    else:
        f_eff = float(f_hz)
        f_down = f_eff / (2.0 * float(delta))
        f_up = f_eff / (2.0 * (1.0 - float(delta)))
    return Sample(
        amp=float(amp),
        delta=float(delta),
        f_eff=f_eff,
        f_down=f_down,
        f_up=f_up,
        mean_thrust=mean_thrust,
        mean_lift=mean_lift,
        mean_pin=mean_pin,
        mean_eta=mean_eta,
        sep_ratio=mean_sep,
        peak_power=peak_power,
        peak_normal=peak_normal,
        mean_nc=mean_nc,
        mean_na=mean_na,
        mean_fx_suction=mean_fx_suction,
        mean_fx_camber=mean_fx_camber,
        mean_fx_friction=mean_fx_friction,
        mean_fx_total=mean_fx_total,
        mean_k=mean_k,
        mean_alpha_prime=mean_alpha_prime,
        mean_alpha_le=mean_alpha_le,
        share_na=share_na,
        share_fx_suction=share_fx_suction,
        share_fx_camber=share_fx_camber,
        share_fx_friction=share_fx_friction,
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
    ap.add_argument(
        "--amp",
        type=str,
        default="0.0",
        help="Phase_ff amp list (ignored for phase_warp). Example: 0.0:0.2:0.02.",
    )
    ap.add_argument("--smoothness", type=float, default=0.0, help="Smoothness width (fraction of cycle).")
    ap.add_argument("--mode", type=str, choices=("fixed_f", "fixed_peak_qd"), default="fixed_f")
    ap.add_argument(
        "--modulation-profile",
        type=str,
        choices=("phase_warp", "phase_ff"),
        default="phase_warp",
        help="Modulation profile: phase warp (SCCPFM) or phase feedforward.",
    )
    ap.add_argument("--modulation-ff-amp", type=float, default=0.0, help="Phase feedforward amplitude (fraction).")
    ap.add_argument("--modulation-ff-shift-deg", type=float, default=180.0, help="Phase feedforward shift (deg).")
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
    ap.add_argument(
        "--wing-geom-csv",
        type=Path,
        default=Path("outputs_DeLaurier/right_wing_te_fit_poly5_gap50.csv"),
    )
    ap.add_argument("--N", type=int, default=80, help="Spanwise strip count for BEM discretization.")
    ap.add_argument("--steps", type=int, default=480, help="Simulation steps per run.")
    ap.add_argument("--start-step", type=int, default=0, help="Start step for averaging.")
    ap.add_argument("--end-step", type=int, default=None, help="End step (inclusive) for averaging.")
    ap.add_argument("--extra-args", type=str, default="", help="Extra args forwarded to wind script.")
    ap.add_argument("--rerun", action="store_true", help="Rerun even if CSV exists.")
    ap.add_argument(
        "--plot-per-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write per-run SVG time-series plots.",
    )
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "modulation_scan.csv"

    deltas = _parse_range_list(args.delta)
    amps = _parse_range_list(args.amp)
    extra_args = shlex.split(str(args.extra_args)) if str(args.extra_args).strip() else []

    samples: list[Sample] = []
    for amp in amps:
        for delta in deltas:
            amp_tag = f"amp_{str(amp).replace('.', 'p')}"
            delta_tag = f"delta_{str(delta).replace('.', 'p')}"
            run_dir = out_dir / amp_tag / delta_tag
            csv_path = run_dir / "windtunnel_v50.csv"
            sample = _run_once(
                isaaclab_sh=args.isaaclab_sh,
                wind_script=args.wind_script,
                out_csv=csv_path,
                steps=int(args.steps),
                f_hz=float(args.f_hz),
                f_ref_hz=float(args.f_ref_hz),
                amp=float(amp),
                delta=float(delta),
                smoothness=float(args.smoothness),
                mode=str(args.mode),
                profile=str(args.modulation_profile),
                ff_amp=float(amp) if str(args.modulation_profile) == "phase_ff" else float(args.modulation_ff_amp),
                ff_shift_deg=float(args.modulation_ff_shift_deg),
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
                plot_per_run=bool(args.plot_per_run),
            )
            samples.append(sample)
            print(
                f"[amp {sample.amp:.3f} delta {sample.delta:.3f}] T={sample.mean_thrust:+.4f} "
                f"L={sample.mean_lift:+.4f} Pin={sample.mean_pin:+.4f} eta={sample.mean_eta:+.4f}"
            )

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "modulation_profile",
                "modulation_ff_amp",
                "modulation_ff_shift_deg",
                "amp",
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
                "mean_Nc_N",
                "mean_Na_N",
                "mean_Fx_suction_N",
                "mean_Fx_camber_N",
                "mean_Fx_friction_N",
                "mean_Fx_total_N",
                "mean_k",
                "mean_alpha_prime",
                "mean_alpha_le",
                "share_Na",
                "share_Fx_suction",
                "share_Fx_camber",
                "share_Fx_friction",
                "csv",
            ]
        )
        for s in samples:
            ff_amp_val = s.amp if str(args.modulation_profile) == "phase_ff" else float(args.modulation_ff_amp)
            w.writerow(
                [
                    str(args.modulation_profile),
                    f"{float(ff_amp_val):.6f}",
                    f"{float(args.modulation_ff_shift_deg):.6f}",
                    f"{s.amp:.6f}",
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
                    f"{s.mean_nc:.6f}",
                    f"{s.mean_na:.6f}",
                    f"{s.mean_fx_suction:.6f}",
                    f"{s.mean_fx_camber:.6f}",
                    f"{s.mean_fx_friction:.6f}",
                    f"{s.mean_fx_total:.6f}",
                    f"{s.mean_k:.6f}",
                    f"{s.mean_alpha_prime:.6f}",
                    f"{s.mean_alpha_le:.6f}",
                    f"{s.share_na:.6f}",
                    f"{s.share_fx_suction:.6f}",
                    f"{s.share_fx_camber:.6f}",
                    f"{s.share_fx_friction:.6f}",
                    str(s.csv),
                ]
            )

    xs = [s.delta for s in samples]
    amps_sorted = sorted({s.amp for s in samples})
    for amp in amps_sorted:
        subset = [s for s in samples if abs(s.amp - amp) < 1e-9]
        subset = sorted(subset, key=lambda s: s.delta)
        xs_sub = [s.delta for s in subset]
        tag = f"amp_{str(amp).replace('.', 'p')}"
        _write_svg_line(
            out_dir / f"mod_Tbar_{tag}.svg",
            xs_sub,
            [s.mean_thrust for s in subset],
            title=f"Mean Thrust vs delta (amp={amp:.2f})",
            xlabel="downstroke ratio delta",
            ylabel="T_bar (N)",
        )
        _write_svg_line(
            out_dir / f"mod_Lbar_{tag}.svg",
            xs_sub,
            [s.mean_lift for s in subset],
            title=f"Mean Lift vs delta (amp={amp:.2f})",
            xlabel="downstroke ratio delta",
            ylabel="L_bar (N)",
        )
        _write_svg_line(
            out_dir / f"mod_Pinbar_{tag}.svg",
            xs_sub,
            [s.mean_pin for s in subset],
            title=f"Mean Input Power vs delta (amp={amp:.2f})",
            xlabel="downstroke ratio delta",
            ylabel="P_in_bar (W)",
        )
        _write_svg_line(
            out_dir / f"mod_eta_{tag}.svg",
            xs_sub,
            [s.mean_eta for s in subset],
            title=f"Propulsive Efficiency vs delta (amp={amp:.2f})",
            xlabel="downstroke ratio delta",
            ylabel="eta_bar",
        )
        _write_svg_line(
            out_dir / f"mod_sep_ratio_{tag}.svg",
            xs_sub,
            [s.sep_ratio for s in subset],
            title=f"Separated-Flow Ratio vs delta (amp={amp:.2f})",
            xlabel="downstroke ratio delta",
            ylabel="sep_ratio",
            color="#b04a2f",
        )
    if len(amps_sorted) == 1:
        _write_svg_line(
            out_dir / "mod_Tbar.svg",
            xs,
            [s.mean_thrust for s in samples],
            title="Mean Thrust vs delta",
            xlabel="downstroke ratio delta",
            ylabel="T_bar (N)",
        )
        _write_svg_line(
            out_dir / "mod_Lbar.svg",
            xs,
            [s.mean_lift for s in samples],
            title="Mean Lift vs delta",
            xlabel="downstroke ratio delta",
            ylabel="L_bar (N)",
        )
        _write_svg_line(
            out_dir / "mod_Pinbar.svg",
            xs,
            [s.mean_pin for s in samples],
            title="Mean Input Power vs delta",
            xlabel="downstroke ratio delta",
            ylabel="P_in_bar (W)",
        )
        _write_svg_line(
            out_dir / "mod_eta.svg",
            xs,
            [s.mean_eta for s in samples],
            title="Propulsive Efficiency vs delta",
            xlabel="downstroke ratio delta",
            ylabel="eta_bar",
        )
        _write_svg_line(
            out_dir / "mod_sep_ratio.svg",
            xs,
            [s.sep_ratio for s in samples],
            title="Separated-Flow Ratio vs delta",
            xlabel="downstroke ratio delta",
            ylabel="sep_ratio",
            color="#b04a2f",
        )
        _write_svg_line(
            out_dir / "mod_peak_power.svg",
            xs,
            [s.peak_power for s in samples],
            title="Peak Power vs delta",
            xlabel="downstroke ratio delta",
            ylabel="P_peak (W)",
            color="#8a3a91",
        )
        _write_svg_line(
            out_dir / "mod_share_suction.svg",
            xs,
            [s.share_fx_suction for s in samples],
            title="Suction Share vs delta",
            xlabel="downstroke ratio delta",
            ylabel="|Fx_suction| share",
            color="#2d8f5f",
        )
        _write_svg_line(
            out_dir / "mod_share_na.svg",
            xs,
            [s.share_na for s in samples],
            title="Added-Mass Share vs delta",
            xlabel="downstroke ratio delta",
            ylabel="Na share",
            color="#d28b26",
        )
    # Relative improvements vs delta=0.5, per amp (if present).
    rel_csv = out_dir / "modulation_scan_relative.csv"
    with rel_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "amp",
                "delta",
                "rel_Tbar",
                "rel_Lbar",
                "rel_Pinbar",
                "rel_eta",
                "rel_sep_ratio",
                "rel_peak_power",
                "rel_peak_normal",
                "rel_share_suction",
                "rel_share_Na",
            ]
        )
        for amp in amps_sorted:
            subset = [s for s in samples if abs(s.amp - amp) < 1e-9]
            ref = next((s for s in subset if abs(s.delta - 0.5) < 1e-6), None)
            if ref is None:
                continue
            for s in subset:
                def _rel(v: float, b: float) -> float:
                    return (v - b) / abs(b) if _is_finite(v) and _is_finite(b) and abs(b) > 1e-9 else float("nan")

                w.writerow(
                    [
                        f"{s.amp:.6f}",
                        f"{s.delta:.6f}",
                        f"{_rel(s.mean_thrust, ref.mean_thrust):.6f}",
                        f"{_rel(s.mean_lift, ref.mean_lift):.6f}",
                        f"{_rel(s.mean_pin, ref.mean_pin):.6f}",
                        f"{_rel(s.mean_eta, ref.mean_eta):.6f}",
                        f"{_rel(s.sep_ratio, ref.sep_ratio):.6f}",
                        f"{_rel(s.peak_power, ref.peak_power):.6f}",
                        f"{_rel(s.peak_normal, ref.peak_normal):.6f}",
                        f"{_rel(s.share_fx_suction, ref.share_fx_suction):.6f}",
                        f"{_rel(s.share_na, ref.share_na):.6f}",
                    ]
                )
    summary_txt = out_dir / "modulation_scan_summary.txt"
    with summary_txt.open("w", newline="") as f:
        for amp in amps_sorted:
            subset = [s for s in samples if abs(s.amp - amp) < 1e-9]
            ref = next((s for s in subset if abs(s.delta - 0.5) < 1e-6), None)
            if ref is None:
                f.write(f"amp={amp:.2f}: missing delta=0.5 reference\n")
                continue

            def _best(metric: str, higher_is_better: bool) -> tuple[float, float]:
                best_delta = float("nan")
                best_gain = -1e9
                base = getattr(ref, metric)
                if not _is_finite(base) or abs(base) <= 1e-9:
                    return best_delta, float("nan")
                for s in subset:
                    val = getattr(s, metric)
                    if not _is_finite(val):
                        continue
                    rel = (val - base) / abs(base)
                    gain = rel if higher_is_better else -rel
                    if gain > best_gain:
                        best_gain = gain
                        best_delta = s.delta
                return best_delta, best_gain

            f.write(f"amp={amp:.2f} (reference delta=0.5)\n")
            for metric, label, higher in [
                ("mean_thrust", "Tbar", True),
                ("mean_lift", "Lbar", True),
                ("mean_eta", "eta", True),
                ("mean_pin", "Pinbar (lower better)", False),
                ("sep_ratio", "sep_ratio (lower better)", False),
                ("peak_power", "peak_power (lower better)", False),
                ("peak_normal", "peak_normal (lower better)", False),
            ]:
                d_best, gain = _best(metric, higher)
                if _is_finite(gain):
                    f.write(f"  {label}: best delta={d_best:.3f}, rel_gain={gain:+.3f}\n")
            f.write("  Shares (higher better):\n")
            d_best, gain = _best("share_fx_suction", True)
            if _is_finite(gain):
                f.write(f"  Fx_suction share: best delta={d_best:.3f}, rel_gain={gain:+.3f}\n")
            d_best, gain = _best("share_na", True)
            if _is_finite(gain):
                f.write(f"  Na share: best delta={d_best:.3f}, rel_gain={gain:+.3f}\n")
            f.write("\n")
    print(f"[OK] relative CSV: {rel_csv}")
    print(f"[OK] summary txt: {summary_txt}")
    print(f"[OK] summary CSV: {out_csv}")
    print(f"[OK] plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
