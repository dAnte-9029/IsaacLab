#!/usr/bin/env python3
"""Sweep wind-tunnel airspeed V and summarize mean thrust/lift vs V.

This runs `scripts/isaac_wind_tunnel_flappingbot_v50.py` multiple times (one per V),
then reads the produced CSVs and computes mean forces in world frame.

Use-case:
    - With `--profile-cd0 0`, visualize the raw QSM thrust/lift trends vs forward speed.
    - Find:
        V1: thrust(V) crosses 0 (thrust -> drag).
        V2: lift(V) crosses mg (insufficient -> sufficient).
"""

from __future__ import annotations

import argparse
import csv
import math
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


def _parse_range_list(s: str) -> list[float]:
    s = str(s).strip()
    if ":" in s:
        # start:stop:step (inclusive stop if it lands exactly)
        parts = s.split(":")
        if len(parts) != 3:
            raise ValueError("Range must be 'start:stop:step'.")
        start, stop, step = (float(p) for p in parts)
        if step <= 0:
            raise ValueError("Range step must be > 0.")
        out: list[float] = []
        v = start
        # include stop with a small tolerance
        while v <= stop + 1e-9:
            out.append(float(v))
            v += step
        return out
    # comma-separated
    vals = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(float(tok))
    if not vals:
        raise ValueError("Empty V list.")
    return vals


def _safe_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _format_V(v: float) -> str:
    return f"{v:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def _mean_over_steps(csv_path: Path, *, start_step: int, end_step: int | None) -> dict[str, float]:
    keys = ["F_world_x_T", "F_world_y_T", "F_world_z_T"]
    acc = {k: 0.0 for k in keys}
    n = 0
    with csv_path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            step = int(float(row.get("step", "nan")))
            if step < start_step:
                continue
            if end_step is not None and step > end_step:
                continue
            ok = True
            vals: dict[str, float] = {}
            for k in keys:
                v = row.get(k, "")
                if v == "" or v.lower() == "nan":
                    ok = False
                    break
                vals[k] = _safe_float(v)
            if not ok:
                continue
            for k in keys:
                acc[k] += vals[k]
            n += 1
    if n == 0:
        raise ValueError(f"No valid samples found in {csv_path} (check --start-step/--end-step).")
    return {k: acc[k] / n for k in keys} | {"n": float(n)}


def _normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = v
    n = math.sqrt(x * x + y * y + z * z)
    if n <= 1e-12:
        raise ValueError("flow_dir_world must be non-zero.")
    return (x / n, y / n, z / n)


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(a: tuple[float, float, float], s: float) -> tuple[float, float, float]:
    return (a[0] * s, a[1] * s, a[2] * s)


def _find_crossing(xs: list[float], ys: list[float]) -> float | None:
    """Return x where y crosses 0 (linear interpolation). xs must be increasing."""
    for i in range(len(xs) - 1):
        y0, y1 = ys[i], ys[i + 1]
        if not (math.isfinite(y0) and math.isfinite(y1)):
            continue
        if y0 == 0.0:
            return xs[i]
        if y0 * y1 < 0.0:
            t = abs(y0) / (abs(y0) + abs(y1))
            return xs[i] + t * (xs[i + 1] - xs[i])
    return None


@dataclass(frozen=True)
class Sample:
    V: float
    thrust: float
    lift_z: float
    lift_rel: float
    csv: Path


def _run_once(
    *,
    isaaclab_sh: Path,
    wind_script: Path,
    out_csv: Path,
    steps: int,
    f_hz: float,
    body_pitch_deg: float,
    V: float,
    flow_dir_world: tuple[float, float, float],
    wing_geom_csv: Path,
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
            # Ensure no URDF auto-geom in sweeps (reproducibility).
            "--no-auto-geom-from-urdf",
            # IMPORTANT for this sweep: disable the added profile drag term.
            "--profile-cd0",
            "0.0",
        ]
        if extra_args:
            cmd += list(extra_args)
        subprocess.run(cmd, check=True)

    means = _mean_over_steps(out_csv, start_step=int(start_step), end_step=end_step)
    Fx = float(means["F_world_x_T"])
    Fy = float(means["F_world_y_T"])
    Fz = float(means["F_world_z_T"])

    # Define forward axis as opposite of wind direction (air flows towards flow_dir_world).
    flow_hat = _normalize(flow_dir_world)
    e_forward = _scale(flow_hat, -1.0)
    thrust = _dot((Fx, Fy, Fz), e_forward)

    # Define "up ⟂ flow" by projecting +Z onto plane orthogonal to flow.
    z_w = (0.0, 0.0, 1.0)
    z_perp = _sub(z_w, _scale(flow_hat, _dot(z_w, flow_hat)))
    z_perp_hat = _normalize(z_perp) if math.sqrt(_dot(z_perp, z_perp)) > 1e-12 else z_w
    lift_rel = _dot((Fx, Fy, Fz), z_perp_hat)

    return Sample(V=float(V), thrust=float(thrust), lift_z=float(Fz), lift_rel=float(lift_rel), csv=out_csv)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep wind speed V and summarize thrust/lift vs V (no profile CD0).")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--wind-script", type=Path, default=Path("scripts/isaac_wind_tunnel_flappingbot_v50.py"))
    ap.add_argument(
        "--isaaclab-sh",
        type=Path,
        default=(Path(__file__).resolve().parents[1] / "isaaclab.sh"),
        help="Path to repo wrapper script (default: <repo>/isaaclab.sh).",
    )
    ap.add_argument("--wing-geom-csv", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=2400)
    ap.add_argument("--start-step", type=int, default=240)
    ap.add_argument("--end-step", type=int, default=None)
    ap.add_argument("--f-hz", type=float, required=True)
    ap.add_argument("--pitch-deg", type=float, required=True)
    ap.add_argument("--mass-kg", type=float, required=True)
    ap.add_argument("--flow-dir-world", type=float, nargs=3, default=(-1.0, 0.0, 0.0))
    ap.add_argument("--V", type=str, required=True, help="Comma list '0,3,6' or range '0:10:0.5'.")
    ap.add_argument(
        "--extra-args",
        type=str,
        default="",
        help="Extra args appended to wind script (shell-style string). Prefer using '-- <args...>' to avoid quoting issues.",
    )
    ap.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra args passed to wind script after '--' (recommended; avoids shell quoting).",
    )
    ap.add_argument("--rerun", action="store_true", help="Re-run even if per-V CSV already exists.")
    ap.add_argument("--plot", action="store_true", help="Write a PNG plot in out-dir.")
    args = ap.parse_args()

    Vs = sorted(_parse_range_list(str(args.V)))
    flow_dir = (float(args.flow_dir_world[0]), float(args.flow_dir_world[1]), float(args.flow_dir_world[2]))
    mg = float(args.mass_kg) * 9.81

    extra_tokens: list[str] = []
    if str(args.extra_args).strip():
        extra_tokens += shlex.split(str(args.extra_args))
    if args.extra:
        # argparse.REMAINDER includes everything after '--'. Depending on how the script is invoked,
        # it may include a literal "--" token; strip it so we don't forward it to the wind script.
        tail = list(args.extra)
        if tail and tail[0] == "--":
            tail = tail[1:]
        extra_tokens += tail

    samples: list[Sample] = []
    for V in Vs:
        out_csv = args.out_dir / f"V{_format_V(V)}.csv"
        s = _run_once(
            isaaclab_sh=args.isaaclab_sh,
            wind_script=args.wind_script,
            out_csv=out_csv,
            steps=int(args.steps),
            f_hz=float(args.f_hz),
            body_pitch_deg=float(args.pitch_deg),
            V=float(V),
            flow_dir_world=flow_dir,
            wing_geom_csv=args.wing_geom_csv,
            start_step=int(args.start_step),
            end_step=args.end_step,
            extra_args=extra_tokens,
            rerun=bool(args.rerun),
        )
        print(f"[OK] V={s.V:.3f}  thrust={s.thrust:+.6f} N  lift_z={s.lift_z:+.6f} N  lift_rel={s.lift_rel:+.6f} N  csv={s.csv}")
        samples.append(s)

    # Write summary
    summary = args.out_dir / "V_sweep_summary.csv"
    with summary.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["V_mps", "mean_thrust_N", "mean_lift_z_N", "mean_lift_rel_N", "mg_N", "csv"])
        for s in samples:
            w.writerow([f"{s.V:.9g}", f"{s.thrust:.9g}", f"{s.lift_z:.9g}", f"{s.lift_rel:.9g}", f"{mg:.9g}", str(s.csv)])
    print(f"[OK] wrote: {summary}")

    V1 = _find_crossing([s.V for s in samples], [s.thrust for s in samples])
    V2 = _find_crossing([s.V for s in samples], [s.lift_rel - mg for s in samples])
    if V1 is not None:
        print(f"[OK] V1 (thrust=0): {V1:.6f} m/s")
    else:
        print("[WARN] V1 (thrust=0) not bracketed by the provided V samples.")
    if V2 is not None:
        print(f"[OK] V2 (lift_rel=mg): {V2:.6f} m/s  (mg={mg:.6f} N)")
    else:
        print("[WARN] V2 (lift_rel=mg) not bracketed by the provided V samples.")

    if args.plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # type: ignore

            xs = [s.V for s in samples]
            thrusts = [s.thrust for s in samples]
            lifts = [s.lift_rel for s in samples]

            fig, ax = plt.subplots(1, 1, figsize=(7, 4.2))
            ax.plot(xs, thrusts, label="thrust (F·e_forward)", lw=2)
            ax.axhline(0.0, color="k", lw=1, alpha=0.6)
            ax2 = ax.twinx()
            ax2.plot(xs, lifts, label="lift_rel (up ⟂ flow)", lw=2, color="tab:orange")
            ax2.axhline(mg, color="tab:orange", lw=1, alpha=0.6, linestyle="--")

            if V1 is not None:
                ax.axvline(V1, color="tab:blue", lw=1, alpha=0.5, linestyle="--")
            if V2 is not None:
                ax.axvline(V2, color="tab:orange", lw=1, alpha=0.5, linestyle="--")

            ax.set_xlabel("V (m/s)")
            ax.set_ylabel("thrust (N)")
            ax2.set_ylabel("lift_rel (N)")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            out_png = args.out_dir / "V_sweep_thrust_lift.png"
            fig.savefig(out_png, dpi=180)
            print(f"[OK] wrote: {out_png}")
        except Exception as exc:
            print(f"[WARN] plot failed: {exc}")


if __name__ == "__main__":
    main()
