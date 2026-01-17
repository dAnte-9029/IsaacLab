"""Summarize twist behavior across many wind-tunnel CSV files.

Reads CSVs produced by `scripts/isaac_wind_tunnel_flappingbot_v50.py` and outputs:
  - per-file peak/rms of eta_tip (deg)
  - peak wing_qd (rad/s)
  - mean airspeed (m/s)
  - best-effort parsing of f_hz / twist params from filename tags (auto_sweep naming)
Optionally produces a quick scatter plot.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Row:
    csv: str
    f_hz: float | None
    airspeed: float
    body_pitch_deg: float | None
    eta_tip_peak_deg: float
    eta_tip_rms_deg: float
    wing_qd_peak: float

    eta_amp_deg: float | None = None
    eta_phase_deg: float | None = None


def _safe_mean(xs: list[float]) -> float:
    ys = [x for x in xs if not (math.isnan(x) or math.isinf(x))]
    return float(sum(ys) / len(ys)) if ys else float("nan")


def _read_cols(path: Path, *, start_step: int, end_step: int | None) -> dict[str, np.ndarray]:
    need = ("t", "wing_qd", "eta_tip_L", "eta_tip_R", "airspeed")
    cols: dict[str, list[float]] = {k: [] for k in need}
    extra = ("body_pitch_deg",)
    for k in extra:
        cols[k] = []

    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        names = set(r.fieldnames)
        missing = [k for k in need if k not in names]
        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")
        for row in r:
            for k in need:
                cols[k].append(float(row[k]))
            for k in extra:
                if k in names:
                    cols[k].append(float(row[k]))

    n = len(cols["t"])
    i0 = max(0, int(start_step))
    i1 = int(end_step) if end_step is not None else n
    i1 = max(i0, min(i1, n))

    out: dict[str, np.ndarray] = {}
    for k, vs in cols.items():
        if not vs:
            continue
        out[k] = np.asarray(vs[i0:i1], dtype=np.float64)
    return out


def _parse_tag_float(tag: str) -> float | None:
    # auto_sweep style: 6p035 -> 6.035 ; m5p000 -> -5.0
    s = tag.replace("m", "-").replace("p", ".")
    try:
        return float(s)
    except Exception:
        return None


def _parse_from_name(path: Path) -> dict[str, float]:
    name = path.stem
    out: dict[str, float] = {}

    # auto_sweep: f5p000_amp10p000_ph90p000_pitch10p000_V6p193
    for key, pref in (("f_hz", "f"), ("eta_amp_deg", "amp"), ("eta_phase_deg", "ph"), ("pitch_deg", "pitch"), ("V", "V")):
        m = re.search(rf"(?:^|_){pref}(?P<v>m?[0-9]+p[0-9]+)", name)
        if m:
            v = _parse_tag_float(m.group("v"))
            if v is not None:
                out[key] = v

    # manual: f3_V4_amp10
    m = re.search(r"(?:^|_)f(?P<f>[0-9]+(?:\\.[0-9]+)?)", name)
    if m and "f_hz" not in out:
        out["f_hz"] = float(m.group("f"))
    m = re.search(r"(?:^|_)V(?P<V>[0-9]+(?:\\.[0-9]+)?)", name)
    if m and "V" not in out:
        out["V"] = float(m.group("V"))
    m = re.search(r"(?:^|_)amp(?P<a>[0-9]+(?:\\.[0-9]+)?)", name)
    if m and "eta_amp_deg" not in out:
        out["eta_amp_deg"] = float(m.group("a"))

    return out


def _iter_csvs(globs: list[str]) -> Iterable[Path]:
    seen: set[Path] = set()
    for g in globs:
        for p in sorted(Path().glob(g)):
            if p.is_file() and p.suffix.lower() == ".csv" and p not in seen:
                seen.add(p)
                yield p


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize twist vs sweep parameters from many CSVs.")
    p.add_argument("--glob", action="append", required=True, help="Glob(s) for input CSVs (repeatable).")
    p.add_argument("--out-csv", type=Path, default=Path("outputs/twist_sweep_summary.csv"), help="Output summary CSV.")
    p.add_argument("--start-step", type=int, default=240, help="Discard first N steps.")
    p.add_argument("--end-step", type=int, default=None, help="Stop before this step.")
    p.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True, help="Also write a quick scatter plot PNG.")
    p.add_argument("--out-png", type=Path, default=Path("outputs/twist_sweep_summary.png"), help="Output plot PNG.")
    args = p.parse_args()

    rows: list[Row] = []
    for csv_path in _iter_csvs(list(args.glob)):
        cols = _read_cols(csv_path, start_step=int(args.start_step), end_step=args.end_step)
        etaL = np.degrees(cols["eta_tip_L"])
        etaR = np.degrees(cols["eta_tip_R"])
        eta_abs = np.maximum(np.abs(etaL), np.abs(etaR))
        eta_peak = float(np.nanmax(eta_abs))
        eta_rms = float(math.sqrt(float(np.nanmean(eta_abs**2))))
        qd_peak = float(np.nanmax(np.abs(cols["wing_qd"])))
        airspeed = float(np.nanmean(cols["airspeed"]))
        pitch = float(np.nanmean(cols["body_pitch_deg"])) if "body_pitch_deg" in cols else None

        meta = _parse_from_name(csv_path)
        f_hz = float(meta["f_hz"]) if "f_hz" in meta else None
        eta_amp = float(meta["eta_amp_deg"]) if "eta_amp_deg" in meta else None
        eta_ph = float(meta["eta_phase_deg"]) if "eta_phase_deg" in meta else None

        rows.append(
            Row(
                csv=str(csv_path),
                f_hz=f_hz,
                airspeed=airspeed,
                body_pitch_deg=pitch,
                eta_tip_peak_deg=eta_peak,
                eta_tip_rms_deg=eta_rms,
                wing_qd_peak=qd_peak,
                eta_amp_deg=eta_amp,
                eta_phase_deg=eta_ph,
            )
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "csv",
                "f_hz",
                "airspeed",
                "body_pitch_deg",
                "eta_tip_peak_deg",
                "eta_tip_rms_deg",
                "wing_qd_peak",
                "eta_amp_deg",
                "eta_phase_deg",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.csv,
                    "" if r.f_hz is None else f"{r.f_hz:g}",
                    f"{r.airspeed:g}",
                    "" if r.body_pitch_deg is None else f"{r.body_pitch_deg:g}",
                    f"{r.eta_tip_peak_deg:g}",
                    f"{r.eta_tip_rms_deg:g}",
                    f"{r.wing_qd_peak:g}",
                    "" if r.eta_amp_deg is None else f"{r.eta_amp_deg:g}",
                    "" if r.eta_phase_deg is None else f"{r.eta_phase_deg:g}",
                ]
            )

    print(f"[OK] wrote: {args.out_csv}")

    if bool(args.plot):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            raise RuntimeError(f"matplotlib is required to plot but could not be imported: {exc}") from exc

        # Simple scatter: eta_peak vs airspeed, colored by f (if available).
        xs = np.asarray([r.airspeed for r in rows], dtype=np.float64)
        ys = np.asarray([r.eta_tip_peak_deg for r in rows], dtype=np.float64)
        fs = np.asarray([float("nan") if r.f_hz is None else r.f_hz for r in rows], dtype=np.float64)

        fig, ax = plt.subplots(1, 1, figsize=(8.5, 5))
        if np.all(np.isnan(fs)):
            ax.scatter(xs, ys, s=18, alpha=0.8)
        else:
            sc = ax.scatter(xs, ys, c=fs, s=18, alpha=0.9, cmap="viridis")
            fig.colorbar(sc, ax=ax, label="f_hz")
        ax.set_xlabel("airspeed (m/s)")
        ax.set_ylabel("eta_tip_peak (deg)")
        ax.grid(True)
        fig.tight_layout()
        args.out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out_png, dpi=200)
        plt.close(fig)
        print(f"[OK] wrote: {args.out_png}")


if __name__ == "__main__":
    main()
