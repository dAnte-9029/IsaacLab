"""Plot tracking curves from the CSV produced by run_minimal_env.py.

Usage (from IsaacLab root):

  # show only
  .\\isaaclab.bat -p source/flapping_bot/scripts/plot_tracking.py --csv logs/track.csv --show

  # save to file (no GUI)
  .\\isaaclab.bat -p source/flapping_bot/scripts/plot_tracking.py --csv logs/track.csv --save plots/track.png

The script is resilient to the presence/absence of mid-tail columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot tracking curves from CSV")
    p.add_argument("--csv", required=True, help="Path to logs/track.csv")
    p.add_argument("--save", default="", help="Optional path to save a PNG figure")
    p.add_argument("--show", action="store_true", help="Show an interactive matplotlib window")
    p.add_argument("--title", default="FlappingBot Tracking", help="Figure title")
    p.add_argument("--deg", action="store_true", help="Convert radians to degrees for plotting")
    p.add_argument("--dpi", type=int, default=120, help="Figure DPI when saving")
    return p.parse_args()


def load_csv(path: Path) -> tuple[list[str], list[list[float]]]:
    rows: list[list[float]] = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for r in reader:
            try:
                rows.append([float(x) for x in r])
            except ValueError:
                # skip malformed lines
                continue
    return header, rows


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERR] CSV not found: {csv_path}")
        return 2

    header, rows = load_csv(csv_path)
    if not rows:
        print("[WARN] Empty CSV data; nothing to plot.")
        return 0

    # Build column index map
    h2i = {h: i for i, h in enumerate(header)}
    req = [
        "time",
        "freq_hz",
        "tgt_wing_L",
        "tgt_wing_R",
        "pos_wing_L",
        "pos_wing_R",
        "tgt_tail_L",
        "tgt_tail_R",
        "pos_tail_L",
        "pos_tail_R",
    ]
    for k in req:
        if k not in h2i:
            print(f"[ERR] Missing column in CSV: {k}")
            return 2

    has_mid = ("tgt_mid" in h2i) and ("pos_mid" in h2i)

    # Convert to lists for plotting
    def col(name: str) -> list[float]:
        return [r[h2i[name]] for r in rows]

    t = col("time")
    freq = col("freq_hz")

    # wings
    tgt_wL, tgt_wR = col("tgt_wing_L"), col("tgt_wing_R")
    pos_wL, pos_wR = col("pos_wing_L"), col("pos_wing_R")

    # tails
    tgt_tL, tgt_tR = col("tgt_tail_L"), col("tgt_tail_R")
    pos_tL, pos_tR = col("pos_tail_L"), col("pos_tail_R")

    # mid tail (optional)
    tgt_m, pos_m = (col("tgt_mid"), col("pos_mid")) if has_mid else (None, None)

    if args.deg:
        import math
        to_deg = lambda arr: [math.degrees(x) for x in arr]
        tgt_wL, tgt_wR, pos_wL, pos_wR = map(to_deg, (tgt_wL, tgt_wR, pos_wL, pos_wR))
        tgt_tL, tgt_tR, pos_tL, pos_tR = map(to_deg, (tgt_tL, tgt_tR, pos_tL, pos_tR))
        if has_mid:
            tgt_m, pos_m = to_deg(tgt_m), to_deg(pos_m)

    # Plot
    import matplotlib.pyplot as plt
    import numpy as np

    nrows = 3 if has_mid else 2
    fig, axs = plt.subplots(nrows, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(args.title)
    axs = np.atleast_1d(axs)

    # wings
    axs[0].plot(t, tgt_wL, label="tgt wing L")
    axs[0].plot(t, pos_wL, label="pos wing L")
    axs[0].plot(t, tgt_wR, label="tgt wing R", ls="--")
    axs[0].plot(t, pos_wR, label="pos wing R", ls="--")
    axs[0].set_ylabel("wing (deg)" if args.deg else "wing (rad)")
    ax2 = axs[0].twinx(); ax2.plot(t, freq, color="gray", alpha=0.35); ax2.set_ylabel("freq (Hz)")
    axs[0].legend(loc="upper right")

    # tails
    axs[1].plot(t, tgt_tL, label="tgt tail L")
    axs[1].plot(t, pos_tL, label="pos tail L")
    axs[1].plot(t, tgt_tR, label="tgt tail R", ls="--")
    axs[1].plot(t, pos_tR, label="pos tail R", ls="--")
    axs[1].set_ylabel("tail (deg)" if args.deg else "tail (rad)")
    axs[1].legend(loc="upper right")

    if has_mid and tgt_m is not None and pos_m is not None:
        axs[2].plot(t, tgt_m, label="tgt mid")
        axs[2].plot(t, pos_m, label="pos mid")
        axs[2].set_ylabel("mid (deg)" if args.deg else "mid (rad)")
        axs[2].legend(loc="upper right")

    axs[-1].set_xlabel("time (s)")
    fig.tight_layout()

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=args.dpi)
        print(f"[OK] Saved figure to {args.save}")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

