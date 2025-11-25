"""Plot wing joint angle response from CSV logged by run_mass6_env.py.

Usage (from repo root):

    conda activate env_isaaclab
    cd /home/zn/IsaacLab
    ./isaaclab.sh -p source/flapping_bot/scripts/plot_wing_angle.py \
        --csv outputs/flapping_wing_angle.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot wing joint angle response from CSV.")
    parser.add_argument(
        "--csv",
        type=str,
        default="outputs/flapping_wing_angle.csv",
        help="Path to CSV file produced by run_mass6_env.py (--log-wing-angle).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.is_file():
        print(f"[WING-PLOT] CSV file not found: {csv_path}")
        return 1

    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover - import guard
        print("[WING-PLOT] Missing dependencies (pandas/matplotlib).")
        print("           Install them inside env_isaaclab, e.g.:")
        print("               ./isaaclab.sh -p -m pip install pandas matplotlib")
        print("           Error:", repr(e))
        return 1

    df = pd.read_csv(csv_path)
    required_cols = {"t", "q", "q_target"}
    if not required_cols.issubset(df.columns):
        print(f"[WING-PLOT] CSV missing required columns {required_cols}, got {list(df.columns)}")
        return 1

    t = df["t"].to_numpy()
    q = df["q"].to_numpy()
    q_target = df["q_target"].to_numpy()

    # Restrict to the last 1s window for clearer comparison
    t_max = float(t.max())
    t_min = max(0.0, t_max - 1.0)
    mask = (t >= t_min) & (t <= t_max)
    if not mask.any():
        print("[WING-PLOT] No samples in 1s window; plotting full duration instead.")
        t_win = t
        q_win = q
        q_tgt_win = q_target
        t0 = t_win[0]
    else:
        t_win = t[mask]
        q_win = q[mask]
        q_tgt_win = q_target[mask]
        t0 = t_min

    # Normalize time axis so window is [0, 1] s
    t_norm = t_win - t0

    out_dir = csv_path.parent if csv_path.parent.name else Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Actual angle
    fig_actual, ax_actual = plt.subplots(1, 1, figsize=(9, 3))
    ax_actual.plot(t_norm, q_win, label="wing angle (actual)")
    ax_actual.set_xlabel("time in window [s]")
    ax_actual.set_ylabel("angle [rad]")
    ax_actual.set_title("Wing angle (actual) – last 1s window")
    ax_actual.grid(True)
    ax_actual.legend()
    plt.tight_layout()
    out_actual = out_dir / "flapping_wing_angle_actual.png"
    fig_actual.savefig(out_actual, dpi=150)
    plt.close(fig_actual)
    print(f"[WING-PLOT] Saved actual-angle figure to {out_actual}")

    # Target angle
    fig_tgt, ax_tgt = plt.subplots(1, 1, figsize=(9, 3))
    ax_tgt.plot(t_norm, q_tgt_win, label="wing angle (target)")
    ax_tgt.set_xlabel("time in window [s]")
    ax_tgt.set_ylabel("angle [rad]")
    ax_tgt.set_title("Wing angle (target) – last 1s window")
    ax_tgt.grid(True)
    ax_tgt.legend()
    plt.tight_layout()
    out_tgt = out_dir / "flapping_wing_angle_target.png"
    fig_tgt.savefig(out_tgt, dpi=150)
    plt.close(fig_tgt)
    print(f"[WING-PLOT] Saved target-angle figure to {out_tgt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
