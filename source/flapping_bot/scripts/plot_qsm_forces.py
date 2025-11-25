"""Plot QSM lift/thrust curves from CSV logged by run_mass6_env.py.

Usage (from repo root):

    conda activate env_isaaclab
    cd /home/zn/IsaacLab
    ./isaaclab.sh -p source/flapping_bot/scripts/plot_qsm_forces.py \
        --csv outputs/flapping_qsm_forces.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot QSM lift/thrust curves from CSV.")
    parser.add_argument(
        "--csv",
        type=str,
        default="outputs/flapping_qsm_forces.csv",
        help="Path to CSV file produced by run_mass6_env.py.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.is_file():
        print(f"[QSM-PLOT] CSV file not found: {csv_path}")
        return 1

    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover - import guard
        print("[QSM-PLOT] Missing dependencies (pandas/matplotlib).")
        print("          Install them inside env_isaaclab, e.g.:")
        print("              ./isaaclab.sh -p -m pip install pandas matplotlib")
        print("          Error:", repr(e))
        return 1

    df = pd.read_csv(csv_path)
    required_cols = {"t", "thrust_bx", "side_by", "lift_bz", "roll_tx", "pitch_ty", "yaw_tz"}
    if not required_cols.issubset(df.columns):
        print(f"[QSM-PLOT] CSV missing required columns {required_cols}, got {list(df.columns)}")
        return 1

    # Base signals (totals)
    t = df["t"].to_numpy()
    thrust = df["thrust_bx"].to_numpy()
    side = df["side_by"].to_numpy()
    lift = df["lift_bz"].to_numpy()
    roll = df["roll_tx"].to_numpy()
    pitch = df["pitch_ty"].to_numpy()
    yaw = df["yaw_tz"].to_numpy()

    # Optional split into wings vs tails
    split_cols = {
        "wing_fx",
        "wing_fy",
        "wing_fz",
        "wing_tx",
        "wing_ty",
        "wing_tz",
        "tail_fx",
        "tail_fy",
        "tail_fz",
        "tail_tx",
        "tail_ty",
        "tail_tz",
    }
    has_split = split_cols.issubset(df.columns)
    if has_split:
        wing_fx = df["wing_fx"].to_numpy()
        wing_fy = df["wing_fy"].to_numpy()
        wing_fz = df["wing_fz"].to_numpy()
        wing_tx = df["wing_tx"].to_numpy()
        wing_ty = df["wing_ty"].to_numpy()
        wing_tz = df["wing_tz"].to_numpy()
        tail_fx = df["tail_fx"].to_numpy()
        tail_fy = df["tail_fy"].to_numpy()
        tail_fz = df["tail_fz"].to_numpy()
        tail_tx = df["tail_tx"].to_numpy()
        tail_ty = df["tail_ty"].to_numpy()
        tail_tz = df["tail_tz"].to_numpy()

    def _plot_block(
        t_arr,
        fx,
        fy,
        fz,
        tx,
        ty,
        tz,
        title_prefix: str,
        out_path: Path,
    ) -> None:
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

        # Forces
        ax_f = axes[0]
        ax_f.plot(t_arr, fz, label="lift (body z)")
        ax_f.plot(t_arr, fx, label="thrust (body x)")
        ax_f.plot(t_arr, fy, label="side (body y)")
        ax_f.axhline(0.0, color="k", linewidth=0.5)
        ax_f.set_ylabel("force [N]")
        ax_f.set_title(f"{title_prefix} force (body frame)")
        ax_f.legend()
        ax_f.grid(True)

        # Torques
        ax_tau = axes[1]
        ax_tau.plot(t_arr, ty, label="pitch τ_y")
        ax_tau.plot(t_arr, tx, label="roll τ_x")
        ax_tau.plot(t_arr, tz, label="yaw τ_z")
        ax_tau.axhline(0.0, color="k", linewidth=0.5)
        ax_tau.set_xlabel("sim time [s]")
        ax_tau.set_ylabel("torque [N·m]")
        ax_tau.set_title(f"{title_prefix} torque (body frame)")
        ax_tau.legend()
        ax_tau.grid(True)

        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"[QSM-PLOT] Saved figure to {out_path}")

    out_dir = csv_path.parent if csv_path.parent.name else Path("outputs")

    # 1) Total contributions
    _plot_block(
        t,
        fx=thrust,
        fy=side,
        fz=lift,
        tx=roll,
        ty=pitch,
        tz=yaw,
        title_prefix="QSM total",
        out_path=out_dir / "flapping_qsm_forces_total.png",
    )

    # 2) Wings only (if available)
    if has_split:
        _plot_block(
            t,
            fx=wing_fx,
            fy=wing_fy,
            fz=wing_fz,
            tx=wing_tx,
            ty=wing_ty,
            tz=wing_tz,
            title_prefix="QSM wings",
            out_path=out_dir / "flapping_qsm_forces_wing.png",
        )

        # 3) Tails only
        _plot_block(
            t,
            fx=tail_fx,
            fy=tail_fy,
            fz=tail_fz,
            tx=tail_tx,
            ty=tail_ty,
            tz=tail_tz,
            title_prefix="QSM tails",
            out_path=out_dir / "flapping_qsm_forces_tail.png",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
