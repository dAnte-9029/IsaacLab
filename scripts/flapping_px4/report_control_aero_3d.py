from pathlib import Path
import pandas as pd, numpy as np, json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = Path("logs/control_aero_3d")
out = Path("docs/audits/2026-10-29-control-oriented-3d-aero")
means = pd.read_csv(base / "ll_final/means.csv")
trim = pd.read_csv(base / "system_trim_compare/means.csv")
phase = pd.read_csv(base / "system_trim_compare/phase.csv.gz")
cases = pd.read_csv(base / "ll_final/cases.csv").set_index("case")
fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
for model in ["L3D", "I3D"]:
    for ax, col in zip(axes[0], ["Fy", "Mx"]):
        x = means[
            (means.model == model)
            & (means.twist == 35)
            & (means.alpha == 0)
            & (means.hz == 4)
            & (means.component == "aircraft")
            & (means.case.isin(["neutral", "beta-10", "beta-5", "beta5", "beta10"]))
        ].copy()
        x["beta"] = x.case.map(cases.beta)
        x = x.sort_values("beta")
        ax.plot(x.beta, x[col], "-o", label=model)
        ax.set(
            xlabel="Sideslip beta (deg)",
            ylabel=col + (" (N)" if col == "Fy" else " (Nm)"),
            title="4 Hz, alpha=0, twist=35 deg",
        )
    x = trim[
        (trim.model == model)
        & (trim.twist == 35)
        & (trim.component == "aircraft")
        & (trim.case.isin(["neutral", "beta-10", "beta-5", "beta5", "beta10"]))
    ].copy()
    x["beta"] = x.case.map(cases.beta)
    x = x.sort_values("beta")
    axes[1, 0].plot(x.beta, x.Mx, "-o", label=model)
    x = phase[(phase.model == model) & (phase.twist == 35) & (phase.case == "beta10") & (phase.component == "wings")]
    axes[1, 1].plot(x.phase, x.Mx, label=model)
axes[1, 0].set(
    xlabel="Sideslip beta (deg)", ylabel="Aircraft Mx (Nm)", title="Shared longitudinal candidate, alpha=9.30 deg"
)
axes[1, 1].set(xlabel="Flap phase (deg)", ylabel="Main-wing Mx (Nm)", title="Shared candidate, beta=10 deg")
for ax in axes.flat:
    ax.grid(alpha=0.3)
    ax.legend()
fig.savefig(out / "mechanisms.png", dpi=150)
plt.close(fig)
for source, name in [("ll_final", "zero_incidence"), ("system_trim_compare", "trim")]:
    d = pd.read_csv(base / source / "derivatives.csv")
    d[(d.component == "aircraft") & d.derivative.isin(["beta", "p", "r"])].to_csv(
        out / (name + "_derivatives.csv"), index=False
    )
for model in ["L3D", "I3D"]:
    x = phase[(phase.model == model) & (phase.twist == 35) & (phase.case == "beta10") & (phase.component == "wings")]
    print(model, "phase Mx min,max,std", x.Mx.min(), x.Mx.max(), x.Mx.std(ddof=0))
