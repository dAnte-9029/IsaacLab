"""Frozen artifact analysis for the historical DeLaurier CPU-native policies.

No optimizer is constructed, no backward pass is performed, and no environment
is imported. Inputs are explicit development/training artifacts only.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn

from flapping_bot.px4_like.pure_rl_split_actor import PureRLSplitActor

CHANNELS = ["frequency", "rudder", "left_elevon", "right_elevon", "symmetric", "differential"]
RUNS = Path("logs/rsl_rl/flapping_bot_straight_flight")
PATHS = {
    "c1": RUNS / "2026-08-07_05-05-44_curriculum1_overnight_seed2/model_1300.pt",
    "c2": RUNS / "2026-08-12_17-12-48_pure_rl_c2c_seed0_resume500_to550/model_550.pt",
    "c3a": RUNS / "2026-08-29_10-07-47_pure_rl_c3a_split_frequency_actor_seed0_201iter/model_200.pt",
    "c3b": RUNS / "2026-08-31_17-32-50_pure_rl_c3b_headingcanonical_from100_seed0_101iter/model_75.pt",
    "full_c3": RUNS / "2026-09-01_10-13-43_pure_rl_c3_full_joint_from_c2c_seed0_301iter/model_175.pt",
    "c3c_continuation": RUNS / "2026-08-31_19-52-40_pure_rl_c3c_joint_headingcanonical_from_c3b75_seed0_101iter/model_100.pt",
    "bounded_475": RUNS / "2026-08-14_15-51-03_pure_rl_c3a_bounded_warm_start_c2c_seed0_resume275_to500/model_475.pt",
    "distill_100": RUNS / "2026-08-26_19-28-45_pure_rl_c3a_task_aware_distill005_seed0_101iter/model_100.pt",
    "large_100": RUNS / "2026-08-26_20-38-23_pure_rl_c3a_taskaware_distill005_largeactor512x256_seed0_101iter/model_100.pt",
}
TASK_RUN = RUNS / "2026-08-18_10-41-16_pure_rl_c3a_task_aware_seed0_100iter"
for _index in (0, 25, 50, 75, 99):
    PATHS[f"task_{_index}"] = TASK_RUN / f"model_{_index}.pt"
PILOT = Path("logs/continual_c4/20260922_pilot")
for _index in (0, 25, 50, 75, 100):
    PATHS[f"D{_index}"] = PILOT / "D/training" / f"model_{_index}.pt"


def axes(action: np.ndarray) -> np.ndarray:
    """Append symmetric/differential coordinates without discarding actuators."""
    return np.column_stack((action, (action[:, 2] + action[:, 3]) / 2,
                            (action[:, 2] - action[:, 3]) / 2))


def statistics(values: np.ndarray) -> dict:
    if not len(values):
        return {"n": 0}
    assert np.isfinite(values).all()
    return {"n": len(values), "mean": values.mean(0).tolist(),
            "rms": np.sqrt(np.mean(values ** 2, axis=0)).tolist(),
            "p95_abs": np.quantile(abs(values), .95, axis=0).tolist()}


def load_actor(path: Path) -> tuple:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    actor_state = {k[6:]: v for k, v in state.items() if k.startswith("actor.")}
    split = any(k.startswith("frequency_actor") for k in actor_state)
    if split:
        actor = PureRLSplitActor(555, [256, 128], "elu")
    else:
        widths = [actor_state[f"{i}.weight"].shape for i in (0, 2, 4)]
        assert widths[0][1] == 555 and widths[-1][0] == 4
        actor = nn.Sequential(nn.Linear(555, widths[0][0]), nn.ELU(),
                              nn.Linear(widths[0][0], widths[1][0]), nn.ELU(),
                              nn.Linear(widths[1][0], 4))
    actor.load_state_dict(actor_state, strict=True)
    actor.eval().requires_grad_(False)
    assert not any("normalizer" in k for k in state)
    opt = checkpoint.get("optimizer_state_dict", {})
    steps = [float(v["step"]) for v in opt.get("state", {}).values() if "step" in v]
    metadata = {"path": str(path), "class": "PureRLSplitActor" if split else "shared_ELU_MLP",
                "shapes": {k: list(v.shape) for k, v in actor_state.items()},
                "optimizer_entries": len(opt.get("state", {})),
                "optimizer_step_range": [min(steps), max(steps)] if steps else [],
                "std": state["log_std"].exp().tolist()}
    return actor, state["log_std"].exp().numpy(), metadata


@torch.inference_mode()
def infer(actor: nn.Module, obs: np.ndarray) -> np.ndarray:
    return np.concatenate([actor(torch.from_numpy(np.asarray(part, dtype=np.float32))).numpy()
                           for part in np.array_split(obs, max(1, len(obs) // 2048))])


def gaussian_kl(mean_a: np.ndarray, std_a: np.ndarray,
                mean_b: np.ndarray, std_b: np.ndarray) -> float:
    """KL(N(a)||N(b)), before command clipping; summed over four channels."""
    return float(np.mean(np.sum(np.log(std_b / std_a)
                                + (std_a ** 2 + (mean_a - mean_b) ** 2) / (2 * std_b ** 2) - .5,
                                axis=1)))


def features(obs: np.ndarray) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Physical features and declared tolerances; no repeated-history norm.

    Yaw is omitted so both world and heading-canonical records can be compared.
    Preview remains body-relative. Tolerances define descriptive neighborhoods,
    not a validated probability of behavior coverage.
    """
    history = obs[:, :420].reshape(-1, 30, 14)
    x = history[:, -1]
    w, qx, qy, qz = (x[:, i] for i in range(4))
    roll = np.arctan2(2*(w*qx+qy*qz), 1-2*(qx*qx+qy*qy))
    pitch = np.arcsin(np.clip(2*(w*qy-qz*qx), -1, 1))
    preview = obs[:, 540:].reshape(-1, 5, 3) * 5
    values = np.column_stack((np.rad2deg(roll), np.rad2deg(pitch),
                             x[:, 4:7] * [12, 5, 5], x[:, 7:10]*5,
                             (x[:, 11]+1)*2.5, preview[:, 0, 1:3], preview[:, -1, 1:3],
                             (x[:, 7:10]-history[:, 0, 7:10])*5))
    names = ["roll_deg", "pitch_deg", "u_mps", "v_mps", "w_mps", "p_rad_s", "q_rad_s", "r_rad_s",
             "frequency_hz", "near_y_m", "near_z_m", "far_y_m", "far_z_m", "delta_p", "delta_q", "delta_r"]
    scales = np.array([10, 10, 2, 1, 1, 1, 1, 1, .5, .5, .5, 1, 1, 1, 1, 1])
    return values, names, scales


def coverage(query: np.ndarray, memory: np.ndarray) -> dict:
    q, names, scales = features(query)
    m, _, _ = features(memory)
    distance, _ = cKDTree(m/scales).query(q/scales, p=np.inf, workers=1)
    return {"n": len(q), "feature_names": names, "tolerances": scales.tolist(),
            "fraction_within_tolerance": {str(k): float((distance <= k).mean()) for k in (.5, 1, 2)},
            "nearest_max_scaled_distance_quantiles": np.quantile(distance, [.5, .9, .99]).tolist(),
            "query_ranges": np.quantile(q, [0, .01, .5, .99, 1], axis=0).tolist(),
            "memory_ranges": np.quantile(m, [0, .01, .5, .99, 1], axis=0).tolist()}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def c4_analysis(root: Path, actors: dict, stds: dict, memory: dict, output: Path) -> dict:
    report = {"same_state": {}, "memory": {}, "suite_complementarity": {}}
    m = memory["observations"].numpy()
    families = memory["families"].numpy()
    target = memory["actions"].numpy()
    memory_predictions = {k: infer(actors[k], m) for k in ("full_c3", "D0", "D25", "D50", "D75", "D100")}
    report["memory_teacher_reconstruction_max"] = float(abs(memory_predictions["full_c3"]-target).max())
    assert report["memory_teacher_reconstruction_max"] < 1e-5
    for key, prediction in memory_predictions.items():
        report["memory"][key] = {str(f): statistics(axes(prediction[families == f]-target[families == f]))
                                  for f in range(6)}
    report["memory_coverage_counts"] = {}
    mf, _, _ = features(m)
    for f in range(6):
        a = mf[families == f]
        report["memory_coverage_counts"][str(f)] = {
            "n": len(a), "abs_pitch_above_45": int((abs(a[:,1])>45).sum()),
            "abs_pitch_above_60": int((abs(a[:,1])>60).sum()),
            "abs_roll_above_25": int((abs(a[:,0])>25).sum())}
    plot_rows = []
    for index in (50, 75):
        chunks, trace_chunks, names = [], {}, []
        masks_by_trace = []
        for path in sorted((root/PILOT/f"D/evaluation_{index}/c4/traces").glob("development_*.npz")):
            with np.load(path) as z:
                trace = {k: z[k] for k in z.files}
            chunks.append(trace["policy_input"])
            names.append(path.name)
            n = len(trace["time_s"])
            for k in trace:
                if k != "policy_input":
                    trace_chunks.setdefault(k, []).append(trace[k])
            error = np.maximum(abs(trace["cross_track_error_m"]), abs(trace["height_error_m"]))
            crossing = np.flatnonzero((error > .5) & (trace["time_s"] >= 2))
            pre = np.zeros(n, bool)
            if len(crossing):
                first = crossing[0]
                pre[max(0, first-30):first] = True
            masks_by_trace.append(pre)
        assert len(names) == 96
        obs = np.concatenate(chunks)
        trace = {k: np.concatenate(v) for k,v in trace_chunks.items()}
        predictions = {k: infer(actors[k], obs) for k in ("D0", "D25", "D50", "D75", "D100")}
        replay = float(abs(predictions[f"D{index}"]-trace["raw_action"]).max())
        assert replay < 1e-5
        delta = axes(predictions["D75"]-predictions["D50"])
        curvature, slope = trace["active_curvature_rad_per_m"], trace["active_slope_rad"]
        # Differences are computed per episode, never across reset boundaries.
        changes = np.concatenate([np.abs(np.gradient(v))*60 for v in trace_chunks["active_slope_rad"]])
        masks = {"all": np.ones(len(obs), bool), "first_second": trace["time_s"] <= 1,
                 "after_2s": trace["time_s"] >= 2,
                 "curvature_positive": curvature > .002, "curvature_negative": curvature < -.002,
                 "near_straight": abs(curvature) <= .002,
                 "climb": slope > np.deg2rad(1), "descent": slope < -np.deg2rad(1),
                 "slope_transition": changes > np.deg2rad(1),
                 "pre_first_0p5m_crossing": np.concatenate(masks_by_trace)}
        item = {"reconstruction_max": replay, "states": len(obs), "conditional_delta": {},
                "checkpoint_drift": {}, "coverage": {}, "actual_actions": {}}
        for label, mask in masks.items():
            item["conditional_delta"][label] = statistics(delta[mask])
        for left, right in (("D0", "D25"), ("D25", "D50"), ("D50", "D75"), ("D75", "D100"), ("D0", "D75"), ("D50", "D100")):
            item["checkpoint_drift"][f"{left}->{right}"] = {
                **statistics(axes(predictions[right]-predictions[left])),
                "gaussian_kl": gaussian_kl(predictions[left], stds[left], predictions[right], stds[right])}
        for label in ("all", "slope_transition", "pre_first_0p5m_crossing"):
            mask = masks[label]
            if mask.any():
                item["coverage"][label] = coverage(obs[mask][::10], m[families == 5])
        actual = np.column_stack([trace[k] for k in ("actual_flap_frequency_hz", "actual_rudder_rad",
                                                    "actual_left_elevon_rad", "actual_right_elevon_rad")])
        item["actual_actions"] = statistics(axes(actual))
        item["clipped_delta"] = statistics(axes(np.clip(predictions["D75"], -1, 1)-np.clip(predictions["D50"], -1, 1)))
        item["tail_limit_fraction"] = float(trace["tail_limit_active"].mean())
        item["initial_delta"] = statistics(delta[trace["time_s"] <= 1/60 + 1e-5])
        # Offline switching pressure: a predeclared sign-of-climb gate; not a controller evaluation.
        gate = slope > 0
        boundary = np.concatenate([np.r_[False, np.diff(v > 0) != 0] for v in trace_chunks["active_slope_rad"]])
        item["offline_slope_sign_switch"] = {"switches": int(boundary.sum()),
            "same_state_boundary_gap": statistics(delta[boundary]),
            "note": "Diagnostic gate uses reference slope; no closed-loop success is inferred."}
        report["same_state"][str(index)] = item
        for second in range(20):
            mask = (trace["time_s"] >= second) & (trace["time_s"] < second+1)
            if mask.any():
                plot_rows.append({"checkpoint": index, "second": second,
                    "horizontal_abs_m": float(abs(trace["cross_track_error_m"][mask]).mean()),
                    "vertical_abs_m": float(abs(trace["height_error_m"][mask]).mean()),
                    **{f"delta_{k}": float(delta[mask, j].mean()) for j,k in enumerate(CHANNELS)}})
    for index in (50,75,100):
        base = root/PILOT/f"D/evaluation_{index}"
        old = json.loads((base/"retention/checkpoint_evaluation.json").read_text())[0]
        c4 = json.loads((base/"c4/summary.json").read_text())
        report["suite_complementarity"][str(index)] = {
            "gates": {**old["suite_passed"], "c4": all(x["success"] for x in c4["cases"])},
            "c4_success": sum(x["success"] for x in c4["cases"]),
            "old_success": {k: {j:v for j,v in s.items() if j in ("overall_success_rate", "overall_survival_rate", "mean_abs_horizontal_error_m", "case_count")}
                            for k,s in old["suites"].items()}}
    with (output/"c4_time_bins.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=list(plot_rows[0])); writer.writeheader(); writer.writerows(plot_rows)
    return report


def c2_analysis(root: Path, actors: dict, stds: dict, memory: dict) -> dict:
    report = {"capture": {}, "historical_csv": {}}
    for name, own in (("baseline", "c2"), ("candidate", "task_75")):
        directory = "c2_capture_v2" if name == "baseline" else "c2_capture"
        base = root/"logs/rl_forgetting_mechanisms_v1"/directory/f"{name}_trace"
        if not base.with_suffix(".npz").exists():
            report["capture"][name] = {"available": False}; continue
        with np.load(base.with_suffix(".npz")) as z:
            obs, raw = z["policy_input"], z["raw_action"]
        rows = list(csv.DictReader(base.with_suffix(".csv").open()))
        assert len(rows)==len(obs)
        predictions = {k: infer(actors[k], obs) for k in ("c2", "task_0", "task_25", "task_50", "task_75", "task_99")}
        assert abs(predictions[own]-raw).max()<1e-5
        # Phase uses the PRE-action observation time and previous trace progress.
        progress = np.r_[0, [float(r["progress_m"]) for r in rows[:-1]]]
        tilt = np.rad2deg(np.arccos(np.clip(1-2*(obs[:,406+1]**2+obs[:,406+2]**2),-1,1)))
        times=np.arange(len(obs))/60
        recovered=np.flatnonzero(progress>=42.5)
        recovery_start=times[recovered[0]] if len(recovered) else np.inf
        masks={"entry":progress<17.5, "climb":(progress>=17.5)&(progress<42.5),
               "near_tilt_60deg":tilt>=60,
               "recovery":(times>=recovery_start)&(times<recovery_start+2),
               "post_recovery":times>=recovery_start+2,
               "last_half_second":np.arange(len(obs))>=len(obs)-30}
        item={"available":True,"n":len(obs),"same_state":{},"coverage":{},
              "phase_own_actions":{},"clipped_same_state":{},"tilt_max_deg":float(tilt.max()),
              "reconstruction_max":float(abs(predictions[own]-raw).max())}
        for key,mask in masks.items():
            item["same_state"][key]={k:statistics(axes(p-predictions["c2"])[mask]) for k,p in predictions.items()}
            item["phase_own_actions"][key]=statistics(axes(raw)[mask])
            item["clipped_same_state"][key]=statistics(axes(np.clip(predictions["task_75"],-1,1)-np.clip(predictions["c2"],-1,1))[mask])
            if mask.any():
                mem=memory["observations"][memory["families"]==1].numpy()
                item["coverage"][key]=coverage(obs[mask],mem)
        item["kl"]={k:gaussian_kl(predictions["c2"],stds["c2"],p,stds[k]) for k,p in predictions.items()}
        item["adjacent_kl"]={f"{a}->{b}":gaussian_kl(predictions[a],stds[a],predictions[b],stds[b])
                             for a,b in (("task_0","task_25"),("task_25","task_50"),("task_50","task_75"),("task_75","task_99"))}
        report["capture"][name]=item
    trace_paths=[root/RUNS/"2026-08-27_08-00-35_pure_rl_c3a_joint_from_c1_seed0_201iter/response_compare_model550_vs_model175_slope12_h0_p0"/f"{s}_trace.csv" for s in ("baseline","candidate")]
    trace_paths += [root/"eval/flip_diagnostic_shared_vs_split_model200_slope12_h0_p0"/f"{s}_trace.csv" for s in ("baseline","candidate")]
    for p in trace_paths:
        rows=list(csv.DictReader(p.open()))
        a=np.array([[float(r[f"requested_{k}_action"]) for k in ("flap_frequency","rudder","left_elevon","right_elevon")] for r in rows])
        report["historical_csv"][str(p.relative_to(root))]={phase:statistics(axes(a)[np.array([r["phase"]==phase for r in rows])]) for phase in ("entry","slope","recovery")}
    return report


def update_analysis(root: Path) -> dict:
    """Summarize existing probes without recomputing training gradients."""
    rows=list(csv.DictReader((root/TASK_RUN/"actor_gradient_conflict.csv").open()))
    selected=["gradient_probe/cos_c2c_strong_vs_c3a", "gradient_probe/post_kl_c2c_strong",
              "gradient_probe/post_kl_c3a", "gradient_probe/actor_update_norm",
              "gradient_probe/update_directional_derivative_c2c_strong"]
    task={}
    for key in selected:
        vals=np.array([float(r[key]) for r in rows if r.get(key)])
        task[key]={"n":len(vals),"mean":float(vals.mean()),"negative_fraction":float((vals<0).mean()),
                   "min":float(vals.min()),"max":float(vals.max())}
    drows=[json.loads(s) for s in (root/PILOT/"D/training/update_diagnostics.jsonl").read_text().splitlines()]
    chosen=[r for r in drows if 51<=r["iteration"]<=75]
    d={k:[r[k] for r in chosen if k in r] for k in chosen[0]
       if k.startswith(("cos_","tail_cos_","update_","critic_","post_","gradient_dot_update", "actor_update")) or "kl" in k}
    return {"task_aware":task,"D51_75":d}


def legacy_channel_analysis(root: Path) -> dict:
    """Reuse September 11 interventions; never launch another evaluation."""
    base=root/"logs/continuous_curve"
    actors={i:load_actor(base/"c4_joint_seed0_20260910"/f"model_{i}.pt")[0] for i in (25,50)}
    result={"note":"Original September 10 run, NOT retention arm D", "interventions":{}}
    for directory,names in (("20260911_c4_channel_swap",("25_25","25_50","50_25","50_50")),
                            ("20260911_c4_tail_channels",("rudder","left","right","elevons"))):
        for name in names:
            folder=base/directory/name
            launch=json.loads((folder/"launch.json").read_text())
            assert Path(launch["curves"]).name=="development.json"
            summary=json.loads((folder/"summary.json").read_text())
            assert summary["evaluation_complete"] and summary["case_count"]==96
            main_index=int(Path(launch["checkpoint"]).stem.split("_")[-1])
            tail_index=int(Path(launch["tail_checkpoint"]).stem.split("_")[-1])
            # The earlier all-tail CLI predates this manifest field; run.sh
            # and the evaluator default specify all three channels.
            channels=launch.get("replacement_tail_channels", [1,2,3])
            path=folder/"traces/development_0.npz"
            with np.load(path) as z:
                obs=z["policy_input"]; raw=z["raw_action"]
            expected=infer(actors[main_index],obs)
            expected[:,channels]=infer(actors[tail_index],obs)[:,channels]
            error=float(abs(raw-expected).max())
            assert error<1e-5
            result["interventions"][name]={"summary":str(folder/"summary.json"),
                "main_checkpoint":main_index,"tail_checkpoint":tail_index,"replaced_channels":channels,
                "success":sum(x["success"] for x in summary["cases"]),
                "means":summary["case_balanced_means"],"spotcheck_case":"development_0",
                "spotcheck_reconstruction_max":error}
    return result


def plot_results(output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows=list(csv.DictReader((output/"c4_time_bins.csv").open()))
    data=json.loads((output/"c4.json").read_text())
    fig,axs=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for index in (50,75):
        r=[x for x in rows if int(x["checkpoint"])==index]
        axs[0].plot([float(x["second"])+.5 for x in r],[float(x["vertical_abs_m"]) for x in r],label=f"D{index} vertical")
        axs[0].plot([float(x["second"])+.5 for x in r],[float(x["horizontal_abs_m"]) for x in r],ls="--",label=f"D{index} lateral")
    means=data["same_state"]["50"]["conditional_delta"]
    labels=["curvature_positive","curvature_negative","climb","descent","slope_transition"]
    for j,key in ((1,"rudder"),(4,"symmetric"),(5,"differential")):
        axs[1].plot(range(len(labels)),[means[k]["mean"][j] for k in labels],marker="o",label=key)
    axs[1].set_xticks(range(len(labels)),["turn +","turn -","climb","descent","transition"],rotation=25)
    for j,key in ((1,"rudder"),(4,"symmetric"),(5,"differential")):
        axs[2].plot([0,25,50,75,100],[data["memory"][f"D{i}"]["5"]["rms"][j] for i in (0,25,50,75,100)],marker="o",label=key)
    axs[0].set(xlabel="Time (s)",ylabel="Mean absolute tracking error (m)",title="Existing closed-loop traces")
    axs[1].set(ylabel="Normalized mean D75 - D50",title="Identical D50 observations")
    axs[2].set(xlabel="Checkpoint",ylabel="Normalized action RMS drift",title="C4 memory vs full-C3 teacher")
    for ax in axs: ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.savefig(output/"c4_mechanisms.png",dpi=170)
    plt.close(fig)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path.cwd())
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    actors,stds,inventory={},{},{}
    for key,path in PATHS.items():
        if not (args.root/path).exists():
            inventory[key]={"available":False,"path":str(path)}; continue
        actors[key],stds[key],inventory[key]=load_actor(args.root/path)
        inventory[key]["available"]=True
    memory=torch.load(args.root/"logs/continual_c4/20260922_memory/memory.pt",map_location="cpu",weights_only=True)
    assert memory["role"]=="training_only"
    write_json(args.output/"inventory.json",inventory)
    write_json(args.output/"c4.json",c4_analysis(args.root,actors,stds,memory,args.output))
    write_json(args.output/"c2.json",c2_analysis(args.root,actors,stds,memory))
    parameters={}
    for left,right in (("D50","D75"),("D0","D75"),("c2","task_75")):
        a,b=actors[left].state_dict(),actors[right].state_dict()
        parameters[f"{left}->{right}"]={k:{"delta_norm":float(torch.linalg.vector_norm(b[k]-v)),
             "relative_norm":float(torch.linalg.vector_norm(b[k]-v)/torch.linalg.vector_norm(v).clamp_min(1e-12))} for k,v in a.items()}
    write_json(args.output/"parameter_drift.json",parameters)
    write_json(args.output/"update_probes.json",update_analysis(args.root))
    write_json(args.output/"legacy_channels.json",legacy_channel_analysis(args.root))
    plot_results(args.output)
    print(f"Analysis complete: {args.output}")


if __name__ == "__main__":
    main()
