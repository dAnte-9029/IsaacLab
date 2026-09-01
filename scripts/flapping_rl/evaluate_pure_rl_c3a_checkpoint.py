"""Evaluate C3a/C3b/C3c checkpoints on frozen current and retention suites.

Each checkpoint/suite pair runs in its own fresh CPU-native Isaac process. The
script does not watch a training directory: it evaluates the explicitly named
checkpoint files, writes combined JSON/CSV summaries, and exits.

Example:
  ./isaaclab.sh -p scripts/flapping_rl/evaluate_pure_rl_c3a_checkpoint.py \
    --checkpoint <RUN>/model_75.pt <RUN>/model_100.pt \
    --output-dir <RUN>/eval_authority_75_100 \
    --actor-hidden-dims 512 256
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence


_NATIVE_EXTENSION_ID = "omni.flapping_bot.holonomic_constraint"
_C1_TASK = "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-Direct-v0"
_C2C_TASK = "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C2c-Direct-v0"
_C3A_TASK = "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C3a-Direct-v0"
_C3B_TASK = "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C3b-Direct-v0"
_C3C_TASK = "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C3c-Direct-v0"


@dataclass(frozen=True)
class EvaluationSuite:
    """One frozen authority suite executed in a fresh process."""

    name: str
    task: str
    eval_suite: str
    evaluation_contract: str
    case_count: int
    gate_field: str
    require_finite_metrics: bool = True


EVALUATION_SUITES: tuple[EvaluationSuite, ...] = (
    EvaluationSuite(
        name="c3a",
        task=_C3A_TASK,
        eval_suite="pure_rl_spatial_c3a_v2",
        evaluation_contract="pure_rl_spatial_c3a_v2",
        case_count=96,
        gate_field="promotion_gate_passed",
    ),
    EvaluationSuite(
        name="c1",
        task=_C1_TASK,
        eval_suite="pure_rl_curriculum1_nowind_v2",
        evaluation_contract="pure_rl_curriculum1_v2",
        case_count=16,
        gate_field="success_gate_passed",
        require_finite_metrics=False,
    ),
    EvaluationSuite(
        name="c2c",
        task=_C2C_TASK,
        eval_suite="pure_rl_longitudinal_c2c_v2",
        evaluation_contract="pure_rl_longitudinal_c2c_v2",
        case_count=112,
        gate_field="promotion_gate_passed",
    ),
)


def _evaluation_suites(spatial_stage: str) -> tuple[EvaluationSuite, ...]:
    """Return the frozen current-stage and retention suites in execution order."""

    if spatial_stage == "c3a":
        return EVALUATION_SUITES
    if spatial_stage not in ("c3b", "c3c"):
        raise ValueError(f"Unsupported spatial evaluation stage: {spatial_stage!r}.")
    c3b_suite = EvaluationSuite(
        name="c3b",
        task=_C3B_TASK,
        eval_suite="pure_rl_spatial_c3b_v3",
        evaluation_contract="pure_rl_spatial_c3b_v3",
        case_count=176,
        gate_field="promotion_gate_passed",
    )
    retention_suites = (c3b_suite, EVALUATION_SUITES[0], EVALUATION_SUITES[1], EVALUATION_SUITES[2])
    if spatial_stage == "c3b":
        return retention_suites
    return (
        EvaluationSuite(
            name="c3c",
            task=_C3C_TASK,
            eval_suite="pure_rl_spatial_c3c_v2",
            evaluation_contract="pure_rl_spatial_c3c_v2",
            case_count=96,
            gate_field="promotion_gate_passed",
        ),
        *retention_suites,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        nargs="+",
        required=True,
        help="One or more exact model_<iteration>.pt files to evaluate.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--spatial-stage",
        choices=("c3a", "c3b", "c3c"),
        default="c3a",
        help="Current spatial stage; later stages add every earlier frozen spatial suite.",
    )
    parser.add_argument(
        "--actor-hidden-dims",
        type=int,
        nargs="+",
        default=None,
        help="Explicit actor hidden dimensions forwarded to every fresh evaluator.",
    )
    parser.add_argument(
        "--pure-rl-split-frequency-actor",
        action="store_true",
        help="Evaluate checkpoints that use the independent frequency/tail actor.",
    )
    parser.add_argument(
        "--pure-rl-heading-canonical-observation",
        action="store_true",
        help="Evaluate with route-heading-canonical PureRL attitude observations.",
    )
    parser.add_argument(
        "--native-extension-parent",
        type=Path,
        default=None,
        help="Parent directory containing omni.flapping_bot.holonomic_constraint.",
    )
    parser.add_argument(
        "--portable-root-base",
        type=Path,
        default=None,
        help="Base portable root; defaults inside --output-dir.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse a complete existing per-suite JSON instead of launching that suite again.",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Isaac without a window (default: true).",
    )
    return parser.parse_args(argv)


def _checkpoint_iteration(checkpoint: Path) -> int:
    name = checkpoint.name
    if not name.startswith("model_") or not name.endswith(".pt"):
        raise ValueError(f"Checkpoint must be named model_<iteration>.pt: {checkpoint}")
    raw_iteration = name[len("model_") : -len(".pt")]
    if not raw_iteration.isdigit():
        raise ValueError(f"Checkpoint must be named model_<iteration>.pt: {checkpoint}")
    return int(raw_iteration)


def _child_environment(repo_root: Path) -> dict[str, str]:
    child_env = os.environ.copy()
    local_sources = [
        str((repo_root / "source/flapping_bot").resolve()),
        str((repo_root / "source/isaaclab_assets").resolve()),
        str((repo_root / "source/isaaclab_tasks").resolve()),
    ]
    existing_pythonpath = child_env.get("PYTHONPATH", "")
    if existing_pythonpath:
        local_sources.append(existing_pythonpath)
    child_env["PYTHONPATH"] = os.pathsep.join(local_sources)
    return child_env


def _build_eval_command(
    *,
    repo_root: Path,
    checkpoint: Path,
    suite: EvaluationSuite,
    suite_output_dir: Path,
    portable_root: Path,
    native_extension_parent: Path,
    headless: bool,
    actor_hidden_dims: Sequence[int] | None = None,
    split_frequency_actor: bool = False,
    heading_canonical_observation: bool = False,
) -> list[str]:
    """Build one fresh-process frozen-suite command."""

    robot_asset = (
        repo_root
        / "source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf"
    ).resolve()
    robot_usd_dir = (suite_output_dir / "generated_assets/flap_robot_552").resolve()
    kit_args = (
        f"--portable-root {portable_root.resolve()}"
        " --/apps/extensions/fsWatcherEnabled=false"
        f" --ext-folder {(repo_root / 'source').resolve()}"
        f" --ext-folder {native_extension_parent.resolve()}"
        f" --enable {_NATIVE_EXTENSION_ID}"
    )
    command = [
        sys.executable,
        str((repo_root / "scripts/flapping_rl/watch_and_eval.py").resolve()),
        "--task",
        suite.task,
        "--log_dir",
        str(checkpoint.parent),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(suite_output_dir),
        "--once",
        "--no-best-artifacts",
        "--no_saved_cfg",
        "--device",
        "cpu",
        "--num_envs",
        str(suite.case_count),
        "--episodes",
        str(suite.case_count),
        "--eval_suite",
        suite.eval_suite,
        "--robot-asset-path",
        str(robot_asset),
        "--robot-usd-dir",
        str(robot_usd_dir),
        "--kit_args",
        kit_args,
    ]
    if actor_hidden_dims is not None:
        command.append("--actor-hidden-dims")
        command.extend(str(value) for value in actor_hidden_dims)
    if split_frequency_actor:
        command.append("--pure-rl-split-frequency-actor")
    if heading_canonical_observation:
        command.append("--pure-rl-heading-canonical-observation")
    if headless:
        command.append("--headless")
    return command


def _parse_gate(value: object, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true"}:
            return True
        if normalized in {"0", "false"}:
            return False
    raise ValueError(f"{name} must be boolean or 0/1.")


def _load_suite_row(
    *,
    result_json: Path,
    checkpoint: Path,
    suite: EvaluationSuite,
) -> dict[str, object]:
    if not result_json.is_file():
        raise FileNotFoundError(f"Evaluation did not write result JSON: {result_json}")
    rows = json.loads(result_json.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Evaluation result must be a JSON list: {result_json}")
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and str(row.get("case", "")) == "suite"
        and str(row.get("evaluation_contract", "")) == suite.evaluation_contract
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {suite.evaluation_contract} suite row; found {len(matches)}."
        )
    row = dict(matches[0])
    evaluated_checkpoint = Path(str(row.get("checkpoint", ""))).expanduser().resolve()
    if evaluated_checkpoint != checkpoint:
        raise ValueError(
            f"Evaluation checkpoint mismatch: expected {checkpoint}, received {evaluated_checkpoint}."
        )
    finite = (
        _parse_gate(row.get("finite_metrics"), name="finite_metrics")
        if suite.require_finite_metrics
        else True
    )
    gate = _parse_gate(row.get(suite.gate_field), name=suite.gate_field)
    row["hard_gate_passed"] = bool(finite and gate)
    return row


def _checkpoint_summary(
    *,
    checkpoint: Path,
    suite_rows: Mapping[str, Mapping[str, object]],
    suites: Sequence[EvaluationSuite] = EVALUATION_SUITES,
) -> dict[str, object]:
    required = {suite.name for suite in suites}
    if set(suite_rows) != required:
        raise ValueError(
            f"Checkpoint summary requires suites {sorted(required)}; received {sorted(suite_rows)}."
        )
    suite_passed = {
        name: _parse_gate(row.get("hard_gate_passed"), name=f"{name}.hard_gate_passed")
        for name, row in suite_rows.items()
    }
    return {
        "checkpoint": str(checkpoint),
        "ppo_iteration": _checkpoint_iteration(checkpoint),
        "suite_passed": suite_passed,
        "all_hard_gates_passed": all(suite_passed.values()),
        "suites": {name: dict(row) for name, row in suite_rows.items()},
    }


def _flat_summary_row(result: Mapping[str, object]) -> dict[str, object]:
    suites = result["suites"]
    assert isinstance(suites, Mapping)
    current_stage = "c3c" if "c3c" in suites else ("c3b" if "c3b" in suites else "c3a")
    current = suites[current_stage]
    c3a = suites["c3a"]
    c1 = suites["c1"]
    c2c = suites["c2c"]
    assert isinstance(c3a, Mapping) and isinstance(c1, Mapping) and isinstance(c2c, Mapping)
    suite_passed = result["suite_passed"]
    assert isinstance(suite_passed, Mapping)
    row = {
        "checkpoint": result["checkpoint"],
        "ppo_iteration": result["ppo_iteration"],
        "current_stage": current_stage,
        f"{current_stage}_passed": int(bool(suite_passed[current_stage])),
        "c3a_passed": int(bool(suite_passed["c3a"])),
        "c1_passed": int(bool(suite_passed["c1"])),
        "c2c_passed": int(bool(suite_passed["c2c"])),
        "all_hard_gates_passed": int(bool(result["all_hard_gates_passed"])),
        f"{current_stage}_overall_success_rate": current.get("overall_success_rate", ""),
        f"{current_stage}_overall_survival_rate": current.get("overall_survival_rate", ""),
        "c1_timeout_rate": c1.get("timeout_rate", ""),
        "c1_termination_rate": c1.get("termination_rate", ""),
        "c1_tail_limit_fraction": c1.get("tail_limit_fraction", ""),
        "c2c_overall_survival_rate": c2c.get("overall_survival_rate", ""),
        "c2c_climb_success_rate": c2c.get("climb_success_rate", ""),
        "c2c_descent_success_rate": c2c.get("descent_success_rate", ""),
        "c2c_recovery_reached_rate": c2c.get("recovery_reached_rate", ""),
    }
    if current_stage in ("c3b", "c3c"):
        row["c3a_overall_success_rate"] = c3a.get("overall_success_rate", "")
        row["c3a_overall_survival_rate"] = c3a.get("overall_survival_rate", "")
    if current_stage == "c3c":
        c3b = suites["c3b"]
        assert isinstance(c3b, Mapping)
        row["c3b_passed"] = int(bool(suite_passed["c3b"]))
        row["c3b_overall_success_rate"] = c3b.get("overall_success_rate", "")
        row["c3b_overall_survival_rate"] = c3b.get("overall_survival_rate", "")
    return row


def _write_combined_summaries(output_dir: Path, results: Sequence[Mapping[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoint_evaluation.json").write_text(
        json.dumps(list(results), indent=2) + "\n",
        encoding="utf-8",
    )
    flat_rows = [_flat_summary_row(result) for result in results]
    with (output_dir / "checkpoint_evaluation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    evaluation_suites = _evaluation_suites(str(args.spatial_stage))
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir.expanduser().resolve()
    extension_parent = (
        args.native_extension_parent.expanduser().resolve()
        if args.native_extension_parent is not None
        else (repo_root / "source/flapping_bot/native_extensions").resolve()
    )
    extension_binary = (
        extension_parent
        / _NATIVE_EXTENSION_ID
        / "omni/flapping_bot/holonomic_constraint/_native.so"
    )
    if not extension_binary.is_file():
        raise FileNotFoundError(f"Native extension binary does not exist: {extension_binary}")
    robot_asset = (
        repo_root
        / "source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf"
    )
    if not robot_asset.is_file():
        raise FileNotFoundError(f"Measured PureRL robot asset does not exist: {robot_asset}")
    portable_root_base = (
        args.portable_root_base.expanduser().resolve()
        if args.portable_root_base is not None
        else output_dir / "portable"
    )

    checkpoints = [checkpoint.expanduser().resolve() for checkpoint in args.checkpoint]
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        _checkpoint_iteration(checkpoint)
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError("--checkpoint contains duplicate files.")
    actor_hidden_dims = None
    if args.actor_hidden_dims is not None:
        actor_hidden_dims = tuple(int(value) for value in args.actor_hidden_dims)
        if not actor_hidden_dims or any(value <= 0 for value in actor_hidden_dims):
            raise ValueError("--actor-hidden-dims values must be positive integers.")

    child_env = _child_environment(repo_root)
    results: list[dict[str, object]] = []
    for checkpoint in checkpoints:
        suite_rows: dict[str, dict[str, object]] = {}
        checkpoint_output = output_dir / checkpoint.stem
        for suite in evaluation_suites:
            suite_output = checkpoint_output / suite.name
            result_json = suite_output / f"{checkpoint.stem}.json"
            if result_json.exists() and not bool(args.reuse_existing):
                raise FileExistsError(
                    f"Evaluation output already exists; choose a new --output-dir or use --reuse-existing: "
                    f"{result_json}"
                )
            if not result_json.exists():
                suite_output.mkdir(parents=True, exist_ok=True)
                command = _build_eval_command(
                    repo_root=repo_root,
                    checkpoint=checkpoint,
                    suite=suite,
                    suite_output_dir=suite_output,
                    portable_root=portable_root_base / checkpoint.stem / suite.name,
                    native_extension_parent=extension_parent,
                    headless=bool(args.headless),
                    actor_hidden_dims=actor_hidden_dims,
                    split_frequency_actor=bool(args.pure_rl_split_frequency_actor),
                    heading_canonical_observation=bool(
                        args.pure_rl_heading_canonical_observation
                    ),
                )
                print(f"[INFO] Evaluating {checkpoint.name} on {suite.name} in a fresh process:", flush=True)
                print(" ", " ".join(command), flush=True)
                completed = subprocess.run(command, env=child_env, check=False)
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"{suite.name} evaluation failed for {checkpoint} with return code "
                        f"{completed.returncode}."
                    )
            suite_rows[suite.name] = _load_suite_row(
                result_json=result_json,
                checkpoint=checkpoint,
                suite=suite,
            )
        result = _checkpoint_summary(
            checkpoint=checkpoint,
            suite_rows=suite_rows,
            suites=evaluation_suites,
        )
        results.append(result)
        (checkpoint_output / "checkpoint_summary.json").write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )

    _write_combined_summaries(output_dir, results)
    all_passed = all(bool(result["all_hard_gates_passed"]) for result in results)
    print(
        "[OK] Frozen evaluation complete:",
        {
            "output_dir": str(output_dir),
            "checkpoint_count": len(results),
            "all_hard_gates_passed": all_passed,
        },
        flush=True,
    )
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
