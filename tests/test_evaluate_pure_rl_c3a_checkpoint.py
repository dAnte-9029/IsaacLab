from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/flapping_rl/evaluate_pure_rl_c3a_checkpoint.py"
)
SPEC = importlib.util.spec_from_file_location("evaluate_pure_rl_c3a_checkpoint", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
evaluate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluate
SPEC.loader.exec_module(evaluate)


def test_build_eval_command_is_one_exact_cpu_native_suite(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    checkpoint = tmp_path / "run" / "model_100.pt"
    output_dir = tmp_path / "output" / "model_100" / "c2c"
    extension_parent = repo_root / "source/flapping_bot/native_extensions"
    suite = next(item for item in evaluate.EVALUATION_SUITES if item.name == "c2c")

    command = evaluate._build_eval_command(
        repo_root=repo_root,
        checkpoint=checkpoint,
        suite=suite,
        suite_output_dir=output_dir,
        portable_root=tmp_path / "portable/c2c",
        native_extension_parent=extension_parent,
        headless=True,
        actor_hidden_dims=(512, 256),
        split_frequency_actor=True,
        heading_canonical_observation=True,
    )

    assert command[0] == sys.executable
    assert command[command.index("--checkpoint") + 1] == str(checkpoint)
    assert command[command.index("--output-dir") + 1] == str(output_dir)
    assert command[command.index("--task") + 1] == suite.task
    assert command[command.index("--eval_suite") + 1] == "pure_rl_longitudinal_c2c_v2"
    assert command[command.index("--device") + 1] == "cpu"
    assert command[command.index("--num_envs") + 1] == "112"
    assert "--once" in command
    assert "--no-best-artifacts" in command
    assert "--no_saved_cfg" in command
    assert "--headless" in command
    actor_dims_index = command.index("--actor-hidden-dims")
    assert command[actor_dims_index + 1 : actor_dims_index + 3] == ["512", "256"]
    assert "--pure-rl-split-frequency-actor" in command
    assert "--pure-rl-heading-canonical-observation" in command
    kit_args = command[command.index("--kit_args") + 1]
    assert "--enable omni.flapping_bot.holonomic_constraint" in kit_args


def test_load_suite_row_requires_matching_contract_checkpoint_and_gate(tmp_path: Path) -> None:
    checkpoint = (tmp_path / "model_75.pt").resolve()
    checkpoint.write_bytes(b"checkpoint")
    suite = next(item for item in evaluate.EVALUATION_SUITES if item.name == "c3a")
    result_json = tmp_path / "model_75.json"
    result_json.write_text(
        json.dumps(
            [
                {
                    "checkpoint": str(checkpoint),
                    "case": "suite",
                    "evaluation_contract": suite.evaluation_contract,
                    "finite_metrics": True,
                    "promotion_gate_passed": 1,
                    "overall_success_rate": 1.0,
                }
            ]
        ),
        encoding="utf-8",
    )

    row = evaluate._load_suite_row(
        result_json=result_json,
        checkpoint=checkpoint,
        suite=suite,
    )

    assert row["hard_gate_passed"] is True


def test_load_c1_suite_row_uses_registered_gate_without_finite_field(tmp_path: Path) -> None:
    checkpoint = (tmp_path / "model_75.pt").resolve()
    checkpoint.write_bytes(b"checkpoint")
    suite = next(item for item in evaluate.EVALUATION_SUITES if item.name == "c1")
    result_json = tmp_path / "model_75.json"
    result_json.write_text(
        json.dumps(
            [
                {
                    "checkpoint": str(checkpoint),
                    "case": "suite",
                    "evaluation_contract": suite.evaluation_contract,
                    "success_gate_passed": 1,
                    "tail_limit_fraction": 0.09,
                }
            ]
        ),
        encoding="utf-8",
    )

    row = evaluate._load_suite_row(
        result_json=result_json,
        checkpoint=checkpoint,
        suite=suite,
    )

    assert row["hard_gate_passed"] is True


def test_checkpoint_summary_fails_closed_when_one_suite_fails(tmp_path: Path) -> None:
    checkpoint = (tmp_path / "model_100.pt").resolve()
    rows = {
        "c3a": {"hard_gate_passed": True},
        "c1": {"hard_gate_passed": True},
        "c2c": {"hard_gate_passed": False},
    }

    summary = evaluate._checkpoint_summary(checkpoint=checkpoint, suite_rows=rows)

    assert summary["ppo_iteration"] == 100
    assert summary["suite_passed"] == {"c3a": True, "c1": True, "c2c": False}
    assert summary["all_hard_gates_passed"] is False


def test_c3b_suite_adds_current_stage_before_all_retention_suites() -> None:
    suites = evaluate._evaluation_suites("c3b")

    assert [suite.name for suite in suites] == ["c3b", "c3a", "c1", "c2c"]
    assert suites[0].evaluation_contract == "pure_rl_spatial_c3b_v3"
    assert suites[0].case_count == 176


def test_c3c_suite_adds_c3b_and_all_earlier_retention_suites() -> None:
    suites = evaluate._evaluation_suites("c3c")

    assert [suite.name for suite in suites] == ["c3c", "c3b", "c3a", "c1", "c2c"]
    assert suites[0].evaluation_contract == "pure_rl_spatial_c3c_v2"
    assert suites[0].case_count == 96


def test_checkpoint_iteration_rejects_ambiguous_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="model_<iteration>"):
        evaluate._checkpoint_iteration(tmp_path / "best_model.pt")
