from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/flapping_rl/pure_rl_backend_eval_worker.py"
SPEC = importlib.util.spec_from_file_location("pure_rl_backend_eval_worker", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
worker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = worker
SPEC.loader.exec_module(worker)


CPU_C1 = "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-Direct-v0"
CPU_C2A = "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C2a-Direct-v0"
GPU_C1 = "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-GpuImplicit-Direct-v0"
GPU_C2A = "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C2a-GpuImplicit-Direct-v0"
GPU_PHASE_C1 = "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-GpuPhaseMatched-Direct-v0"
GPU_PHASE_C2A = "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C2a-GpuPhaseMatched-Direct-v0"


def test_worker_adds_current_checkout_as_a_kit_extension_root(tmp_path: Path) -> None:
    source_root = tmp_path / "source"

    result = worker.with_local_extension_root("--portable-root /tmp/kit", source_root)
    repeated = worker.with_local_extension_root(result, source_root)

    assert result == f"--portable-root /tmp/kit --ext-folder {source_root}"
    assert repeated == result


def test_cpu_worker_enables_the_native_constraint_extension_once(tmp_path: Path) -> None:
    native_extensions_root = tmp_path / "native_extensions"

    result = worker.with_native_constraint_extension(
        "--portable-root /tmp/kit",
        native_extensions_root,
    )
    repeated = worker.with_native_constraint_extension(result, native_extensions_root)

    expected = (
        f"--portable-root /tmp/kit --ext-folder {native_extensions_root} "
        "--enable omni.flapping_bot.holonomic_constraint"
    )
    assert result == expected
    assert repeated == expected


def test_policy_observation_tensor_resolves_tensordict_policy_group() -> None:
    policy = np.zeros((2, 555), dtype=np.float32)

    assert worker.policy_observation_tensor({"policy": policy}) is policy
    assert worker.policy_observation_tensor(policy) is policy


def test_tensor_value_is_a_snapshot_of_cpu_tensor_storage() -> None:
    import torch

    source = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

    snapshot = worker._tensor_value(source, 0)
    source[0, 0] = 9.0

    np.testing.assert_array_equal(snapshot, np.array([1.0, 2.0], dtype=np.float32))


def test_worker_redirects_generated_robot_assets_out_of_source_tree(tmp_path: Path) -> None:
    cfg = SimpleNamespace(robot=SimpleNamespace(spawn=SimpleNamespace(usd_dir="source/assets")))

    worker.configure_generated_assets(cfg, tmp_path)

    assert cfg.robot.spawn.usd_dir == str(tmp_path / "generated_assets")


def test_worker_sets_an_explicit_reproducible_environment_seed() -> None:
    cfg = SimpleNamespace(seed=None)

    worker.configure_evaluation_seed(cfg, seed=7)

    assert cfg.seed == 7

    with pytest.raises(ValueError, match="nonnegative"):
        worker.configure_evaluation_seed(cfg, seed=-1)


def test_registered_case_resolution_reuses_the_exact_c1_and_c2a_grids() -> None:
    c1_cases = worker.resolve_registered_cases(GPU_C1)
    c2a_cases = worker.resolve_registered_cases(GPU_C2A)

    assert len(c1_cases) == 16
    assert c1_cases[0].case_id == "c1_h0_p0"
    assert c1_cases[-1].case_id == "c1_h3_p3"
    assert {case.task for case in c1_cases} == {"level"}
    assert len(c2a_cases) == 80
    assert c2a_cases[0].case_id == "c2a_slope_+00.0_h0_p0_promotion"
    assert {case.task for case in c2a_cases} == {"level", "climb", "descent"}

    selected = worker.resolve_registered_cases(GPU_C2A, case_ids=(c2a_cases[4].case_id, c2a_cases[1].case_id))
    assert tuple(case.case_id for case in selected) == (c2a_cases[4].case_id, c2a_cases[1].case_id)

    with pytest.raises(ValueError, match="Unknown case IDs"):
        worker.resolve_registered_cases(GPU_C1, case_ids=("missing",))

    assert worker.resolve_registered_cases(GPU_PHASE_C1)[0].case_id == "c1_h0_p0"
    assert worker.resolve_registered_cases(GPU_PHASE_C2A)[0].stage_id == "c2a"


def test_worker_request_fails_closed_on_backend_or_replay_contract_mismatch(tmp_path: Path) -> None:
    worker.validate_worker_request(
        task=GPU_C1,
        backend_id="gpu_implicit_candidate",
        rollout_mode="closed_loop",
        action_dir=None,
    )
    worker.validate_worker_request(
        task=CPU_C2A,
        backend_id="cpu_native_authority",
        rollout_mode="closed_loop",
        action_dir=None,
    )

    with pytest.raises(ValueError, match="backend"):
        worker.validate_worker_request(
            task=GPU_C1,
            backend_id="cpu_native_authority",
            rollout_mode="closed_loop",
            action_dir=None,
        )
    with pytest.raises(ValueError, match="action_dir"):
        worker.validate_worker_request(
            task=GPU_C1,
            backend_id="gpu_implicit_candidate",
            rollout_mode="action_replay",
            action_dir=None,
        )
    with pytest.raises(ValueError, match="closed_loop or action_replay"):
        worker.validate_worker_request(
            task=GPU_C1,
            backend_id="gpu_implicit_candidate",
            rollout_mode="other",
            action_dir=tmp_path,
        )


def test_action_sequence_round_trip_requires_four_finite_channels(tmp_path: Path) -> None:
    path = tmp_path / "case.npz"
    actions = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.savez_compressed(path, actions=actions)

    np.testing.assert_array_equal(worker.load_action_sequence(path), actions)

    np.savez_compressed(path, actions=np.zeros((3, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="four channels"):
        worker.load_action_sequence(path)
    np.savez_compressed(path, actions=np.array([[0.0, 0.0, 0.0, np.nan]], dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        worker.load_action_sequence(path)


def test_stack_trace_samples_preserves_policy_step_alignment() -> None:
    samples = [
        {
            "time_s": 0.0,
            "actions": np.zeros(4),
            "done": False,
            "terminated": False,
            "time_out": False,
            "physical_state_valid": True,
            "height_error_m": 0.1,
        },
        {
            "time_s": 1.0 / 60.0,
            "actions": np.ones(4),
            "done": False,
            "terminated": False,
            "time_out": False,
            "physical_state_valid": True,
            "height_error_m": 0.2,
        },
    ]

    trace = worker.stack_trace_samples(samples)

    assert trace["time_s"].shape == (2,)
    assert trace["actions"].shape == (2, 4)
    np.testing.assert_allclose(trace["height_error_m"], np.array([0.1, 0.2]))

    with pytest.raises(ValueError, match="same fields"):
        worker.stack_trace_samples([samples[0], {"time_s": 0.1, "actions": np.zeros(4)}])


def test_physical_trace_excludes_done_steps_that_have_already_auto_reset() -> None:
    assert worker.physical_trace_sample_is_valid(done=False)
    assert not worker.physical_trace_sample_is_valid(done=True)
