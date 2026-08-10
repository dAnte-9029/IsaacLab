from __future__ import annotations

import csv
import importlib.util
import sys
import threading
import time
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "flapping_rl" / "train_and_watch.py"
SPEC = importlib.util.spec_from_file_location("train_and_watch", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
train_and_watch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_and_watch)


class _FakeProcess:
    def __init__(self, return_code: int | None = None):
        self._return_code = return_code

    def poll(self):
        return self._return_code


def _write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["checkpoint", "case"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_latest_checkpoint_uses_numeric_sort(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "model_2.pt").write_text("")
    (run_dir / "model_10.pt").write_text("")

    latest = train_and_watch._latest_checkpoint(run_dir)

    assert latest == (run_dir / "model_10.pt").resolve()


def test_needs_final_eval_when_latest_suite_row_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True)
    model_0 = run_dir / "model_0.pt"
    model_1 = run_dir / "model_1.pt"
    model_0.write_text("")
    model_1.write_text("")
    _write_summary(
        eval_dir / "summary.csv",
        [
            {"checkpoint": str(model_0.resolve()), "case": "suite"},
            {"checkpoint": str(model_1.resolve()), "case": "calm"},
        ],
    )

    assert train_and_watch._needs_final_eval(run_dir) is True


def test_needs_final_eval_false_when_latest_suite_row_exists(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True)
    model_0 = run_dir / "model_0.pt"
    model_1 = run_dir / "model_1.pt"
    model_0.write_text("")
    model_1.write_text("")
    _write_summary(
        eval_dir / "summary.csv",
        [
            {"checkpoint": str(model_0.resolve()), "case": "suite"},
            {"checkpoint": str(model_1.resolve()), "case": "suite"},
        ],
    )

    assert train_and_watch._needs_final_eval(run_dir) is False


def test_wait_for_run_dir_succeeds_when_directory_appears(tmp_path: Path) -> None:
    run_dir = tmp_path / "delayed_run"

    def _create_dir() -> None:
        time.sleep(0.1)
        run_dir.mkdir()

    thread = threading.Thread(target=_create_dir, daemon=True)
    thread.start()

    train_and_watch._wait_for_run_dir(run_dir, _FakeProcess(return_code=None), timeout_s=1.0, poll_s=0.01)
    thread.join(timeout=1.0)

    assert run_dir.is_dir()


def test_wait_for_run_dir_raises_if_process_exits_first(tmp_path: Path) -> None:
    run_dir = tmp_path / "never_created"

    try:
        train_and_watch._wait_for_run_dir(run_dir, _FakeProcess(return_code=1), timeout_s=0.1, poll_s=0.01)
    except RuntimeError as exc:
        assert "Training exited before creating a run directory" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when process exits before run dir appears.")

def test_build_train_cmd_includes_resume_arguments() -> None:
    args = train_and_watch.argparse.Namespace(
        task="Isaac-FlappingBot-StraightFlight-DeLaurier-PureRL-Direct-v0",
        run_name="pure_resume",
        num_envs=128,
        eval_num_envs=8,
        max_iterations=400,
        save_interval=20,
        seed=7,
        train_device="cuda:0",
        eval_device="cuda:1",
        episodes=3,
        poll_s=30.0,
        run_dir_timeout_s=120.0,
        eval_suite="straight_standard",
        headless=True,
        resume=True,
        load_run="2026-03-07_19-36-58_task2_smoke_weak",
        checkpoint="best_model.pt",
    )

    cmd = train_and_watch._build_train_cmd(args)

    assert "--resume" in cmd
    assert cmd[cmd.index("--load_run") + 1] == "2026-03-07_19-36-58_task2_smoke_weak"
    assert cmd[cmd.index("--checkpoint") + 1] == "best_model.pt"


def test_build_train_cmd_supports_weights_only_warm_start() -> None:
    args = train_and_watch.argparse.Namespace(
        task="Isaac-FlappingBot-PathTracking-DeLaurier-PrimitivePureRL-Direct-v0",
        run_name="pure_warm_start",
        num_envs=128,
        eval_num_envs=8,
        max_iterations=400,
        save_interval=20,
        seed=7,
        train_device="cuda:0",
        eval_device="cuda:1",
        episodes=3,
        poll_s=30.0,
        run_dir_timeout_s=120.0,
        eval_suite="path_tracking_truth_primitives_nowind_v1",
        headless=True,
        resume=False,
        load_weights_only=True,
        load_run="2026-03-23_10-15-26_pt_bc_abs_primitives_v1",
        checkpoint="model_bc.pt",
    )

    cmd = train_and_watch._build_train_cmd(args)

    assert "--load_weights_only" in cmd
    assert "--resume" not in cmd
    assert cmd[cmd.index("--load_run") + 1] == "2026-03-23_10-15-26_pt_bc_abs_primitives_v1"
    assert cmd[cmd.index("--checkpoint") + 1] == "model_bc.pt"


def test_build_train_cmd_rejects_conflicting_resume_and_weights_only_flags() -> None:
    args = train_and_watch.argparse.Namespace(
        task="Isaac-FlappingBot-PathTracking-DeLaurier-PrimitivePureRL-Direct-v0",
        run_name="conflict",
        num_envs=128,
        eval_num_envs=8,
        max_iterations=400,
        save_interval=20,
        seed=7,
        train_device="cuda:0",
        eval_device="cuda:1",
        episodes=3,
        poll_s=30.0,
        run_dir_timeout_s=120.0,
        eval_suite="path_tracking_truth_primitives_nowind_v1",
        headless=True,
        resume=True,
        load_weights_only=True,
        load_run="2026-03-23_10-15-26_pt_bc_abs_primitives_v1",
        checkpoint="model_bc.pt",
    )

    with pytest.raises(ValueError, match="cannot both be enabled"):
        train_and_watch._build_train_cmd(args)


def test_build_train_cmd_omits_resume_arguments_when_disabled() -> None:
    args = train_and_watch.argparse.Namespace(
        task="Isaac-FlappingBot-StraightFlight-DeLaurier-WeakTeacherRL-Direct-v0",
        run_name="weak_fresh",
        num_envs=64,
        eval_num_envs=4,
        max_iterations=200,
        save_interval=10,
        seed=3,
        train_device="cuda:0",
        eval_device="cuda:1",
        episodes=1,
        poll_s=10.0,
        run_dir_timeout_s=60.0,
        eval_suite="single",
        headless=False,
        resume=False,
        load_run=None,
        checkpoint=None,
    )

    cmd = train_and_watch._build_train_cmd(args)

    assert "--resume" not in cmd
    assert "--load_run" not in cmd
    assert "--checkpoint" not in cmd


def test_build_train_cmd_applies_estimated_teacher_defaults_for_rl_tasks() -> None:
    args = train_and_watch.argparse.Namespace(
        task="Isaac-FlappingBot-PathTracking-DeLaurier-TeacherRL-Direct-v0",
        run_name="estimated_defaults",
        num_envs=64,
        eval_num_envs=4,
        max_iterations=200,
        save_interval=10,
        seed=3,
        train_device="cuda:0",
        eval_device="cuda:1",
        episodes=1,
        poll_s=10.0,
        run_dir_timeout_s=60.0,
        eval_suite="straight_standard",
        headless=False,
        resume=False,
        load_run=None,
        checkpoint=None,
    )

    cmd = train_and_watch._build_train_cmd(args)

    assert "env.teacher_state_source=estimated" in cmd
    assert "env.policy_state_source=estimated" in cmd
    assert "env.imu_source=synthetic" in cmd


def test_build_train_cmd_forwards_mass_override_when_provided() -> None:
    args = train_and_watch.argparse.Namespace(
        task="Isaac-FlappingBot-PathTracking-DeLaurier-TeacherRL-Direct-v0",
        run_name="mass_override",
        num_envs=64,
        eval_num_envs=4,
        max_iterations=200,
        save_interval=10,
        seed=3,
        mass_kg_override=0.95,
        train_device="cuda:0",
        eval_device="cuda:1",
        episodes=1,
        poll_s=10.0,
        run_dir_timeout_s=60.0,
        eval_suite="straight_standard",
        headless=False,
        resume=False,
        load_run=None,
        checkpoint=None,
    )

    cmd = train_and_watch._build_train_cmd(args)

    assert "env.total_mass_kg_override=0.95" in cmd


def test_build_watch_cmd_defaults_path_tracking_task_to_estimated_suite(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    args = train_and_watch.argparse.Namespace(
        task="Isaac-FlappingBot-PathTracking-DeLaurier-TeacherRL-Direct-v0",
        run_name="path_teacher",
        num_envs=64,
        eval_num_envs=16,
        max_iterations=200,
        save_interval=10,
        seed=3,
        train_device="cuda:0",
        eval_device="cuda:1",
        episodes=2,
        poll_s=10.0,
        run_dir_timeout_s=60.0,
        eval_suite="straight_standard",
        headless=True,
        resume=False,
        load_run=None,
        checkpoint=None,
    )

    cmd = train_and_watch._build_watch_cmd(args, run_dir)

    assert cmd[cmd.index("--eval_suite") + 1] == "path_tracking_estimated_nowind_v1"
    assert cmd[cmd.index("--num_envs") + 1] == "16"


def test_build_watch_cmd_defaults_primitive_path_tracking_task_to_estimated_primitive_suite(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    args = train_and_watch.argparse.Namespace(
        task="Isaac-FlappingBot-PathTracking-DeLaurier-PrimitiveWeakTeacherRL-Direct-v0",
        run_name="path_primitive",
        num_envs=64,
        eval_num_envs=16,
        max_iterations=200,
        save_interval=10,
        seed=3,
        train_device="cuda:0",
        eval_device="cuda:1",
        episodes=2,
        poll_s=10.0,
        run_dir_timeout_s=60.0,
        eval_suite="straight_standard",
        headless=True,
        resume=False,
        load_run=None,
        checkpoint=None,
    )

    cmd = train_and_watch._build_watch_cmd(args, run_dir)

    assert cmd[cmd.index("--eval_suite") + 1] == "path_tracking_estimated_primitives_nowind_v1"


def test_build_train_cmd_uses_portable_root_kit_args() -> None:
    args = train_and_watch.argparse.Namespace(
        task="Isaac-FlappingBot-PathTracking-DeLaurier-PrimitiveWeakTeacherRL-Direct-v0",
        run_name="portable_smoke",
        num_envs=128,
        eval_num_envs=4,
        max_iterations=50,
        save_interval=10,
        seed=5,
        train_device="cuda:0",
        eval_device="cuda:1",
        episodes=2,
        poll_s=10.0,
        run_dir_timeout_s=60.0,
        eval_suite="path_tracking_truth_nowind_v1",
        headless=True,
        resume=False,
        load_run=None,
        checkpoint=None,
        portable_root_base=Path("logs/portable/train_and_watch/portable_smoke_seed5_pid123"),
    )

    cmd = train_and_watch._build_train_cmd(args)

    assert cmd[:3] == ["./isaaclab.sh", "-p", "scripts/reinforcement_learning/rsl_rl/train.py"]
    assert "--kit_args" in cmd
    assert cmd[cmd.index("--kit_args") + 1] == "--portable-root logs/portable/train_and_watch/portable_smoke_seed5_pid123/train"


def test_build_watch_cmd_uses_distinct_portable_root_kit_args(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    args = train_and_watch.argparse.Namespace(
        task="Isaac-FlappingBot-PathTracking-DeLaurier-PrimitiveWeakTeacherRL-Direct-v0",
        run_name="portable_smoke",
        num_envs=128,
        eval_num_envs=4,
        max_iterations=50,
        save_interval=10,
        seed=5,
        train_device="cuda:0",
        eval_device="cuda:1",
        episodes=2,
        poll_s=10.0,
        run_dir_timeout_s=60.0,
        eval_suite="path_tracking_truth_nowind_v1",
        headless=True,
        resume=False,
        load_run=None,
        checkpoint=None,
        portable_root_base=Path("logs/portable/train_and_watch/portable_smoke_seed5_pid123"),
    )

    cmd = train_and_watch._build_watch_cmd(args, run_dir)

    assert "--kit_args" in cmd
    assert cmd[cmd.index("--kit_args") + 1] == "--portable-root logs/portable/train_and_watch/portable_smoke_seed5_pid123/watch"


def test_build_train_cmd_native_cpu_adds_extension_and_p0_overrides(tmp_path: Path) -> None:
    extension_parent = tmp_path / "native_extensions"
    args = train_and_watch.argparse.Namespace(
        task="Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-Direct-v0",
        run_name="native_cpu_p0",
        num_envs=64,
        max_iterations=2,
        save_interval=1,
        seed=0,
        train_device="cuda:0",
        eval_device="cuda:1",
        episodes=1,
        poll_s=10.0,
        eval_suite="straight_standard",
        headless=True,
        resume=False,
        load_weights_only=False,
        load_run=None,
        checkpoint=None,
        portable_root_base=tmp_path / "portable",
        native_cpu=True,
        native_extension_parent=extension_parent,
        agent_device=None,
        freeze_steps_after_reset=0,
    )

    cmd = train_and_watch._build_train_cmd(args)
    kit_args = cmd[cmd.index("--kit_args") + 1]

    assert cmd[:2] == [sys.executable, "scripts/reinforcement_learning/rsl_rl/train.py"]
    assert cmd[cmd.index("--device") + 1] == "cpu"
    assert f"--ext-folder {extension_parent.resolve()}" in kit_args
    assert "--enable omni.flapping_bot.holonomic_constraint" in kit_args
    assert "agent.device=cpu" in cmd
    assert "env.freeze_steps_after_reset=0" in cmd
    expected_asset = (
        Path(train_and_watch.__file__).resolve().parents[2]
        / "source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf"
    )
    expected_usd_dir = (tmp_path / "portable/train/generated_assets/flap_robot_552").resolve()
    assert f"env.robot.spawn.asset_path={expected_asset}" in cmd
    assert f"env.robot.spawn.usd_dir={expected_usd_dir}" in cmd


def test_measured_pure_rl_defaults_to_native_cpu_and_64_envs(tmp_path: Path) -> None:
    extension_parent = tmp_path / "native_extensions"
    args = train_and_watch.argparse.Namespace(
        task="Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-Direct-v0",
        run_name="measured_defaults",
        num_envs=None,
        max_iterations=2,
        save_interval=1,
        seed=0,
        train_device="cuda:0",
        eval_device="cuda:1",
        headless=True,
        resume=False,
        load_weights_only=False,
        load_run=None,
        checkpoint=None,
        portable_root_base=tmp_path / "portable",
        native_cpu=False,
        native_extension_parent=extension_parent,
        agent_device=None,
        freeze_steps_after_reset=None,
    )

    cmd = train_and_watch._build_train_cmd(args)
    kit_args = cmd[cmd.index("--kit_args") + 1]

    assert cmd[:2] == [sys.executable, "scripts/reinforcement_learning/rsl_rl/train.py"]
    assert cmd[cmd.index("--device") + 1] == "cpu"
    assert cmd[cmd.index("--num_envs") + 1] == "64"
    assert "agent.device=cpu" in cmd
    assert "env.freeze_steps_after_reset=0" not in cmd
    assert f"--ext-folder {extension_parent.resolve()}" in kit_args
    assert "--enable omni.flapping_bot.holonomic_constraint" in kit_args

    args.eval_num_envs = None
    args.episodes = None
    args.poll_s = 10.0
    args.eval_suite = "straight_standard"
    watch_cmd = train_and_watch._build_watch_cmd(args, tmp_path / "run")
    watch_kit_args = watch_cmd[watch_cmd.index("--kit_args") + 1]
    assert watch_cmd[:2] == [sys.executable, "scripts/flapping_rl/watch_and_eval.py"]
    assert watch_cmd[watch_cmd.index("--device") + 1] == "cpu"
    assert watch_cmd[watch_cmd.index("--eval_suite") + 1] == "pure_rl_curriculum1_nowind_v2"
    assert watch_cmd[watch_cmd.index("--num_envs") + 1] == "16"
    assert watch_cmd[watch_cmd.index("--episodes") + 1] == "16"
    assert f"--ext-folder {extension_parent.resolve()}" in watch_kit_args


def test_non_measured_task_keeps_legacy_launcher_defaults() -> None:
    args = train_and_watch.argparse.Namespace(
        task="Isaac-FlappingBot-StraightFlight-DeLaurier-PureRL-Direct-v0",
        run_name="legacy_defaults",
        num_envs=None,
        max_iterations=2,
        save_interval=1,
        seed=0,
        train_device="cuda:0",
        eval_device="cuda:1",
        headless=True,
        resume=False,
        load_weights_only=False,
        load_run=None,
        checkpoint=None,
        portable_root_base=Path("logs/portable/legacy_defaults"),
        native_cpu=False,
        agent_device=None,
        freeze_steps_after_reset=None,
    )

    cmd = train_and_watch._build_train_cmd(args)

    assert cmd[:3] == ["./isaaclab.sh", "-p", "scripts/reinforcement_learning/rsl_rl/train.py"]
    assert cmd[cmd.index("--device") + 1] == "cuda:0"
    assert cmd[cmd.index("--num_envs") + 1] == "512"
    assert "agent.device=cpu" not in cmd


def test_build_watch_cmd_native_cpu_adds_extension_and_uses_cpu(tmp_path: Path) -> None:
    extension_parent = tmp_path / "native_extensions"
    args = train_and_watch.argparse.Namespace(
        task="Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-Direct-v0",
        run_name="native_cpu_p0",
        eval_num_envs=1,
        seed=0,
        eval_device="cuda:1",
        episodes=1,
        poll_s=10.0,
        eval_suite="straight_standard",
        headless=True,
        portable_root_base=tmp_path / "portable",
        native_cpu=True,
        native_extension_parent=extension_parent,
    )

    cmd = train_and_watch._build_watch_cmd(args, tmp_path / "run")
    kit_args = cmd[cmd.index("--kit_args") + 1]

    assert cmd[:2] == [sys.executable, "scripts/flapping_rl/watch_and_eval.py"]
    assert cmd[cmd.index("--device") + 1] == "cpu"
    assert f"--ext-folder {extension_parent.resolve()}" in kit_args
    assert "--enable omni.flapping_bot.holonomic_constraint" in kit_args


def test_validate_native_extension_requires_built_binary(tmp_path: Path) -> None:
    args = train_and_watch.argparse.Namespace(native_cpu=True, native_extension_parent=tmp_path)

    with pytest.raises(FileNotFoundError, match="build_holonomic_constraint_extension.sh"):
        train_and_watch._validate_native_extension(args)

    binary = (
        tmp_path
        / "omni.flapping_bot.holonomic_constraint"
        / "omni/flapping_bot/holonomic_constraint/_native.so"
    )
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"")

    train_and_watch._validate_native_extension(args)


def test_native_child_environment_prioritizes_current_worktree_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "/external/pythonpath")
    args = train_and_watch.argparse.Namespace(native_cpu=True)

    child_env = train_and_watch._child_process_env(args)
    entries = child_env["PYTHONPATH"].split(train_and_watch.os.pathsep)
    repo_root = Path(train_and_watch.__file__).resolve().parents[2]

    assert entries[:2] == [
        str((repo_root / "source/flapping_bot").resolve()),
        str((repo_root / "source/isaaclab_assets").resolve()),
    ]
    assert entries[2] == "/external/pythonpath"
