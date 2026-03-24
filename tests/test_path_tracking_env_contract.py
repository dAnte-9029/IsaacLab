from __future__ import annotations

import ast
from pathlib import Path

from flapping_bot.direct.flapping_bot.path_tracking_env import (
    FlappingBotPathTrackingEnv,
    FlappingBotPathTrackingEnvCfg,
    PATH_TRACKING_RUNTIME_AVAILABLE,
    _compute_curriculum_schedule_step,
    _compute_mission_seed,
    _resolve_loiter_curriculum_stage,
    _resolve_path_tracking_curriculum,
)


REGISTRATION_FILE = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "isaaclab_tasks"
    / "isaaclab_tasks"
    / "direct"
    / "flapping_bot"
    / "__init__.py"
)


def _registered_env_cfgs() -> dict[str, str]:
    module = ast.parse(REGISTRATION_FILE.read_text())
    registered: dict[str, str] = {}
    for node in module.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not isinstance(call.func, ast.Attribute):
            continue
        if not isinstance(call.func.value, ast.Name) or call.func.value.id != "gym" or call.func.attr != "register":
            continue

        task_id = None
        env_cfg = None
        for kw in call.keywords:
            if kw.arg == "id" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                task_id = kw.value.value
            if kw.arg == "kwargs" and isinstance(kw.value, ast.Dict):
                for key_node, value_node in zip(kw.value.keys, kw.value.values, strict=True):
                    if isinstance(key_node, ast.Constant) and key_node.value == "env_cfg_entry_point":
                        if isinstance(value_node, ast.Name):
                            env_cfg = value_node.id
        if task_id is not None and env_cfg is not None:
            registered[task_id] = env_cfg
    return registered


def test_path_tracking_task_is_registered() -> None:
    registered = _registered_env_cfgs()
    assert "Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0" in registered


def test_path_tracking_task_uses_expected_env_cfg() -> None:
    registered = _registered_env_cfgs()
    assert registered["Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0"] == "FlappingBotPathTrackingEnvCfg"


def test_path_tracking_teacher_rl_task_is_registered() -> None:
    registered = _registered_env_cfgs()
    assert "Isaac-FlappingBot-PathTracking-DeLaurier-TeacherRL-Direct-v0" in registered


def test_path_tracking_teacher_rl_task_uses_expected_env_cfg() -> None:
    registered = _registered_env_cfgs()
    assert registered["Isaac-FlappingBot-PathTracking-DeLaurier-TeacherRL-Direct-v0"] == "FlappingBotPathTrackingEnvCfg"


def test_path_tracking_primitive_teacher_rl_task_is_registered() -> None:
    registered = _registered_env_cfgs()
    assert "Isaac-FlappingBot-PathTracking-DeLaurier-PrimitiveTeacherRL-Direct-v0" in registered


def test_path_tracking_primitive_teacher_rl_task_uses_expected_env_cfg() -> None:
    registered = _registered_env_cfgs()
    assert (
        registered["Isaac-FlappingBot-PathTracking-DeLaurier-PrimitiveTeacherRL-Direct-v0"]
        == "FlappingBotPathTrackingPrimitiveTeacherRLEnvCfg"
    )


def test_path_tracking_weak_teacher_rl_task_is_registered() -> None:
    registered = _registered_env_cfgs()
    assert "Isaac-FlappingBot-PathTracking-DeLaurier-WeakTeacherRL-Direct-v0" in registered


def test_path_tracking_weak_teacher_rl_task_uses_expected_env_cfg() -> None:
    registered = _registered_env_cfgs()
    assert registered["Isaac-FlappingBot-PathTracking-DeLaurier-WeakTeacherRL-Direct-v0"] == "FlappingBotPathTrackingWeakTeacherRLEnvCfg"


def test_path_tracking_pure_rl_task_is_registered() -> None:
    registered = _registered_env_cfgs()
    assert "Isaac-FlappingBot-PathTracking-DeLaurier-PureRL-Direct-v0" in registered


def test_path_tracking_pure_rl_task_uses_expected_env_cfg() -> None:
    registered = _registered_env_cfgs()
    assert registered["Isaac-FlappingBot-PathTracking-DeLaurier-PureRL-Direct-v0"] == "FlappingBotPathTrackingPureRLEnvCfg"


def test_path_tracking_module_exports_task_symbols() -> None:
    assert FlappingBotPathTrackingEnv.__name__ == "FlappingBotPathTrackingEnv"
    assert FlappingBotPathTrackingEnvCfg.__name__ == "FlappingBotPathTrackingEnvCfg"
    assert isinstance(PATH_TRACKING_RUNTIME_AVAILABLE, bool)


def test_path_tracking_env_defaults_enable_teacher_guidance() -> None:
    module = ast.parse(
        (
            Path(__file__).resolve().parents[1]
            / "source"
            / "flapping_bot"
            / "flapping_bot"
            / "direct"
            / "flapping_bot"
            / "path_tracking_env.py"
        ).read_text()
    )
    for node in ast.walk(module):
        if not isinstance(node, ast.ClassDef) or node.name != "FlappingBotPathTrackingEnvCfg":
            continue
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) and item.target.id == "teacher_guidance_enabled":
                assert isinstance(item.value, ast.Constant) and item.value.value is True
                return
    raise AssertionError("teacher_guidance_enabled=True not found in FlappingBotPathTrackingEnvCfg")


def test_path_tracking_env_defaults_are_no_wind_for_truth_training() -> None:
    module = ast.parse(
        (
            Path(__file__).resolve().parents[1]
            / "source"
            / "flapping_bot"
            / "flapping_bot"
            / "direct"
            / "flapping_bot"
            / "path_tracking_env.py"
        ).read_text()
    )

    expected_fields = {
        "wind_enabled": False,
        "randomize_wind": False,
        "wind_ou_enabled": False,
        "wind_curriculum_enabled": False,
    }
    found_fields: dict[str, bool] = {}

    for node in ast.walk(module):
        if not isinstance(node, ast.ClassDef) or node.name != "FlappingBotPathTrackingEnvCfg":
            continue
        for item in node.body:
            if not isinstance(item, ast.AnnAssign) or not isinstance(item.target, ast.Name):
                continue
            field_name = item.target.id
            if field_name not in expected_fields:
                continue
            assert isinstance(item.value, ast.Constant)
            found_fields[field_name] = item.value.value
        break

    assert found_fields == expected_fields


def test_path_tracking_env_defaults_disable_action_filtering() -> None:
    module = ast.parse(
        (
            Path(__file__).resolve().parents[1]
            / "source"
            / "flapping_bot"
            / "flapping_bot"
            / "direct"
            / "flapping_bot"
            / "path_tracking_env.py"
        ).read_text()
    )

    expected_fields = {
        "act_lpf_tau_s": 0.0,
        "act_rate_limit_per_s": 0.0,
    }
    found_fields: dict[str, float] = {}

    for node in ast.walk(module):
        if not isinstance(node, ast.ClassDef) or node.name != "FlappingBotPathTrackingEnvCfg":
            continue
        for item in node.body:
            if not isinstance(item, ast.AnnAssign) or not isinstance(item.target, ast.Name):
                continue
            field_name = item.target.id
            if field_name not in expected_fields:
                continue
            assert isinstance(item.value, ast.Constant)
            found_fields[field_name] = float(item.value.value)
        break

    assert found_fields == expected_fields


def test_path_tracking_env_exposes_no_progress_stall_cfg_fields() -> None:
    module = ast.parse(
        (
            Path(__file__).resolve().parents[1]
            / "source"
            / "flapping_bot"
            / "flapping_bot"
            / "direct"
            / "flapping_bot"
            / "path_tracking_env.py"
        ).read_text()
    )

    required_fields = {
        "no_progress_warmup_s",
        "no_progress_window_s",
        "no_progress_min_delta_s_m",
        "no_progress_penalty",
    }
    found: dict[str, float] = {}

    for node in ast.walk(module):
        if not isinstance(node, ast.ClassDef) or node.name != "FlappingBotPathTrackingEnvCfg":
            continue
        for item in node.body:
            if not isinstance(item, ast.AnnAssign) or not isinstance(item.target, ast.Name):
                continue
            if item.target.id not in required_fields:
                continue
            assert isinstance(item.value, ast.Constant)
            found[item.target.id] = float(item.value.value)
        break

    assert required_fields == set(found.keys())
    assert found["no_progress_warmup_s"] >= 0.0
    assert found["no_progress_window_s"] > 0.0
    assert found["no_progress_min_delta_s_m"] >= 0.0
    assert found["no_progress_penalty"] >= 0.0


def test_path_tracking_reset_keeps_path_specific_episode_horizon() -> None:
    module = ast.parse(
        (
            Path(__file__).resolve().parents[1]
            / "source"
            / "flapping_bot"
            / "flapping_bot"
            / "direct"
            / "flapping_bot"
            / "path_tracking_env.py"
        ).read_text()
    )

    for node in ast.walk(module):
        if not isinstance(node, ast.ClassDef) or node.name != "FlappingBotPathTrackingEnv":
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef) or item.name != "_reset_idx":
                continue
            for subnode in ast.walk(item):
                if not isinstance(subnode, ast.Assign):
                    continue
                for target in subnode.targets:
                    if not isinstance(target, ast.Subscript):
                        continue
                    if not isinstance(target.value, ast.Attribute) or target.value.attr != "_path_episode_horizon_steps":
                        continue
                    if (
                        isinstance(subnode.value, ast.Call)
                        and isinstance(subnode.value.func, ast.Name)
                        and subnode.value.func.id == "int"
                        and len(subnode.value.args) == 1
                        and isinstance(subnode.value.args[0], ast.Attribute)
                        and isinstance(subnode.value.args[0].value, ast.Name)
                        and subnode.value.args[0].value.id == "self"
                        and subnode.value.args[0].attr == "max_episode_length"
                    ):
                        raise AssertionError("_reset_idx should not overwrite path-specific episode horizons with self.max_episode_length")
            return

    raise AssertionError("FlappingBotPathTrackingEnv._reset_idx not found")


def test_path_tracking_weak_teacher_cfg_relaxes_teacher_schedule_earlier() -> None:
    module = ast.parse(
        (
            Path(__file__).resolve().parents[1]
            / "source"
            / "flapping_bot"
            / "flapping_bot"
            / "direct"
            / "flapping_bot"
            / "path_tracking_env.py"
        ).read_text()
    )

    schedule_steps = None
    schedule_deltas = None
    for node in ast.walk(module):
        if not isinstance(node, ast.ClassDef) or node.name != "FlappingBotPathTrackingWeakTeacherRLEnvCfg":
            continue
        for item in node.body:
            if not isinstance(item, ast.AnnAssign) or not isinstance(item.target, ast.Name):
                continue
            if item.target.id == "teacher_guidance_schedule_steps":
                assert isinstance(item.value, ast.Tuple)
                schedule_steps = tuple(elt.value for elt in item.value.elts)
            if item.target.id == "teacher_guidance_schedule_deltas":
                assert isinstance(item.value, ast.Tuple)
                schedule_deltas = tuple(elt.value for elt in item.value.elts)
        break

    assert schedule_steps == (0, 10_000, 30_000, 60_000)
    assert schedule_deltas == (0.35, 0.75, 1.5, 2.0)


def test_path_tracking_weak_teacher_cfg_applies_teacher_gap_penalty() -> None:
    module = ast.parse(
        (
            Path(__file__).resolve().parents[1]
            / "source"
            / "flapping_bot"
            / "flapping_bot"
            / "direct"
            / "flapping_bot"
            / "path_tracking_env.py"
        ).read_text()
    )

    teacher_gap_penalty = None
    for node in ast.walk(module):
        if not isinstance(node, ast.ClassDef) or node.name != "FlappingBotPathTrackingWeakTeacherRLEnvCfg":
            continue
        for item in node.body:
            if not isinstance(item, ast.AnnAssign) or not isinstance(item.target, ast.Name):
                continue
            if item.target.id == "teacher_action_gap_penalty":
                assert isinstance(item.value, ast.Constant)
                teacher_gap_penalty = float(item.value.value)
        break

    assert teacher_gap_penalty is not None
    assert teacher_gap_penalty > 0.0


def test_path_tracking_weak_teacher_cfg_uses_residual_teacher_guidance_mode() -> None:
    module = ast.parse(
        (
            Path(__file__).resolve().parents[1]
            / "source"
            / "flapping_bot"
            / "flapping_bot"
            / "direct"
            / "flapping_bot"
            / "path_tracking_env.py"
        ).read_text()
    )

    expected_modes = {
        "FlappingBotPathTrackingWeakTeacherRLEnvCfg": "residual",
        "FlappingBotPathTrackingPrimitiveWeakTeacherRLEnvCfg": "residual",
    }
    found_modes: dict[str, str] = {}

    for node in ast.walk(module):
        if not isinstance(node, ast.ClassDef) or node.name not in expected_modes:
            continue
        for item in node.body:
            if not isinstance(item, ast.AnnAssign) or not isinstance(item.target, ast.Name):
                continue
            if item.target.id != "teacher_guidance_mode":
                continue
            assert isinstance(item.value, ast.Constant) and isinstance(item.value.value, str)
            found_modes[node.name] = item.value.value

    assert found_modes == expected_modes


def test_path_tracking_weak_teacher_cfg_enables_zero_actor_bootstrap() -> None:
    module = ast.parse(
        (
            Path(__file__).resolve().parents[1]
            / "source"
            / "flapping_bot"
            / "flapping_bot"
            / "direct"
            / "flapping_bot"
            / "path_tracking_env.py"
        ).read_text()
    )

    expected_flags = {
        "FlappingBotPathTrackingWeakTeacherRLEnvCfg": True,
        "FlappingBotPathTrackingPrimitiveWeakTeacherRLEnvCfg": True,
    }
    found_flags: dict[str, bool] = {}

    for node in ast.walk(module):
        if not isinstance(node, ast.ClassDef) or node.name not in expected_flags:
            continue
        for item in node.body:
            if not isinstance(item, ast.AnnAssign) or not isinstance(item.target, ast.Name):
                continue
            if item.target.id != "teacher_guidance_zero_actor_init":
                continue
            assert isinstance(item.value, ast.Constant) and isinstance(item.value.value, bool)
            found_flags[node.name] = item.value.value

    assert found_flags == expected_flags


def test_path_tracking_pure_rl_cfg_disables_teacher_guidance() -> None:
    module = ast.parse(
        (
            Path(__file__).resolve().parents[1]
            / "source"
            / "flapping_bot"
            / "flapping_bot"
            / "direct"
            / "flapping_bot"
            / "path_tracking_env.py"
        ).read_text()
    )

    found = {}
    for node in ast.walk(module):
        if not isinstance(node, ast.ClassDef) or node.name != "FlappingBotPathTrackingPureRLEnvCfg":
            continue
        for item in node.body:
            if not isinstance(item, ast.AnnAssign) or not isinstance(item.target, ast.Name):
                continue
            if item.target.id in {"teacher_guidance_enabled", "teacher_guidance_disable_after_steps"}:
                assert isinstance(item.value, ast.Constant)
                found[item.target.id] = item.value.value
        break

    assert found == {
        "teacher_guidance_enabled": False,
        "teacher_guidance_disable_after_steps": 0,
    }


def test_path_tracking_primitive_weak_teacher_cfg_uses_single_segment_primitives() -> None:
    module = ast.parse(
        (
            Path(__file__).resolve().parents[1]
            / "source"
            / "flapping_bot"
            / "flapping_bot"
            / "direct"
            / "flapping_bot"
            / "path_tracking_env.py"
        ).read_text()
    )

    found = {}
    for node in ast.walk(module):
        if not isinstance(node, ast.ClassDef) or node.name != "FlappingBotPathTrackingPrimitiveWeakTeacherRLEnvCfg":
            continue
        for item in node.body:
            if not isinstance(item, ast.AnnAssign) or not isinstance(item.target, ast.Name):
                continue
            if item.target.id in {
                "mission_num_segments_min",
                "mission_num_segments_max",
                "mission_allow_straight",
                "mission_allow_turn",
                "mission_allow_loiter",
                "mission_allow_climb_on_straight",
            }:
                assert isinstance(item.value, ast.Constant)
                found[item.target.id] = item.value.value
        break

    assert found == {
        "mission_num_segments_min": 1,
        "mission_num_segments_max": 1,
        "mission_allow_straight": True,
        "mission_allow_turn": True,
        "mission_allow_loiter": True,
        "mission_allow_climb_on_straight": False,
    }


def test_path_tracking_primitive_pure_rl_cfg_uses_single_segment_primitives() -> None:
    module = ast.parse(
        (
            Path(__file__).resolve().parents[1]
            / "source"
            / "flapping_bot"
            / "flapping_bot"
            / "direct"
            / "flapping_bot"
            / "path_tracking_env.py"
        ).read_text()
    )

    found = {}
    for node in ast.walk(module):
        if not isinstance(node, ast.ClassDef) or node.name != "FlappingBotPathTrackingPrimitivePureRLEnvCfg":
            continue
        for item in node.body:
            if not isinstance(item, ast.AnnAssign) or not isinstance(item.target, ast.Name):
                continue
            if item.target.id in {
                "mission_num_segments_min",
                "mission_num_segments_max",
                "mission_allow_straight",
                "mission_allow_turn",
                "mission_allow_loiter",
                "mission_allow_climb_on_straight",
            }:
                assert isinstance(item.value, ast.Constant)
                found[item.target.id] = item.value.value
        break

    assert found == {
        "mission_num_segments_min": 1,
        "mission_num_segments_max": 1,
        "mission_allow_straight": True,
        "mission_allow_turn": True,
        "mission_allow_loiter": True,
        "mission_allow_climb_on_straight": False,
    }


def test_path_tracking_primitive_pure_rl_cfg_slows_curriculum_for_continuation() -> None:
    module = ast.parse(
        (
            Path(__file__).resolve().parents[1]
            / "source"
            / "flapping_bot"
            / "flapping_bot"
            / "direct"
            / "flapping_bot"
            / "path_tracking_env.py"
        ).read_text()
    )

    found = {}
    for node in ast.walk(module):
        if not isinstance(node, ast.ClassDef) or node.name != "FlappingBotPathTrackingPrimitivePureRLEnvCfg":
            continue
        for item in node.body:
            if not isinstance(item, ast.AnnAssign) or not isinstance(item.target, ast.Name):
                continue
            if item.target.id in {"mission_curriculum_stage_steps", "loiter_curriculum_stage_steps"}:
                assert isinstance(item.value, ast.Tuple)
                found[item.target.id] = tuple(elt.value for elt in item.value.elts)
        break

    assert found == {
        "mission_curriculum_stage_steps": (0, 400_000, 800_000, 1_200_000),
        "loiter_curriculum_stage_steps": (0, 400_000, 800_000, 1_200_000),
    }


def test_path_tracking_env_uses_fixed_five_point_preview_contract() -> None:
    module = ast.parse(
        (
            Path(__file__).resolve().parents[1]
            / "source"
            / "flapping_bot"
            / "flapping_bot"
            / "direct"
            / "flapping_bot"
            / "path_tracking_env.py"
        ).read_text()
    )

    preview_tuple_values: list[tuple[int, ...]] = []
    observation_space_value: int | None = None

    for node in ast.walk(module):
        if isinstance(node, ast.ClassDef) and node.name == "FlappingBotPathTrackingEnvCfg":
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) and item.target.id == "observation_space":
                    assert isinstance(item.value, ast.Constant) and isinstance(item.value.value, int)
                    observation_space_value = item.value.value

        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if not isinstance(target, ast.Attribute) or target.attr not in {
                "_path_preview_points_xyz",
                "_path_preview_points_body_xyz",
            }:
                continue
            call = node.value
            if not isinstance(call, ast.Call):
                continue
            if not isinstance(call.func, ast.Attribute) or call.func.attr != "zeros":
                continue
            shape_arg = call.args[0] if call.args else None
            if not isinstance(shape_arg, ast.Tuple):
                continue
            dims: list[int] = []
            for elt in shape_arg.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
                    dims.append(elt.value)
            if dims:
                preview_tuple_values.append(tuple(dims))

    assert observation_space_value == 96
    assert (5, 3) in {dims[-2:] for dims in preview_tuple_values}


def test_compute_mission_seed_can_freeze_eval_missions_across_resets() -> None:
    changing_seed = _compute_mission_seed(
        base_seed=101,
        path_reset_counter=4,
        env_id=2,
        increment_per_reset=True,
    )
    frozen_seed = _compute_mission_seed(
        base_seed=101,
        path_reset_counter=4,
        env_id=2,
        increment_per_reset=False,
    )

    assert changing_seed == 101 + 4 * 7919 + 2
    assert frozen_seed == 103


def test_resolve_loiter_curriculum_stage_maps_quarter_half_and_full() -> None:
    quarter_stage = _resolve_loiter_curriculum_stage(
        step=15_000,
        enabled=True,
        stage_steps=(0, 12_000, 24_000, 36_000),
        stage_modes=("turn_only", "loiter_quarter", "loiter_half", "loiter_full_with_straight_rehearsal"),
        default_loiter_turns=1.0,
        straight_rehearsal_prob=0.2,
    )
    half_stage = _resolve_loiter_curriculum_stage(
        step=30_000,
        enabled=True,
        stage_steps=(0, 12_000, 24_000, 36_000),
        stage_modes=("turn_only", "loiter_quarter", "loiter_half", "loiter_full_with_straight_rehearsal"),
        default_loiter_turns=1.0,
        straight_rehearsal_prob=0.2,
    )
    full_stage = _resolve_loiter_curriculum_stage(
        step=40_000,
        enabled=True,
        stage_steps=(0, 12_000, 24_000, 36_000),
        stage_modes=("turn_only", "loiter_quarter", "loiter_half", "loiter_full_with_straight_rehearsal"),
        default_loiter_turns=1.0,
        straight_rehearsal_prob=0.2,
    )

    assert quarter_stage.loiter_turns == 0.25
    assert quarter_stage.straight_rehearsal_prob == 0.0
    assert half_stage.loiter_turns == 0.5
    assert half_stage.straight_rehearsal_prob == 0.0
    assert full_stage.loiter_turns == 1.0
    assert full_stage.straight_rehearsal_prob == 0.2


def test_resolve_path_tracking_curriculum_supports_loiter_progressive_modes() -> None:
    quarter_flags = _resolve_path_tracking_curriculum(
        step=15_000,
        enabled=True,
        stage_steps=(0, 12_000, 24_000, 36_000),
        stage_modes=("turn_only", "loiter_quarter", "loiter_half", "loiter_full_with_straight_rehearsal"),
        allow_straight=True,
        allow_turn=True,
        allow_loiter=True,
    )
    full_flags = _resolve_path_tracking_curriculum(
        step=40_000,
        enabled=True,
        stage_steps=(0, 12_000, 24_000, 36_000),
        stage_modes=("turn_only", "loiter_quarter", "loiter_half", "loiter_full_with_straight_rehearsal"),
        allow_straight=True,
        allow_turn=True,
        allow_loiter=True,
    )

    assert quarter_flags == (False, False, True)
    assert full_flags == (True, False, True)


def test_resolve_path_tracking_curriculum_preserves_explicit_eval_loiter_case() -> None:
    eval_loiter_flags = _resolve_path_tracking_curriculum(
        step=40_000,
        enabled=True,
        stage_steps=(0, 12_000, 24_000, 36_000),
        stage_modes=("turn_only", "loiter_quarter", "loiter_half", "loiter_full_with_straight_rehearsal"),
        allow_straight=False,
        allow_turn=False,
        allow_loiter=True,
    )

    assert eval_loiter_flags == (False, False, True)


def test_curriculum_schedule_step_scales_common_steps_by_num_envs() -> None:
    assert _compute_curriculum_schedule_step(common_step_counter=192, num_envs=64) == 12_288


def test_primitive_curriculum_reaches_loiter_quarter_after_one_rollout_of_64x192_samples() -> None:
    schedule_step = _compute_curriculum_schedule_step(common_step_counter=192, num_envs=64)

    flags = _resolve_path_tracking_curriculum(
        step=schedule_step,
        enabled=True,
        stage_steps=(0, 12_000, 24_000, 36_000),
        stage_modes=("turn_only", "loiter_quarter", "loiter_half", "loiter_full_with_straight_rehearsal"),
        allow_straight=True,
        allow_turn=True,
        allow_loiter=True,
    )
    loiter_stage = _resolve_loiter_curriculum_stage(
        step=schedule_step,
        enabled=True,
        stage_steps=(0, 12_000, 24_000, 36_000),
        stage_modes=("turn_only", "loiter_quarter", "loiter_half", "loiter_full_with_straight_rehearsal"),
        stage_turns=(1.0, 0.25, 0.5, 1.0),
        default_loiter_turns=1.0,
        straight_rehearsal_prob=0.2,
    )

    assert flags == (False, False, True)
    assert loiter_stage.mode == "loiter_quarter"
    assert loiter_stage.loiter_turns == 0.25
    assert loiter_stage.straight_rehearsal_prob == 0.0


def test_path_tracking_env_exposes_eval_snapshot_buffers() -> None:
    module = ast.parse(
        (
            Path(__file__).resolve().parents[1]
            / "source"
            / "flapping_bot"
            / "flapping_bot"
            / "direct"
            / "flapping_bot"
            / "path_tracking_env.py"
        ).read_text()
    )

    snapshot_attrs = set()
    for node in ast.walk(module):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute) and target.attr.startswith("_eval_"):
                snapshot_attrs.add(target.attr)

    assert "_eval_abs_lateral_error_m" in snapshot_attrs
    assert "_eval_abs_height_error_m" in snapshot_attrs
    assert "_eval_abs_align_error_deg" in snapshot_attrs
    assert "_eval_progress_ratio" in snapshot_attrs
    assert "_eval_stalled" in snapshot_attrs


def test_path_tracking_env_exposes_reset_stalled_signal_for_eval() -> None:
    source_text = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "flapping_bot"
        / "flapping_bot"
        / "direct"
        / "flapping_bot"
        / "path_tracking_env.py"
    ).read_text()

    assert "reset_stalled" in source_text


def test_path_tracking_env_preserves_reset_stalled_until_post_step_eval_reads_it() -> None:
    source_text = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "flapping_bot"
        / "flapping_bot"
        / "direct"
        / "flapping_bot"
        / "path_tracking_env.py"
    ).read_text()

    assert "self.reset_stalled[env_ids] = False" not in source_text


def test_path_tracking_headless_fallback_defines_all_exported_cfg_placeholders() -> None:
    module = ast.parse(
        (
            Path(__file__).resolve().parents[1]
            / "source"
            / "flapping_bot"
            / "flapping_bot"
            / "direct"
            / "flapping_bot"
            / "path_tracking_env.py"
        ).read_text()
    )

    fallback_class_names = set()
    for node in module.body:
        if not isinstance(node, ast.Try):
            continue
        for handler in node.handlers:
            if not isinstance(handler.type, ast.Name) or handler.type.id != "ModuleNotFoundError":
                continue
            for item in handler.body:
                if isinstance(item, ast.ClassDef):
                    fallback_class_names.add(item.name)
            break

    assert {
        "FlappingBotPathTrackingEnvCfg",
        "FlappingBotPathTrackingWeakTeacherRLEnvCfg",
        "FlappingBotPathTrackingPureRLEnvCfg",
        "FlappingBotPathTrackingPrimitiveWeakTeacherRLEnvCfg",
        "FlappingBotPathTrackingPrimitivePureRLEnvCfg",
        "FlappingBotPathTrackingEnv",
    } <= fallback_class_names


def test_top_level_flapping_bot_path_tracking_exports_do_not_import_flapping_env() -> None:
    module = ast.parse(
        (
            Path(__file__).resolve().parents[1]
            / "source"
            / "flapping_bot"
            / "flapping_bot"
            / "__init__.py"
        ).read_text()
    )

    path_tracking_imports = []
    for node in ast.walk(module):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not isinstance(test, ast.Compare) or not isinstance(test.left, ast.Name) or test.left.id != "name":
            continue
        if not test.comparators or not isinstance(test.comparators[0], ast.Tuple):
            continue
        names = {
            elt.value
            for elt in test.comparators[0].elts
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
        }
        if "FlappingBotPathTrackingEnvCfg" not in names:
            continue
        for item in node.body:
            if isinstance(item, ast.ImportFrom) and isinstance(item.module, str) and item.module.endswith("direct.flapping_bot"):
                path_tracking_imports.extend(alias.name for alias in item.names)

    assert "FlappingBotEnv" not in path_tracking_imports
    assert "FlappingBotEnvCfg" not in path_tracking_imports
