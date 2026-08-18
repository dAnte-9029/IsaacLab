from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / "source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py"
DIRECT_INIT_FILE = ROOT / "source/flapping_bot/flapping_bot/direct/flapping_bot/__init__.py"
PACKAGE_INIT_FILE = ROOT / "source/flapping_bot/flapping_bot/__init__.py"


def _module(path: Path = ENV_FILE) -> ast.Module:
    return ast.parse(path.read_text())


def _class(module: ast.Module, name: str) -> ast.ClassDef:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found")


def _method(class_node: ast.ClassDef, name: str) -> ast.FunctionDef:
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"method {name} not found")


def _ann_assign(class_node: ast.ClassDef, name: str) -> ast.AnnAssign:
    for node in class_node.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            return node
    raise AssertionError(f"field {name} not found")


def _assigned_names(class_node: ast.ClassDef) -> set[str]:
    return {
        node.target.id
        for node in class_node.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }


def _called_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if isinstance(child.func, ast.Name):
            names.add(child.func.id)
        elif isinstance(child.func, ast.Attribute):
            names.add(child.func.attr)
    return names


def _source(node: ast.AST) -> str:
    return ast.unparse(node)


def test_c3_configs_add_only_spatial_stage_and_twenty_second_episode() -> None:
    module = _module()
    base = _class(module, "FlappingBotStraightFlightEnvCfg")
    measured_c1 = _class(module, "FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg")

    assert ast.literal_eval(_ann_assign(base, "pure_rl_spatial_stage_id").value) is None
    assert ast.literal_eval(_ann_assign(measured_c1, "pure_rl_spatial_stage_id").value) is None
    for schedule_name in (
        "pure_rl_eval_spatial_template_schedule",
        "pure_rl_eval_spatial_geometry_roll_deg_schedule",
        "pure_rl_eval_spatial_slope_deg_schedule",
        "pure_rl_eval_spatial_turn_sign_schedule",
    ):
        assert ast.literal_eval(_ann_assign(base, schedule_name).value) is None
    for suffix in ("C2a", "C2b", "C2c"):
        c2 = _class(module, f"FlappingBotStraightFlightDeLaurierMeasuredPureRL{suffix}EnvCfg")
        assert "pure_rl_spatial_stage_id" not in _assigned_names(c2)

    forbidden_overrides = {
        "action_interface",
        "observation_space",
        "wind_enabled",
        "randomize_wind",
        "wind_ou_enabled",
        "decimation",
    }
    for suffix, stage_id in (("C3a", "c3a"), ("C3b", "c3b"), ("C3c", "c3c")):
        cfg = _class(module, f"FlappingBotStraightFlightDeLaurierMeasuredPureRL{suffix}EnvCfg")
        assert len(cfg.bases) == 1
        assert isinstance(cfg.bases[0], ast.Name)
        assert cfg.bases[0].id == "FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg"
        expected_names = {"pure_rl_spatial_stage_id", "episode_length_s"}
        if suffix == "C3a":
            expected_names.add("pure_rl_warm_start_guard_enabled")
            assert ast.literal_eval(_ann_assign(cfg, "pure_rl_warm_start_guard_enabled").value) is True
        assert _assigned_names(cfg) == expected_names
        assert ast.literal_eval(_ann_assign(cfg, "pure_rl_spatial_stage_id").value) == stage_id
        assert ast.literal_eval(_ann_assign(cfg, "episode_length_s").value) == 20.0
        assert not (_assigned_names(cfg) & forbidden_overrides)


def test_constructor_resolves_one_route_stage_and_rejects_both() -> None:
    env = _class(_module(), "FlappingBotStraightFlightEnv")
    constructor = _method(env, "__init__")
    calls = _called_names(constructor)
    source = _source(constructor)

    assert "resolve_longitudinal_stage" in calls
    assert "resolve_spatial_stage" in calls
    assert "cfg.pure_rl_longitudinal_stage_id" in source
    assert "cfg.pure_rl_spatial_stage_id" in source
    assert "longitudinal_stage is not None and spatial_stage is not None" in source
    assert "raise ValueError" in source
    assert "self._pure_rl_spatial_stage" in source
    assert "self._pure_rl_spatial_path" in source
    assert "self._pure_rl_spatial_progress_m" in source
    assert "self._pure_rl_c2c_rehearsal_path" in source


def test_c3_reset_samples_selected_rows_and_aligns_heading_to_first_tangent() -> None:
    env = _class(_module(), "FlappingBotStraightFlightEnv")
    reset = _method(env, "_reset_idx")
    calls = _called_names(reset)
    source = _source(reset)

    assert "sample_spatial_path_batch" in calls
    assert "write_spatial_path_batch_rows_" in calls
    assert "initial_altitude_m=self._height_cmd[env_ids]" in source
    assert "evaluation_template_id" in source
    assert "evaluation_geometry_roll_deg" in source
    assert "evaluation_slope_deg" in source
    assert "evaluation_turn_sign" in source
    assert "evaluation_heading_rad" in source
    assert "self._pure_rl_spatial_progress_m[env_ids] = 0.0" in source
    assert "sampled_path.tangent_world[:, 0, 1]" in source
    assert "sampled_path.tangent_world[:, 0, 0]" in source
    assert "atan2" in calls
    heading_line = min(
        child.lineno
        for child in ast.walk(reset)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == "atan2"
    )
    quaternion_line = min(
        child.lineno
        for child in ast.walk(reset)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id == "quat_from_euler_xyz"
    )
    assert heading_line < quaternion_line
    assert "self._pure_rl_history_valid[env_ids]" in source
    assert "self._pure_rl_previous_reward_action[env_ids]" in source
    assert "resolve_longitudinal_stage" in calls
    assert "replace" in calls
    assert "sample_longitudinal_path_batch" in calls
    assert "write_longitudinal_path_batch_rows_" in calls
    assert "REHEARSAL_C2C_TASK_FAMILY_ID" in source
    assert "sampled_c2c_path.heading_rad" in source


def test_spatial_query_is_explicit_stable_state_and_c3_observation_keeps_555_contract() -> None:
    module = _module()
    env = _class(module, "FlappingBotStraightFlightEnv")
    query = _method(env, "_query_pure_rl_spatial_path")
    observation = _method(env, "_get_pure_rl_observations")
    query_source = _source(query)
    observation_source = _source(observation)

    assert _called_names(query) >= {"query_spatial_path", "copy_"}
    assert "previous_progress_m=self._pure_rl_spatial_progress_m" in query_source
    assert "position_world_m=self._robot.data.root_pos_w - self.scene.env_origins" in query_source
    assert "ground_velocity_world_mps=self._robot.data.root_lin_vel_w" in query_source
    assert "self._pure_rl_spatial_progress_m.copy_(query.progress_m)" in query_source
    assert "query_longitudinal_path" in _called_names(query)
    assert "self._pure_rl_c2c_rehearsal_path" in query_source
    assert "REHEARSAL_C2C_TASK_FAMILY_ID" in query_source
    assert "torch.where" in query_source
    assert "setattr" not in _called_names(query)
    assert "if self._pure_rl_spatial_stage is not None" in observation_source
    assert "self._query_pure_rl_spatial_path().preview_points_world_m" in observation_source
    measured = _class(module, "FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg")
    assert "PURE_RL_SHARED_CONTRACT.observation_dim" in _source(
        _ann_assign(measured, "observation_space").value
    )


def test_c3_actor_distillation_is_opt_in_and_kept_outside_policy_observation() -> None:
    module = _module()
    base = _class(module, "FlappingBotStraightFlightEnvCfg")
    env = _class(module, "FlappingBotStraightFlightEnv")
    constructor_source = _source(_method(env, "__init__"))
    observation_source = _source(_method(env, "_get_pure_rl_observations"))

    assert ast.literal_eval(_ann_assign(base, "pure_rl_actor_distillation_coefficient").value) == 0.0
    assert "actor_distillation_coefficient > 0.0" in constructor_source
    assert 'spatial_stage.stage_id != \'c3a\'' in constructor_source
    assert 'observations = {\'policy\': observation}' in observation_source
    assert "actor_distillation_mask" in observation_source
    assert "task_family_id <= REHEARSAL_C2C_TASK_FAMILY_ID" in observation_source


def test_c3_actor_gradient_probe_is_opt_in_and_kept_outside_policy_observation() -> None:
    module = _module()
    base = _class(module, "FlappingBotStraightFlightEnvCfg")
    env = _class(module, "FlappingBotStraightFlightEnv")
    constructor_source = _source(_method(env, "__init__"))
    observation_source = _source(_method(env, "_get_pure_rl_observations"))

    assert ast.literal_eval(_ann_assign(base, "pure_rl_actor_gradient_probe_enabled").value) is False
    assert "actor_gradient_probe_enabled" in constructor_source
    assert "spatial_stage.stage_id != 'c3a'" in constructor_source
    assert "ACTOR_GRADIENT_PROBE_GROUP_KEY" in observation_source
    assert "ACTOR_GRADIENT_PROBE_STRONG_C2C_GROUP" in observation_source


def test_c3a_enables_bounded_weights_only_warm_start_guard() -> None:
    module = _module()
    base = _class(module, "FlappingBotStraightFlightEnvCfg")
    c3a = _class(module, "FlappingBotStraightFlightDeLaurierMeasuredPureRLC3aEnvCfg")
    constructor_source = _source(_method(_class(module, "FlappingBotStraightFlightEnv"), "__init__"))

    assert ast.literal_eval(_ann_assign(base, "pure_rl_warm_start_guard_enabled").value) is False
    assert ast.literal_eval(_ann_assign(base, "pure_rl_warm_start_burn_in_iterations").value) == 3
    assert ast.literal_eval(_ann_assign(base, "pure_rl_warm_start_update_iterations").value) == 10
    assert ast.literal_eval(_ann_assign(base, "pure_rl_warm_start_initial_learning_rate").value) == 1.0e-5
    assert ast.literal_eval(_ann_assign(base, "pure_rl_warm_start_target_learning_rate").value) == 5.0e-5
    assert ast.literal_eval(_ann_assign(base, "pure_rl_warm_start_num_learning_epochs").value) == 1
    assert ast.literal_eval(_ann_assign(base, "pure_rl_warm_start_actor_update_norm_limit").value) == 0.10
    assert ast.literal_eval(_ann_assign(c3a, "pure_rl_warm_start_guard_enabled").value) is True
    assert "warm_start_guard_enabled" in constructor_source
    assert "spatial_stage.stage_id != 'c3a'" in constructor_source


def test_c3_adaptive_task_sampling_is_opt_in_and_uses_completed_episode_signals() -> None:
    module = _module()
    base = _class(module, "FlappingBotStraightFlightEnvCfg")
    env = _class(module, "FlappingBotStraightFlightEnv")
    constructor_source = _source(_method(env, "__init__"))
    reset_source = _source(_method(env, "_reset_idx"))
    dones_source = _source(_method(env, "_get_dones"))
    update_source = _source(_method(env, "_update_pure_rl_adaptive_task_sampling"))

    assert ast.literal_eval(_ann_assign(base, "pure_rl_adaptive_task_sampling_enabled").value) is False
    assert "spatial_stage.stage_id != 'c3a'" in constructor_source
    assert "task_probabilities=self._pure_rl_adaptive_task_probabilities" in reset_source
    assert "self._pure_rl_adaptive_strong_climb_probability" in reset_source
    assert "self._update_pure_rl_adaptive_task_sampling" in dones_source
    assert "c1_completed & ~terminated" in update_source
    assert "strong_c2c_completed & successful_c2c_recovery" in update_source
    assert "c3a_completed & query.reached_all_events & ~terminated" in update_source
    assert "update_retention_aware_task_probabilities" in update_source
    assert "AdaptiveSampling/c3a_probability" in update_source


def test_spatial_query_is_cached_once_per_policy_step_and_reset_invalidates_it() -> None:
    env = _class(_module(), "FlappingBotStraightFlightEnv")
    constructor = _method(env, "__init__")
    query = _method(env, "_query_pure_rl_spatial_path")
    reset = _method(env, "_reset_idx")
    constructor_source = _source(constructor)
    query_source = _source(query)
    reset_source = _source(reset)

    assert "self._pure_rl_spatial_query_cache" in constructor_source
    assert "self._pure_rl_spatial_query_step" in constructor_source
    assert "self.common_step_counter" in query_source
    assert "self._pure_rl_spatial_query_step == int(self.common_step_counter)" in query_source
    assert "return self._pure_rl_spatial_query_cache" in query_source
    assert "self._pure_rl_spatial_query_cache = None" in reset_source
    assert "self._pure_rl_spatial_query_step = -1" in reset_source


def test_c3_reward_precedes_c2_and_copies_spatial_path_telemetry() -> None:
    env = _class(_module(), "FlappingBotStraightFlightEnv")
    reward = _method(env, "_get_pure_rl_curriculum1_reward")
    calls = _called_names(reward)
    source = _source(reward)

    assert "compute_pure_rl_spatial_path_reward_terms" in calls
    assert "turn_activity=query.turn_activity" in source
    assert source.index("if self._pure_rl_spatial_stage is not None") < source.index(
        "elif self._pure_rl_longitudinal_stage is not None"
    )
    for field in (
        "horizontal_normal_error_m",
        "vertical_normal_error_m",
        "progress_m",
        "tangent_world",
        "lateral_normal_world",
        "vertical_normal_world",
        "active_curvature_rad_per_m",
        "active_slope_rad",
        "turn_activity",
        "reached_all_events",
        "task_family_id",
        "template_id",
        "turn_sign",
        "peak_geometry_roll_rad",
        "peak_slope_rad",
    ):
        assert field in source
    assert "roll_rad.abs()" in source


def test_c3_dones_use_spatial_termination_and_distinct_roll_limit_telemetry() -> None:
    env = _class(_module(), "FlappingBotStraightFlightEnv")
    dones = _method(env, "_get_dones")
    calls = _called_names(dones)
    source = _source(dones)

    assert "compute_pure_rl_spatial_termination_terms" in calls
    assert "roll_rad" in source
    assert "query.horizontal_normal_error_m" in source
    assert "query.vertical_normal_error_m" in source
    assert "self._eval_pure_rl_roll_limit_termination.copy_(terms.roll_limit)" in source
    assert "PureRLTermination/roll_limit_fraction" in source
    assert "REHEARSAL_C2C_TASK_FAMILY_ID" in source
    assert "terms.roll_limit & ~c2c_rehearsal" in source
    assert "self.cfg.pure_rl_c2c_recycle_on_recovery" in source
    assert "c2c_rehearsal & query.reached_all_events & ~terms.terminated" in source
    assert "return (terms.terminated, timed_out)" in source


def test_c3_reset_forwards_event_balanced_strong_climb_quota() -> None:
    env = _class(_module(), "FlappingBotStraightFlightEnv")
    reset_source = _source(_method(env, "_reset_idx"))

    assert "climb_strong_slope_probability" in reset_source
    assert "self.cfg.pure_rl_c2c_strong_climb_probability" in reset_source
    assert "climb_strong_slope_minimum_deg" in reset_source


def test_c3_configs_are_lazy_exported_without_path_tracking_or_upstream_dependencies() -> None:
    for file_path in (DIRECT_INIT_FILE, PACKAGE_INIT_FILE):
        text = file_path.read_text()
        for suffix in ("C3a", "C3b", "C3c"):
            name = f"FlappingBotStraightFlightDeLaurierMeasuredPureRL{suffix}EnvCfg"
            assert name in text
            assert name in ast.literal_eval(
                next(
                    node.value
                    for node in _module(file_path).body
                    if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
                )
            )
    env_text = ENV_FILE.read_text()
    assert "path_tracking_env" not in env_text
    assert "isaaclab_tasks" not in env_text
    assert "print(" not in _source(_method(_class(_module(), "FlappingBotStraightFlightEnv"), "_get_pure_rl_curriculum1_reward"))

def test_pure_rl_eval_snapshots_survive_auto_reset_until_watcher_reads_them() -> None:
    module = ast.parse(ENV_FILE.read_text())
    reset_method = next(
        node
        for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_reset_idx"
    )
    reset_loops = [
        node
        for node in ast.walk(reset_method)
        if isinstance(node, ast.For)
        and any(
            isinstance(item, ast.Attribute) and item.attr.startswith("_eval_pure_rl_")
            for item in ast.walk(node.iter)
        )
    ]
    assert reset_loops == []
