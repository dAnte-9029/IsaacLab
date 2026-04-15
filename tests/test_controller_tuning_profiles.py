from flapping_bot.flapping_bot.px4_like.controller_tuning_profiles import (
    apply_controller_tuning_profile,
    resolve_controller_tuning_profile,
)


def test_truth_profile_keeps_baseline_kwargs_unchanged() -> None:
    base_kwargs = {
        "guidance_period_s": 2.2,
        "guidance_roll_time_const_s": 0.18,
        "heading_p_gain": 1.8,
        "roll_kp": 4.5,
        "roll_kd": 0.85,
        "max_roll_deg": 45.0,
    }

    profile_name = resolve_controller_tuning_profile(controller_state_source="truth")

    assert profile_name == "truth_baseline"
    assert apply_controller_tuning_profile(base_kwargs, controller_state_source="truth") == base_kwargs


def test_estimated_profile_applies_expected_lateral_overrides() -> None:
    tuned_kwargs = apply_controller_tuning_profile(
        {
            "guidance_period_s": 2.2,
            "guidance_roll_time_const_s": 0.18,
            "heading_p_gain": 1.8,
            "roll_kp": 4.5,
            "roll_kd": 0.85,
            "max_roll_deg": 45.0,
        },
        controller_state_source="estimated",
    )

    assert resolve_controller_tuning_profile(controller_state_source="estimated") == "estimated_teacher"
    assert tuned_kwargs["guidance_period_s"] == 2.2
    assert tuned_kwargs["guidance_roll_time_const_s"] == 0.18
    assert tuned_kwargs["heading_p_gain"] == 1.8
    assert tuned_kwargs["roll_kp"] == 4.5
    assert tuned_kwargs["roll_kd"] == 0.55
    assert tuned_kwargs["max_roll_deg"] == 35.0


def test_estimated_profile_can_apply_loiter_only_roll_rate_override() -> None:
    straight_kwargs = {
        "max_roll_deg": 45.0,
        "roll_kd": 0.85,
        "inner_elevon_roll_rate_limit_per_s": 6.0,
        "tecs_load_factor_use_roll_sp": True,
        "guidance_min_ground_speed_mps": 0.0,
        "lateral_guidance_uncertainty_start_deg": 0.0,
        "lateral_guidance_uncertainty_full_deg": 0.0,
        "lateral_guidance_uncertainty_min_scale": 1.0,
    }
    loiter_kwargs = dict(straight_kwargs)

    straight_result = apply_controller_tuning_profile(
        straight_kwargs,
        controller_state_source="estimated",
        controller_kind="straight_line",
    )
    loiter_result = apply_controller_tuning_profile(
        loiter_kwargs,
        controller_state_source="estimated",
        controller_kind="loiter",
    )

    assert straight_result["max_roll_deg"] == 35.0
    assert straight_result["roll_kd"] == 0.55
    assert straight_result["inner_elevon_roll_rate_limit_per_s"] == 6.0
    assert straight_result["tecs_load_factor_use_roll_sp"] is False
    assert straight_result["guidance_min_ground_speed_mps"] == 0.0
    assert straight_result["lateral_guidance_uncertainty_min_scale"] == 1.0

    assert loiter_result["max_roll_deg"] == 40.0
    assert loiter_result["roll_kd"] == 0.45
    assert loiter_result["inner_elevon_roll_rate_limit_per_s"] == 16.0
    assert loiter_result["tecs_load_factor_use_roll_sp"] is False
    assert loiter_result["guidance_min_ground_speed_mps"] == 2.0
    assert loiter_result["lateral_guidance_uncertainty_start_deg"] == 8.0
    assert loiter_result["lateral_guidance_uncertainty_full_deg"] == 25.0
    assert loiter_result["lateral_guidance_uncertainty_min_scale"] == 0.6


def test_estimated_profile_applies_path_tracking_heading_fusion_overrides() -> None:
    result = apply_controller_tuning_profile(
        {
            "heading_p_gain": 1.8,
            "lateral_heading_yaw_blend": 0.0,
            "lateral_heading_yaw_correction_limit_deg": 180.0,
            "tecs_load_factor_use_roll_sp": True,
            "guidance_min_ground_speed_mps": 0.0,
            "lateral_guidance_uncertainty_start_deg": 0.0,
            "lateral_guidance_uncertainty_full_deg": 0.0,
            "lateral_guidance_uncertainty_min_scale": 1.0,
        },
        controller_state_source="estimated",
        controller_kind="path_tracking",
    )

    assert result["heading_p_gain"] == 1.4
    assert result["lateral_heading_yaw_blend"] == 0.75
    assert result["lateral_heading_yaw_correction_limit_deg"] == 20.0
    assert result["tecs_load_factor_use_roll_sp"] is False
    assert result["guidance_min_ground_speed_mps"] == 2.0
    assert result["lateral_guidance_uncertainty_start_deg"] == 8.0
    assert result["lateral_guidance_uncertainty_full_deg"] == 25.0
    assert result["lateral_guidance_uncertainty_min_scale"] == 0.6


def test_truth_profile_does_not_apply_loiter_roll_rate_override() -> None:
    kwargs = {
        "max_roll_deg": 45.0,
        "roll_kd": 0.85,
        "inner_elevon_roll_rate_limit_per_s": 6.0,
        "tecs_load_factor_use_roll_sp": True,
        "guidance_min_ground_speed_mps": 0.0,
        "lateral_guidance_uncertainty_start_deg": 0.0,
        "lateral_guidance_uncertainty_full_deg": 0.0,
        "lateral_guidance_uncertainty_min_scale": 1.0,
    }

    result = apply_controller_tuning_profile(
        kwargs,
        controller_state_source="truth",
        controller_kind="loiter",
    )

    assert result["max_roll_deg"] == 45.0
    assert result["roll_kd"] == 0.85
    assert result["inner_elevon_roll_rate_limit_per_s"] == 6.0
    assert result["tecs_load_factor_use_roll_sp"] is True
    assert result["guidance_min_ground_speed_mps"] == 0.0
    assert result["lateral_guidance_uncertainty_min_scale"] == 1.0


def test_explicit_profile_overrides_win_over_estimated_profile_defaults() -> None:
    result = apply_controller_tuning_profile(
        {
            "max_roll_deg": 45.0,
            "roll_kd": 0.85,
            "inner_elevon_roll_rate_limit_per_s": 6.0,
            "tecs_load_factor_use_roll_sp": True,
            "guidance_min_ground_speed_mps": 0.0,
            "lateral_guidance_uncertainty_start_deg": 0.0,
            "lateral_guidance_uncertainty_full_deg": 0.0,
            "lateral_guidance_uncertainty_min_scale": 1.0,
        },
        controller_state_source="estimated",
        controller_kind="loiter",
        explicit_overrides={
            "max_roll_deg": 40.0,
            "roll_kd": 0.70,
            "inner_elevon_roll_rate_limit_per_s": 15.0,
            "tecs_load_factor_use_roll_sp": True,
            "guidance_min_ground_speed_mps": 1.0,
            "lateral_guidance_uncertainty_start_deg": 6.0,
            "lateral_guidance_uncertainty_full_deg": 18.0,
            "lateral_guidance_uncertainty_min_scale": 0.7,
        },
    )

    assert result["max_roll_deg"] == 40.0
    assert result["roll_kd"] == 0.70
    assert result["inner_elevon_roll_rate_limit_per_s"] == 15.0
    assert result["tecs_load_factor_use_roll_sp"] is True
    assert result["guidance_min_ground_speed_mps"] == 1.0
    assert result["lateral_guidance_uncertainty_start_deg"] == 6.0
    assert result["lateral_guidance_uncertainty_full_deg"] == 18.0
    assert result["lateral_guidance_uncertainty_min_scale"] == 0.7
