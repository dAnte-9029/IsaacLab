from scripts.flapping_rl.eval_suites import build_eval_cases, get_eval_suite_choices


def test_path_tracking_truth_nowind_suite_is_registered() -> None:
    assert "path_tracking_truth_nowind_v1" in get_eval_suite_choices()


def test_pure_rl_curriculum1_nowind_suite_has_fixed_heading_phase_grid() -> None:
    assert "pure_rl_curriculum1_nowind_v1" in get_eval_suite_choices()
    assert "pure_rl_curriculum1_nowind_v2" in get_eval_suite_choices()

    cases = build_eval_cases("pure_rl_curriculum1_nowind_v2")

    assert len(cases) == 1
    case = cases[0]
    pairs = set(zip(case["straight_line_heading_schedule_rad"], case["flap_phase_schedule_rad"], strict=True))
    assert len(pairs) == 16
    assert len(set(case["straight_line_heading_schedule_rad"])) == 4
    assert len(set(case["flap_phase_schedule_rad"])) == 4
    assert case["wind_enabled"] is False
    assert case["wind_ou_enabled"] is False
    assert case["teacher_state_source"] == "estimated"


def test_path_tracking_estimated_nowind_suite_is_registered() -> None:
    assert "path_tracking_estimated_nowind_v1" in get_eval_suite_choices()


def test_path_tracking_truth_primitives_nowind_suite_is_registered() -> None:
    assert "path_tracking_truth_primitives_nowind_v1" in get_eval_suite_choices()


def test_path_tracking_estimated_primitives_nowind_suite_is_registered() -> None:
    assert "path_tracking_estimated_primitives_nowind_v1" in get_eval_suite_choices()


def test_path_tracking_truth_nowind_suite_covers_required_cases() -> None:
    cases = build_eval_cases("path_tracking_truth_nowind_v1")

    case_names = {case["name"] for case in cases}
    assert {"straight_nowind", "loiter_nowind", "random_mission_nowind"} <= case_names

    for case in cases:
        assert case["wind_enabled"] is False
        assert case["wind_xy_mps"] == (0.0, 0.0)
        assert case["wind_ou_enabled"] is False
        assert case["wind_ou_sigma_xy_mps"] == (0.0, 0.0)
        assert case["teacher_state_source"] == "truth"
        assert case["policy_state_source"] == "truth"
    assert case["imu_source"] == "synthetic"


def test_pure_rl_longitudinal_suites_expose_promotion_and_diagnostic_schedules() -> None:
    expected_counts = {"c2a": 80, "c2b": 112, "c2c": 144}
    for stage_id, count in expected_counts.items():
        suite = f"pure_rl_longitudinal_{stage_id}_v1"
        assert suite in get_eval_suite_choices()
        promotion, diagnostic = build_eval_cases(suite)

        assert promotion["name"] == f"{stage_id}_promotion_grid"
        assert len(promotion["longitudinal_case_ids"]) == count
        assert len(promotion["straight_line_heading_schedule_rad"]) == count
        assert len(promotion["flap_phase_schedule_rad"]) == count
        assert len(promotion["longitudinal_task_schedule"]) == count
        assert len(promotion["longitudinal_slope_deg_schedule"]) == count
        assert set(promotion["longitudinal_entry_length_m_schedule"]) == {17.5}
        assert set(promotion["longitudinal_slope_length_m_schedule"]) == {25.0}
        assert promotion["promotion_eligible"] is True

        assert diagnostic["name"] == f"{stage_id}_signed_10deg_diagnostic"
        assert len(diagnostic["longitudinal_case_ids"]) == 32
        assert set(diagnostic["longitudinal_slope_deg_schedule"]) == {-10.0, 10.0}
        assert diagnostic["promotion_eligible"] is False


def test_spatial_suites_expose_exact_registered_schedules() -> None:
    expected_counts = {"c3a": 96, "c3b": 112, "c3c": 96}
    choices = get_eval_suite_choices()
    for stage, expected_count in expected_counts.items():
        suite = f"pure_rl_spatial_{stage}_v1"
        assert suite in choices
        cases = build_eval_cases(suite)
        assert len(cases) == 1
        case = cases[0]
        assert case["spatial_stage_id"] == stage
        assert len(case["spatial_case_ids"]) == expected_count
        for schedule_name in (
            "straight_line_heading_schedule_rad",
            "flap_phase_schedule_rad",
            "spatial_template_schedule",
            "spatial_geometry_roll_deg_schedule",
            "spatial_slope_deg_schedule",
            "spatial_turn_sign_schedule",
        ):
            assert len(case[schedule_name]) == expected_count


def test_path_tracking_estimated_nowind_suite_uses_estimated_teacher_contract() -> None:
    cases = build_eval_cases("path_tracking_estimated_nowind_v1")

    case_names = {case["name"] for case in cases}
    assert {"straight_nowind", "loiter_nowind", "random_mission_nowind"} <= case_names

    for case in cases:
        assert case["teacher_state_source"] == "estimated"
        assert case["policy_state_source"] == "estimated"
        assert case["imu_source"] == "synthetic"


def test_path_tracking_truth_nowind_suite_freezes_case_specific_mission_shapes() -> None:
    cases = {case["name"]: case for case in build_eval_cases("path_tracking_truth_nowind_v1")}

    straight_case = cases["straight_nowind"]
    assert straight_case["mission_seed"] == 101
    assert straight_case["mission_increment_seed_per_reset"] is False
    assert straight_case["mission_num_segments_min"] == 1
    assert straight_case["mission_num_segments_max"] == 1
    assert straight_case["mission_allow_straight"] is True
    assert straight_case["mission_allow_turn"] is False
    assert straight_case["mission_allow_loiter"] is False

    loiter_case = cases["loiter_nowind"]
    assert loiter_case["mission_seed"] == 202
    assert loiter_case["mission_increment_seed_per_reset"] is False
    assert loiter_case["mission_num_segments_min"] == 1
    assert loiter_case["mission_num_segments_max"] == 1
    assert loiter_case["mission_allow_straight"] is False
    assert loiter_case["mission_allow_turn"] is False
    assert loiter_case["mission_allow_loiter"] is True

    random_case = cases["random_mission_nowind"]
    assert random_case["mission_seed"] == 303
    assert random_case["mission_increment_seed_per_reset"] is False
    assert random_case["mission_num_segments_min"] == 3
    assert random_case["mission_num_segments_max"] == 3
    assert random_case["mission_allow_straight"] is True
    assert random_case["mission_allow_turn"] is True
    assert random_case["mission_allow_loiter"] is True


def test_path_tracking_truth_primitives_nowind_suite_covers_required_cases() -> None:
    cases = {case["name"]: case for case in build_eval_cases("path_tracking_truth_primitives_nowind_v1")}

    assert {"straight_primitive_nowind", "turn_primitive_nowind", "loiter_primitive_nowind"} <= set(cases.keys())

    straight_case = cases["straight_primitive_nowind"]
    assert straight_case["wind_enabled"] is False
    assert straight_case["teacher_state_source"] == "truth"
    assert straight_case["policy_state_source"] == "truth"
    assert straight_case["imu_source"] == "synthetic"
    assert straight_case["mission_seed"] == 111
    assert straight_case["mission_increment_seed_per_reset"] is False
    assert straight_case["mission_num_segments_min"] == 1
    assert straight_case["mission_num_segments_max"] == 1
    assert straight_case["mission_allow_straight"] is True
    assert straight_case["mission_allow_turn"] is False
    assert straight_case["mission_allow_loiter"] is False
    assert straight_case["mission_allow_climb_on_straight"] is False

    turn_case = cases["turn_primitive_nowind"]
    assert turn_case["wind_enabled"] is False
    assert turn_case["teacher_state_source"] == "truth"
    assert turn_case["policy_state_source"] == "truth"
    assert turn_case["imu_source"] == "synthetic"
    assert turn_case["mission_seed"] == 222
    assert turn_case["mission_increment_seed_per_reset"] is False
    assert turn_case["mission_num_segments_min"] == 1
    assert turn_case["mission_num_segments_max"] == 1
    assert turn_case["mission_allow_straight"] is False
    assert turn_case["mission_allow_turn"] is True
    assert turn_case["mission_allow_loiter"] is False
    assert turn_case["mission_allow_climb_on_straight"] is False

    loiter_case = cases["loiter_primitive_nowind"]
    assert loiter_case["wind_enabled"] is False
    assert loiter_case["teacher_state_source"] == "truth"
    assert loiter_case["policy_state_source"] == "truth"
    assert loiter_case["imu_source"] == "synthetic"
    assert loiter_case["mission_seed"] == 333
    assert loiter_case["mission_increment_seed_per_reset"] is False
    assert loiter_case["mission_num_segments_min"] == 1
    assert loiter_case["mission_num_segments_max"] == 1
    assert loiter_case["mission_allow_straight"] is False
    assert loiter_case["mission_allow_turn"] is False
    assert loiter_case["mission_allow_loiter"] is True
    assert loiter_case["mission_allow_climb_on_straight"] is False


def test_path_tracking_truth_primitives_nowind_suite_does_not_leak_training_curriculum_fields() -> None:
    cases = {case["name"]: case for case in build_eval_cases("path_tracking_truth_primitives_nowind_v1")}

    loiter_case = cases["loiter_primitive_nowind"]
    assert "loiter_curriculum_enabled" not in loiter_case
    assert "loiter_curriculum_stage_turns" not in loiter_case
    assert "loiter_curriculum_straight_rehearsal_prob" not in loiter_case


def test_path_tracking_estimated_primitives_nowind_suite_uses_estimated_teacher_contract() -> None:
    cases = {case["name"]: case for case in build_eval_cases("path_tracking_estimated_primitives_nowind_v1")}

    assert {"straight_primitive_nowind", "turn_primitive_nowind", "loiter_primitive_nowind"} <= set(cases.keys())
    for case in cases.values():
        assert case["teacher_state_source"] == "estimated"
        assert case["policy_state_source"] == "estimated"
        assert case["imu_source"] == "synthetic"
