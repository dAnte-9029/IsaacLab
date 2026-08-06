from scripts.flapping_rl.eval_suites import build_eval_cases, get_eval_suite_choices


def test_path_tracking_truth_nowind_suite_is_registered() -> None:
    assert "path_tracking_truth_nowind_v1" in get_eval_suite_choices()


def test_pure_rl_curriculum1_nowind_suite_has_fixed_heading_phase_grid() -> None:
    assert "pure_rl_curriculum1_nowind_v1" in get_eval_suite_choices()

    cases = build_eval_cases("pure_rl_curriculum1_nowind_v1")

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
