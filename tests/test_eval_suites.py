from scripts.flapping_rl.eval_suites import build_eval_cases, get_eval_suite_choices


def test_path_tracking_truth_nowind_suite_is_registered() -> None:
    assert "path_tracking_truth_nowind_v1" in get_eval_suite_choices()


def test_path_tracking_truth_nowind_suite_covers_required_cases() -> None:
    cases = build_eval_cases("path_tracking_truth_nowind_v1")

    case_names = {case["name"] for case in cases}
    assert {"straight_nowind", "loiter_nowind", "random_mission_nowind"} <= case_names

    for case in cases:
        assert case["wind_enabled"] is False
        assert case["wind_xy_mps"] == (0.0, 0.0)
        assert case["wind_ou_enabled"] is False
        assert case["wind_ou_sigma_xy_mps"] == (0.0, 0.0)


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
