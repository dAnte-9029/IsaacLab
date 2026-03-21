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
