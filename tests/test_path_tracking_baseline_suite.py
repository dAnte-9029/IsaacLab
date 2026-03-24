from scripts.flapping_px4.run_path_tracking_baseline_suite import build_phase_names, build_suite_cases


def test_baseline_suite_includes_all_primitive_phases() -> None:
    names = build_phase_names()
    assert "level_straight" in names
    assert "climb_straight" in names
    assert "descent_straight" in names
    assert "level_turn" in names
    assert "level_loiter" in names
    assert "multi_segment" in names


def test_baseline_suite_cases_cover_each_phase_once() -> None:
    cases = build_suite_cases()
    phases = [case["phase"] for case in cases]
    assert phases == build_phase_names()
    assert all(int(case["steps"]) > 0 for case in cases)
