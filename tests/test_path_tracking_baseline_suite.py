from scripts.flapping_px4.run_path_tracking_baseline_suite import build_phase_names


def test_baseline_suite_includes_all_primitive_phases() -> None:
    names = build_phase_names()
    assert "level_straight" in names
    assert "climb_straight" in names
    assert "descent_straight" in names
    assert "level_turn" in names
    assert "level_loiter" in names
