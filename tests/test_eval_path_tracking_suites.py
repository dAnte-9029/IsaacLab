from scripts.flapping_rl.eval_suites import get_eval_suite_choices


def test_path_tracking_eval_suite_is_registered() -> None:
    assert "path_tracking_standard" in get_eval_suite_choices()
