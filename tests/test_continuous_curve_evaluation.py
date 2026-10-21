"""Fail-closed episode scoring independent of Isaac Sim."""
import numpy as np

from flapping_bot.analysis.continuous_curve_evaluation import METRICS, score_episode, summarize


def test_completion_does_not_replace_survival_or_tracking():
    trace = {key: np.zeros(10) for key in METRICS}
    trace.update(time_s=np.linspace(0, 20, 10), velocity_world_mps=np.zeros((10, 3)),
                 requested_frequency_hz=np.zeros(10), applied_frequency_hz=np.zeros(10))
    trace["along_track_progress_m"] = np.linspace(0, 141, 10)
    assert score_episode(trace, terminated=False, timed_out=True)["success"]
    assert not score_episode(trace, terminated=True, timed_out=False)["success"]
    assert not score_episode(trace, terminated=False, timed_out=False)["success"]
    trace["height_error_m"][:] = .6
    assert not score_episode(trace, terminated=False, timed_out=True)["success"]
    trace["height_error_m"][0] = np.nan
    assert not score_episode(trace, terminated=False, timed_out=True)["finite"]


def test_missing_duplicate_and_incomplete_cases_fail_closed():
    row = dict(curve_id="a", family="coupled", horizontal_mirror=1, vertical_mirror=1,
               terminated=False, time_out=True, survival=True, completion=True, success=True)
    assert summarize([row], ["a"])["evaluation_complete"]
    assert not summarize([row], ["a", "b"])["evaluation_complete"]
    assert not summarize([row, row], ["a", "b"])["evaluation_complete"]
    assert not summarize([{**row, "time_out": False}], ["a"])["evaluation_complete"]
    assert not summarize([row], ["a"])["promotion_eligible"]


def test_preview_time_scale_preserves_five_points_and_clamps_endpoint():
    import torch
    import pytest
    from flapping_bot.analysis.continuous_curve_evaluation import resample_preview_points
    points = torch.zeros((1, 101, 3), dtype=torch.float64)
    points[0, :, 0] = torch.arange(101, dtype=torch.float64) * .25
    progress = torch.tensor([2.], dtype=torch.float64)
    speed = torch.tensor([8.], dtype=torch.float64)
    short = resample_preview_points(points, .25, progress, speed, 1.)
    long = resample_preview_points(points, .25, progress, speed, 2.)
    assert short.shape == long.shape == (1, 5, 3)
    torch.testing.assert_close(long[:, :, 0] - 2, 2 * (short[:, :, 0] - 2))
    end = resample_preview_points(points, .25, progress + 22, speed, 2.)
    assert end.max() == 25
    for invalid in (0, -1, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            resample_preview_points(points, .25, progress, speed, invalid)


def test_channel_replacement_preserves_sources_and_rejects_invalid_actions():
    import torch
    import pytest
    from flapping_bot.analysis.continuous_curve_evaluation import combine_frequency_tail_actions
    a = torch.tensor([[1., 2., 3., 4.]], dtype=torch.float64)
    b = torch.tensor([[5., 6., 7., 8.]], dtype=torch.float64)
    result = combine_frequency_tail_actions(a, b)
    torch.testing.assert_close(result, torch.tensor([[1., 6., 7., 8.]], dtype=torch.float64))
    torch.testing.assert_close(combine_frequency_tail_actions(a, a), a)
    result.zero_()
    assert a[0, 0] == 1 and b[0, 1] == 6
    for invalid in (b.float(), b[:, :3], b * float('nan')):
        with pytest.raises(ValueError):
            combine_frequency_tail_actions(a, invalid)


def test_individual_tail_replacement():
    import torch
    import pytest
    from flapping_bot.analysis.continuous_curve_evaluation import combine_frequency_tail_actions
    a = torch.zeros(2, 4)
    b = torch.ones(2, 4)
    for channels in ((1,), (2,), (3,), (2, 3)):
        result = combine_frequency_tail_actions(a, b, channels)
        for index in range(4):
            assert (result[:, index] == int(index in channels)).all()
    for channels in ((), (0,), (1, 1), (4,)):
        with pytest.raises(ValueError):
            combine_frequency_tail_actions(a, b, channels)
