import torch

try:
    from flapping_bot.px4_like.rl_training_utils import compute_recovery_teacher_mask
except ModuleNotFoundError:  # pragma: no cover - compatibility import path
    from flapping_bot.flapping_bot.px4_like.rl_training_utils import compute_recovery_teacher_mask


def test_recovery_mask_triggers_on_large_error_or_low_speed():
    mask = compute_recovery_teacher_mask(
        lateral_error=torch.tensor([0.1, 5.0]),
        height_error=torch.tensor([0.2, 0.2]),
        airspeed=torch.tensor([7.0, 2.0]),
        tilt_deg=torch.tensor([10.0, 20.0]),
        ang_rate_deg_s=torch.tensor([10.0, 20.0]),
        lateral_error_trigger_m=2.0,
        height_error_trigger_m=1.0,
        min_safe_airspeed_mps=4.0,
        tilt_trigger_deg=45.0,
        ang_rate_trigger_deg_s=120.0,
    )
    assert mask.tolist() == [False, True]
