"""Focused numerical checks for offline analysis coordinates and KL."""

import numpy as np

from flapping_bot.analysis.rl_forgetting_mechanisms import axes, coverage, features, gaussian_kl


def test_tail_coordinates_keep_actuator_order_and_are_invertible():
    a = np.array([[.3, -.2, .6, -.4], [-.1, .5, -.7, -.7]])
    b = axes(a)
    np.testing.assert_allclose(b[:, :4], a)
    np.testing.assert_allclose(b[:, 4] + b[:, 5], a[:, 2])
    np.testing.assert_allclose(b[:, 4] - b[:, 5], a[:, 3])


def test_gaussian_kl_orientation_and_channel_sum():
    a = np.zeros((2, 4)); b = np.ones((2, 4))
    assert gaussian_kl(a, np.ones(4), a, np.ones(4)) == 0
    assert gaussian_kl(a, np.ones(4), b, np.ones(4)) == 2
    expected = 4 * (np.log(2) + 1/8 - .5)
    np.testing.assert_allclose(gaussian_kl(a, np.ones(4), a, np.ones(4)*2), expected)


def test_physical_features_ignore_yaw_and_quaternion_double_cover():
    obs = np.zeros((3, 555), dtype=np.float32)
    frames = obs[:, :420].reshape(3, 30, 14)
    frames[0, :, 0] = 1
    frames[1, :, 3] = 1  # Same level attitude at 180-degree yaw.
    frames[2, :, 0] = -1
    frames[:, :, 4] = .5
    frames[:, :, 7] = .2
    frames[:, :, 11] = .2
    obs[:, 540:] = .2
    physical, _, _ = features(obs)
    np.testing.assert_allclose(physical[0], physical[1], atol=1e-6)
    np.testing.assert_allclose(physical[0], physical[2], atol=1e-6)
    np.testing.assert_allclose(physical[0, [2, 5, 8, 9]], [6, 1, 3, 1])
    result = coverage(obs, obs[:1])
    assert result["fraction_within_tolerance"]["0.5"] == 1
