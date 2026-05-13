import numpy as np
import pandas as pd

from flapping_bot.analysis.short_window_rollout import RigidBodyParams, integrate_wrench_window


def _base_frame(sample_count: int, dt: float = 0.1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": np.arange(sample_count, dtype=float) * dt,
            "vehicle_local_position.x": np.zeros(sample_count),
            "vehicle_local_position.y": np.zeros(sample_count),
            "vehicle_local_position.z": np.zeros(sample_count),
            "vehicle_local_position.vx": np.zeros(sample_count),
            "vehicle_local_position.vy": np.zeros(sample_count),
            "vehicle_local_position.vz": np.zeros(sample_count),
            "vehicle_attitude.q[0]": np.ones(sample_count),
            "vehicle_attitude.q[1]": np.zeros(sample_count),
            "vehicle_attitude.q[2]": np.zeros(sample_count),
            "vehicle_attitude.q[3]": np.zeros(sample_count),
            "vehicle_angular_velocity.xyz[0]": np.zeros(sample_count),
            "vehicle_angular_velocity.xyz[1]": np.zeros(sample_count),
            "vehicle_angular_velocity.xyz[2]": np.zeros(sample_count),
        }
    )


def test_integrate_hover_force_preserves_level_state() -> None:
    params = RigidBodyParams.default_flapper()
    frame = _base_frame(11)
    wrench = pd.DataFrame(
        {
            "fx_b": np.zeros(len(frame)),
            "fy_b": np.zeros(len(frame)),
            "fz_b": np.full(len(frame), -params.mass_kg * params.gravity_m_s2),
            "mx_b": np.zeros(len(frame)),
            "my_b": np.zeros(len(frame)),
            "mz_b": np.zeros(len(frame)),
        }
    )

    rollout = integrate_wrench_window(frame, wrench, params=params)

    np.testing.assert_allclose(rollout["pred_x"], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(rollout["pred_z"], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(rollout["pred_vx"], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(rollout["pred_vz"], 0.0, atol=1.0e-12)


def test_integrate_constant_forward_force_matches_constant_acceleration() -> None:
    params = RigidBodyParams.default_flapper()
    frame = _base_frame(11)
    wrench = pd.DataFrame(
        {
            "fx_b": np.full(len(frame), params.mass_kg),
            "fy_b": np.zeros(len(frame)),
            "fz_b": np.full(len(frame), -params.mass_kg * params.gravity_m_s2),
            "mx_b": np.zeros(len(frame)),
            "my_b": np.zeros(len(frame)),
            "mz_b": np.zeros(len(frame)),
        }
    )

    rollout = integrate_wrench_window(frame, wrench, params=params)

    assert rollout["pred_vx"].iloc[-1] == np.float64(1.0)
    assert rollout["pred_x"].iloc[-1] == np.float64(0.5)


def test_integrate_marks_divergence_instead_of_raising_on_unphysical_moment() -> None:
    params = RigidBodyParams.default_flapper()
    frame = _base_frame(4)
    wrench = pd.DataFrame(
        {
            "fx_b": np.zeros(len(frame)),
            "fy_b": np.zeros(len(frame)),
            "fz_b": np.full(len(frame), -params.mass_kg * params.gravity_m_s2),
            "mx_b": np.full(len(frame), 1.0e6),
            "my_b": np.zeros(len(frame)),
            "mz_b": np.zeros(len(frame)),
        }
    )

    rollout = integrate_wrench_window(frame, wrench, params=params, max_omega_radps=10.0)

    assert rollout["diverged"].any()
