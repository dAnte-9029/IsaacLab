import math

import pytest

from flapping_bot.path_tracking.mission_primitives import Mission, MissionSegment
from flapping_bot.path_tracking.path_manager import PathManager, PathManagerCfg


def test_path_manager_produces_finite_curvature_and_progress():
    mission = Mission(
        segments=[
            MissionSegment(kind="straight", altitude_changes=False),
            MissionSegment(kind="turn", altitude_changes=False),
        ]
    )
    manager = PathManager(PathManagerCfg(max_roll_deg=35.0, max_flight_path_angle_deg=10.0), mission)
    query = manager.query(position_xy=(0.0, 0.0), altitude_m=10.0, speed_mps=7.0)
    assert query.curvature_m_inv == query.curvature_m_inv
    assert query.progress_s >= 0.0
    assert len(query.preview_points_xyz) == 5


def test_path_manager_projects_correctly_onto_initial_straight_segment():
    mission = Mission(segments=[MissionSegment(kind="straight", altitude_changes=False)])
    manager = PathManager(
        PathManagerCfg(
            max_roll_deg=35.0,
            max_flight_path_angle_deg=10.0,
            straight_length_m=40.0,
            turn_radius_m=20.0,
            loiter_radius_m=18.0,
            initial_altitude_m=10.0,
        ),
        mission,
    )

    query = manager.query(position_xy=(5.0, 2.0), altitude_m=10.0, speed_mps=7.0)

    assert query.closest_point_xyz == pytest.approx((5.0, 0.0, 10.0))
    assert query.tangent_xy == pytest.approx((1.0, 0.0))
    assert query.lateral_error_m == pytest.approx(2.0)
    assert query.progress_s == pytest.approx(5.0)
    assert query.curvature_m_inv == pytest.approx(0.0)
    assert query.preview_points_xyz[0][0] > query.closest_point_xyz[0]


def test_path_manager_returns_arc_geometry_on_turn_segment():
    mission = Mission(
        segments=[
            MissionSegment(kind="straight", altitude_changes=False),
            MissionSegment(kind="turn", altitude_changes=False),
        ]
    )
    manager = PathManager(
        PathManagerCfg(
            max_roll_deg=35.0,
            max_flight_path_angle_deg=10.0,
            straight_length_m=40.0,
            turn_radius_m=20.0,
            loiter_radius_m=18.0,
            initial_altitude_m=10.0,
        ),
        mission,
    )

    angle_rad = -0.25 * math.pi
    position_xy = (40.0 + 20.0 * math.cos(angle_rad), 20.0 + 20.0 * math.sin(angle_rad))
    query = manager.query(position_xy=position_xy, altitude_m=10.0, speed_mps=8.0)

    assert query.closest_point_xyz == pytest.approx((position_xy[0], position_xy[1], 10.0))
    assert query.tangent_xy == pytest.approx((math.sqrt(0.5), math.sqrt(0.5)), rel=1.0e-4, abs=1.0e-4)
    assert query.curvature_m_inv == pytest.approx(0.05)
    assert query.progress_s == pytest.approx(40.0 + 20.0 * (0.25 * math.pi), rel=1.0e-4)


def test_repeated_closed_loop_segments_do_not_jump_progress_on_tie():
    mission = Mission(
        segments=[
            MissionSegment(kind="loiter", altitude_changes=False),
            MissionSegment(kind="loiter", altitude_changes=False),
        ]
    )
    manager = PathManager(
        PathManagerCfg(
            max_roll_deg=35.0,
            max_flight_path_angle_deg=10.0,
            loiter_radius_m=20.0,
            initial_altitude_m=10.0,
        ),
        mission,
    )

    query = manager.query(position_xy=(0.0, 0.0), altitude_m=10.0, speed_mps=7.0)

    assert query.progress_s == pytest.approx(0.0)
