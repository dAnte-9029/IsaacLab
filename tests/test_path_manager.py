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


def test_path_manager_does_not_skip_initial_loiter_when_followed_by_straight():
    mission = Mission(
        segments=[
            MissionSegment(kind="loiter", altitude_changes=False),
            MissionSegment(kind="straight", altitude_changes=False),
        ]
    )
    manager = PathManager(
        PathManagerCfg(
            max_roll_deg=35.0,
            max_flight_path_angle_deg=10.0,
            straight_length_m=60.0,
            loiter_radius_m=20.0,
            initial_altitude_m=10.0,
        ),
        mission,
    )

    query = manager.query(position_xy=(0.0, 0.0), altitude_m=10.0, speed_mps=7.0)

    assert query.progress_s == pytest.approx(0.0)


def test_path_manager_closed_loop_wrap_stays_near_current_loop_progress():
    mission = Mission(
        segments=[
            MissionSegment(kind="straight", altitude_changes=False),
            MissionSegment(kind="loiter", altitude_changes=False),
        ]
    )
    manager = PathManager(
        PathManagerCfg(
            max_roll_deg=35.0,
            max_flight_path_angle_deg=10.0,
            straight_length_m=60.0,
            loiter_radius_m=20.0,
            initial_altitude_m=10.0,
        ),
        mission,
    )

    manager._last_progress_s = 80.0

    wrapped_query = manager.query(position_xy=(59.86, 0.00014), altitude_m=10.0, speed_mps=7.0)

    assert wrapped_query.progress_s == pytest.approx(80.0, abs=0.5)


def test_path_manager_advances_into_loiter_instead_of_sticking_to_entry():
    mission = Mission(
        segments=[
            MissionSegment(kind="straight", altitude_changes=False),
            MissionSegment(kind="loiter", altitude_changes=False),
        ]
    )
    manager = PathManager(
        PathManagerCfg(
            max_roll_deg=35.0,
            max_flight_path_angle_deg=10.0,
            straight_length_m=60.0,
            loiter_radius_m=20.0,
            initial_altitude_m=10.0,
        ),
        mission,
    )

    manager._last_progress_s = 60.0

    query = manager.query(position_xy=(61.0, 0.0), altitude_m=10.0, speed_mps=7.0)

    assert query.progress_s > 60.0


def test_path_manager_does_not_teleport_from_loiter_to_following_straight():
    mission = Mission(
        segments=[
            MissionSegment(kind="loiter", altitude_changes=False),
            MissionSegment(kind="straight", altitude_changes=False),
        ]
    )
    manager = PathManager(
        PathManagerCfg(
            max_roll_deg=35.0,
            max_flight_path_angle_deg=10.0,
            straight_length_m=60.0,
            loiter_radius_m=20.0,
            initial_altitude_m=10.0,
        ),
        mission,
    )

    query = manager.query(position_xy=(1.0, 0.0), altitude_m=10.0, speed_mps=7.0)

    assert query.progress_s < 5.0


def test_path_manager_repeated_loiter_does_not_jump_a_full_loop():
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

    progress = [manager.query(position_xy=position_xy, altitude_m=10.0, speed_mps=7.0).progress_s for position_xy in [(0.0, 0.0), (1.0, 0.0)]]
    progress_deltas = [end - start for start, end in zip(progress, progress[1:])]

    assert max(progress_deltas) < 5.0


def test_path_manager_loiter_to_straight_overlap_stays_on_active_loiter() -> None:
    mission = Mission(
        segments=[
            MissionSegment(kind="loiter", altitude_changes=False),
            MissionSegment(kind="straight", altitude_changes=False),
        ]
    )
    manager = PathManager(
        PathManagerCfg(
            max_roll_deg=35.0,
            max_flight_path_angle_deg=10.0,
            straight_length_m=60.0,
            loiter_radius_m=20.0,
            initial_altitude_m=10.0,
        ),
        mission,
    )

    manager._last_progress_s = 1.9602609872817993

    query = manager.query(position_xy=(2.0234274864196777, 0.028812089934945107), altitude_m=10.0, speed_mps=7.0)

    assert query.progress_s - 1.9602609872817993 < 5.0
    assert query.progress_s == pytest.approx(2.0194554257282116, abs=0.5)


def test_path_manager_does_not_skip_multiple_segments_across_repeated_loiters() -> None:
    mission = Mission(
        segments=[
            MissionSegment(kind="loiter", altitude_changes=False),
            MissionSegment(kind="loiter", altitude_changes=False),
            MissionSegment(kind="straight", altitude_changes=False),
        ]
    )
    manager = PathManager(
        PathManagerCfg(
            max_roll_deg=35.0,
            max_flight_path_angle_deg=10.0,
            straight_length_m=60.0,
            loiter_radius_m=20.0,
            initial_altitude_m=10.0,
        ),
        mission,
    )

    manager._last_progress_s = 1.9602609872817993

    query = manager.query(position_xy=(2.0234274864196777, 0.028812089934945107), altitude_m=10.0, speed_mps=7.0)

    assert query.progress_s - 1.9602609872817993 < 5.0
    assert query.progress_s < 10.0


def test_path_manager_turn_loiter_straight_overlap_does_not_teleport_to_straight() -> None:
    mission = Mission(
        segments=[
            MissionSegment(kind="turn", altitude_changes=False),
            MissionSegment(kind="loiter", altitude_changes=False),
            MissionSegment(kind="straight", altitude_changes=False),
            MissionSegment(kind="loiter", altitude_changes=False),
        ]
    )
    manager = PathManager(
        PathManagerCfg(
            max_roll_deg=35.0,
            max_flight_path_angle_deg=10.0,
            straight_length_m=60.0,
            turn_radius_m=20.0,
            loiter_radius_m=20.0,
            turn_sweep_deg=90.0,
            initial_altitude_m=10.0,
        ),
        mission,
    )

    manager._last_progress_s = 33.33671569824219

    query = manager.query(position_xy=(19.60279083251953, 22.022756576538086), altitude_m=10.0, speed_mps=7.0)

    assert query.progress_s - 33.33671569824219 < 5.0
    assert query.progress_s == pytest.approx(33.39282894432516, abs=0.5)


def test_path_manager_repeated_loiter_boundary_stays_on_adjacent_loop() -> None:
    mission = Mission(
        segments=[
            MissionSegment(kind="loiter", altitude_changes=False),
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

    manager._last_progress_s = 127.54136657714844

    query = manager.query(position_xy=(1.9760551452636719, 0.37093302607536316), altitude_m=10.0, speed_mps=7.0)

    assert query.progress_s - 127.54136657714844 < 5.0
    assert query.progress_s == pytest.approx(127.59659974844565, abs=0.75)


def test_path_manager_multiturn_loiter_advances_past_full_loop_boundary() -> None:
    mission = Mission(segments=[MissionSegment(kind="loiter", altitude_changes=False)])
    manager = PathManager(
        PathManagerCfg(
            max_roll_deg=35.0,
            max_flight_path_angle_deg=10.0,
            loiter_radius_m=20.0,
            loiter_turns=1.5,
            initial_altitude_m=10.0,
        ),
        mission,
    )

    one_loop_progress_s = 2.0 * math.pi * 20.0
    manager._last_progress_s = one_loop_progress_s

    position_xyz = manager.sample(one_loop_progress_s + 0.5)
    query = manager.query(position_xy=position_xyz[:2], altitude_m=10.0, speed_mps=7.0)

    assert query.progress_s > one_loop_progress_s
    assert query.progress_s == pytest.approx(one_loop_progress_s + 0.5, abs=0.25)


def test_path_manager_supports_explicit_climb_and_descent_segments():
    climb_mission = Mission(segments=[MissionSegment(kind="straight", altitude_changes=True, altitude_direction=1)])
    descent_mission = Mission(segments=[MissionSegment(kind="straight", altitude_changes=True, altitude_direction=-1)])
    cfg = PathManagerCfg(
        max_roll_deg=35.0,
        max_flight_path_angle_deg=10.0,
        straight_length_m=40.0,
        initial_altitude_m=10.0,
    )

    climb_manager = PathManager(cfg, climb_mission)
    descent_manager = PathManager(cfg, descent_mission)

    climb_start = climb_manager.sample(0.0)
    climb_end = climb_manager.sample(climb_manager.total_length_m)
    descent_start = descent_manager.sample(0.0)
    descent_end = descent_manager.sample(descent_manager.total_length_m)

    assert climb_end[2] > climb_start[2]
    assert descent_end[2] < descent_start[2]


def test_path_manager_reports_scored_progress_after_leading_warmup_segment() -> None:
    mission = Mission(
        segments=[
            MissionSegment(kind="straight", altitude_changes=False, counts_toward_progress=False),
            MissionSegment(kind="turn", altitude_changes=False),
        ]
    )
    cfg = PathManagerCfg(
        max_roll_deg=35.0,
        max_flight_path_angle_deg=10.0,
        straight_length_m=25.0,
        turn_radius_m=20.0,
        turn_sweep_deg=90.0,
        initial_altitude_m=10.0,
    )

    manager = PathManager(cfg, mission)

    assert manager.score_start_progress_s == pytest.approx(25.0)
    assert manager.scored_total_length_m == pytest.approx(20.0 * math.pi * 0.5, rel=1.0e-6)
