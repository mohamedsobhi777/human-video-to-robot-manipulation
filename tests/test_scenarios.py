import numpy as np
import pytest
from wpr.geometry import Task, EpisodeMetrics, route_waypoints, route_polyline, route_progress
from wpr.scenarios import (Scenario, make_scenario, validate_scenario, RANGES, SHIFTED_FRICTION)


def test_scenarios_are_deterministic_and_splits_independent():
    a, b = make_scenario("test", 7, "left"), make_scenario("test", 7, "left")
    assert a == b
    assert make_scenario("val", 7, "left").task != a.task


def test_routes_and_disturbances_share_the_layout_of_a_seed():
    tasks = {repr(make_scenario("dev", 3, r, d).task) for r in ("left", "right")
             for d in ("none", "shove", "displace")}
    assert len(tasks) == 1


def test_sampled_layouts_are_in_range_and_route_clear():
    for seed in range(200):
        s = make_scenario("dev", seed, "right" if seed % 2 else "left", "shove")
        assert validate_scenario(s)
        assert RANGES["friction"][0] <= s.task.friction <= RANGES["friction"][1]
        assert 0.3 <= s.disturbance["progress_fraction"] <= 0.7
        # Larger-x routes must stay inside the workspace verified for the arm.
        assert max(route_waypoints(s.task, s.route)[:, 0]) <= 0.69


def test_shifted_split_leaves_training_friction_range():
    for seed in range(50):
        f = make_scenario("shifted", seed, "left").task.friction
        assert any(lo <= f <= hi for lo, hi in SHIFTED_FRICTION)


def test_scenario_round_trips_through_json_dict():
    s = make_scenario("test", 11, "right", "displace")
    assert Scenario.from_dict(s.to_dict()) == s


def test_route_progress_is_arc_length():
    task = Task()
    poly = route_polyline(task, route_waypoints(task, "left"))
    assert route_progress(task.start, poly) == pytest.approx(0)
    assert route_progress((0.33, 0.0), poly) == pytest.approx(0.17 + 0.20)
    assert route_progress(task.goal, poly) == pytest.approx(0.17*2 + 0.40)


def contacts(**kw):
    return {"robot_obstacle": 0., "cube_obstacle": 0., "robot_table": 0., **kw}


def run_to_goal(m, task, **kw):
    for p in m.waypoints: m.update([*p, 0.02], 0, 0, contacts=contacts())
    m.update([*task.goal, 0.02], 0, 0, contacts=contacts(**kw))
    for _ in range(task.settle_steps): m.update([*task.goal, 0.02], 0, 0, contacts=contacts())
    return m.result()


@pytest.mark.parametrize("key,fails", [("robot_obstacle", True), ("cube_obstacle", True),
                                       ("robot_table", False)])
def test_physical_contact_scoring(key, fails):
    task = Task(settle_steps=3)
    result = run_to_goal(EpisodeMetrics(task, route_waypoints(task, "left")), task, **{key: 2.0})
    assert result["contact_scored"]
    assert result["success"] is not fails
    assert result[f"contact_steps_{key}"] == 1


def test_contact_noise_below_threshold_is_ignored():
    task = Task(settle_steps=3)
    result = run_to_goal(EpisodeMetrics(task, route_waypoints(task, "left")), task, robot_obstacle=0.005)
    assert result["route_success"] and not result["obstacle_contact"]


def test_legacy_updates_report_contact_unscored():
    task = Task(settle_steps=2)
    m = EpisodeMetrics(task, [task.goal])
    for _ in range(3): m.update([*task.goal, 0.02], 0, 0)
    assert m.result()["success"] and not m.result()["contact_scored"]


def test_nonfinite_contact_is_rejected():
    with pytest.raises(ValueError):
        EpisodeMetrics(Task(), []).update([0.5, 0, 0.02], 0, 0, contacts=contacts(robot_table=np.nan))
