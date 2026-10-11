import numpy as np
import pytest
from wpr.geometry import Task, route_waypoints, square_overlaps_obstacle, EpisodeMetrics


def test_routes_share_goal_and_avoid_obstacle():
    task = Task()
    for side in ("left", "right"):
        points = np.vstack([task.start, route_waypoints(task, side)])
        assert np.allclose(points[-1], task.goal)
        for a, b in zip(points[:-1], points[1:]):
            for t in np.linspace(0, 1, 100):
                assert not square_overlaps_obstacle(a*(1-t)+b*t, np.pi/4, task)
    assert not np.allclose(route_waypoints(task,"left"), route_waypoints(task,"right"))


def test_overlap_accounts_for_rotation():
    task = Task()
    assert square_overlaps_obstacle(task.obstacle, 0, task)
    assert not square_overlaps_obstacle((0.575,0), 0, task)
    assert square_overlaps_obstacle((0.575,0), np.pi/4, task)


def test_success_requires_settling_and_does_not_hide_collision():
    task = Task(settle_steps=3)
    m = EpisodeMetrics(task, route_waypoints(task,"left"))
    for p in route_waypoints(task,"left"):
        m.update([*p,0.02],0,0)
    assert not m.result()["success"]
    for _ in range(3): m.update([*task.goal,0.02],0,0)
    assert m.result()["route_success"]
    m.update([*task.obstacle,0.02],0,0)
    for _ in range(3): m.update([*task.goal,0.02],0,0)
    assert not m.result()["success"]


def test_goal_only_does_not_count_as_route_success():
    task = Task(settle_steps=2)
    m = EpisodeMetrics(task,route_waypoints(task,"right"))
    for _ in range(3): m.update([*task.goal,0.02],0,0)
    assert m.result()["success"] and not m.result()["route_success"]


def test_invalid_state_is_rejected():
    with pytest.raises(ValueError):
        EpisodeMetrics(Task(),[]).update([np.nan,0,0],0,0)


@pytest.mark.parametrize("height,angular_speed,tilt",[(0.04,0,0),(0.02,0.5,0),(0.02,0,0.5)])
def test_goal_requires_supported_nonrotating_object(height,angular_speed,tilt):
    task=Task(settle_steps=2)
    m=EpisodeMetrics(task,[task.goal])
    for _ in range(3): m.update([*task.goal,height],0,0,angular_speed,tilt)
    assert not m.result()["success"]
