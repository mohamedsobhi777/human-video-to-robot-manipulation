import numpy as np
from wpr.geometry import Task, route_waypoints, rect_overlaps_obstacle
from wpr.route_controller import RoutePusher, HAND_HALF, STANDOFF, wrap_half_turn


def state(xy, hand, hand_yaw=0.0):
    return {"object": np.array([*xy, 0.02]), "hand": np.array(hand, dtype=float), "hand_yaw": hand_yaw}


def test_wide_hand_axis_follows_push_direction():
    p = RoutePusher(route_waypoints(Task(), "left"), Task())
    assert abs(p.yaw_for(np.array([0., 1.]))) < 1e-9
    assert abs(abs(p.yaw_for(np.array([1., 0.]))) - np.pi/2) < 1e-9


def test_plans_never_put_the_hand_over_the_obstacle():
    task = Task()
    rng = np.random.default_rng(0)
    for _ in range(300):
        obj = np.array(task.obstacle) + rng.uniform([-0.2, -0.2], [0.2, 0.2])
        if rect_overlaps_obstacle(obj, (0.03, 0.03), 0, task): continue
        p = RoutePusher(route_waypoints(task, "right"), task)
        p.index = int(rng.integers(0, 3))
        plan = p.choose(obj)
        d, yaw = plan["direction"], plan["yaw"]
        if p.feasible(obj, d, yaw):
            assert not rect_overlaps_obstacle(obj - d*STANDOFF, HAND_HALF, yaw, task)


def test_cube_next_to_obstacle_is_pushed_away_or_along_it():
    task = Task()
    p = RoutePusher(route_waypoints(task, "left"), task)
    p.index = 1  # tracking the segment beside the obstacle
    obj = np.array([task.obstacle[0]-0.075, 0.0])  # 2.5 cm from the obstacle face
    d = p.choose(obj)["direction"]
    assert d[0] <= 1e-9  # never toward the obstacle face


def test_lateral_drift_is_corrected_toward_route_line():
    task = Task()
    p = RoutePusher(route_waypoints(task, "left"), task)
    p.index = 1  # segment x = 0.33, travelling +y
    wanted, _ = p.desired_direction(np.array([0.36, -0.05]))
    # Steers back toward the line, but never more than 15 degrees off the segment.
    assert -np.sin(np.radians(15)) - 1e-9 <= wanted[0] < -0.2 and wanted[1] > 0


def test_lift_then_approach_from_behind():
    task = Task()
    p = RoutePusher(route_waypoints(task, "left"), task)
    cmd = p.command(state(task.start, [0.5, -0.2, 0.36]))
    assert p.phase == "lift" and cmd[2] < 0.36
    p.command(state(task.start, [0.5, -0.2, 0.24]))
    assert p.phase == "approach"
    direction = p.plan["direction"]
    assert direction[0] < -0.9 and p.plan["tracking"]  # first segment heads to smaller x


def test_pushing_face_stays_parallel_to_cube_face():
    task = Task()
    p = RoutePusher(route_waypoints(task, "left"), task)
    along, across = p.face_yaws(np.array([np.cos(0.2), np.sin(0.2)]), cube_yaw=0.0)
    assert abs(abs(along) - np.pi/2) < 1e-9 and abs(across) < 1e-9


def test_yaw_command_is_rate_limited_and_symmetric():
    assert abs(wrap_half_turn(np.pi) - 0) < 1e-9
    task = Task()
    p = RoutePusher(route_waypoints(task, "left"), task)
    p.phase = "approach"
    p.plan = dict(direction=np.array([1., 0.]), yaw=-np.pi/2, planned_at=np.array(task.start), tracking=True)
    cmd = p.command(state(task.start, [0.3, -0.2, 0.24], hand_yaw=0.0))
    assert abs(cmd[3]) <= 0.08 + 1e-9


def test_recovery_push_near_obstacle_corner_does_not_loop():
    """Regression: a turned recovery push used to be aborted on its first step."""
    task = Task(obstacle=(0.507, 0.0))
    p = RoutePusher(route_waypoints(task, "left"), task)
    p.index = 1
    obj = np.array([0.436, 0.115])
    plan = p.choose(obj)
    assert not plan["tracking"] and plan["direction"][0] < 0
    p.phase, p.plan = "push", plan
    hand = np.array([*(obj - plan["direction"]*0.03), 0.125])
    p.command(state(obj, hand, plan["yaw"]))
    assert p.phase == "push"
