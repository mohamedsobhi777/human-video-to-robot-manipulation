"""Planar task geometry and explicit, independently testable success checks."""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class Task:
    start: tuple = (0.50, -0.20)
    goal: tuple = (0.50, 0.20)
    obstacle: tuple = (0.50, 0.0)
    obstacle_size: tuple = (0.10, 0.12, 0.07)
    cube_size: float = 0.04
    cube_mass: float = 0.08
    friction: float = 0.45
    goal_tolerance: float = 0.025
    waypoint_tolerance: float = 0.025
    settle_steps: int = 20
    max_control_steps: int = 1800
    control_dt: float = 1 / 40
    route_offset: float = 0.17

    def to_dict(self):
        return asdict(self)


def route_waypoints(task: Task, route: str) -> np.ndarray:
    """Synthetic routes: left is smaller robot-frame X in the overhead plot."""
    if route not in ("left", "right"):
        raise ValueError(f"Unknown route: {route}")
    x = task.obstacle[0] + (-task.route_offset if route == "left" else task.route_offset)
    return np.array([[x, task.start[1]], [x, task.goal[1]], task.goal], dtype=float)


def rect_overlaps_obstacle(xy, half_extents, yaw: float, task: Task, margin: float = 0.0) -> bool:
    """Separating-axis test for an oriented rectangle vs the (inflated) obstacle footprint."""
    c, s = np.cos(yaw), np.sin(yaw)
    axes = np.array([[1., 0.], [0., 1.], [c, s], [-s, c]])
    rot = np.array([[c, -s], [s, c]])
    corners = np.array([[-1,-1], [-1,1], [1,-1], [1,1]], dtype=float)
    rect = corners * np.asarray(half_extents) @ rot.T + np.asarray(xy)
    obstacle = corners * (np.asarray(task.obstacle_size[:2]) / 2 + margin) + task.obstacle
    for axis in axes:
        a, b = rect @ axis, obstacle @ axis
        if a.max() < b.min() or b.max() < a.min():
            return False
    return True


def square_overlaps_obstacle(xy, yaw: float, task: Task) -> bool:
    """Separating-axis test for oriented cube footprint vs rectangular obstacle.

    This measures geometric footprint overlap, not contact force or robot collision.
    """
    return rect_overlaps_obstacle(xy, (task.cube_size / 2,) * 2, yaw, task)


def obstacle_distance(xy, task: Task) -> float:
    """Planar distance from a point to the obstacle footprint (0 inside)."""
    d = np.abs(np.asarray(xy) - task.obstacle) - np.asarray(task.obstacle_size[:2]) / 2
    return float(np.linalg.norm(np.maximum(d, 0)))


def route_polyline(task: Task, waypoints) -> np.ndarray:
    return np.vstack([task.start, np.asarray(waypoints)])


def route_progress(xy, polyline) -> float:
    """Arc length of the closest point on the route polyline to ``xy``."""
    xy = np.asarray(xy, dtype=float)
    best, travelled, progress = float("inf"), 0.0, 0.0
    for a, b in zip(polyline[:-1], polyline[1:]):
        seg = b - a
        length = float(np.linalg.norm(seg))
        t = float(np.clip((xy - a) @ seg / max(length**2, 1e-12), 0, 1))
        d = float(np.linalg.norm(a + t*seg - xy))
        if d < best:
            best, progress = d, travelled + t*length
        travelled += length
    return progress


# Contact forces at or below this norm (N) are treated as numerical noise.
CONTACT_THRESHOLD_N = 0.01
CONTACT_KEYS = ("robot_obstacle", "cube_obstacle", "robot_table")


class EpisodeMetrics:
    def __init__(self, task: Task, waypoints):
        self.task = task
        self.waypoints = np.asarray(waypoints)
        self.next_waypoint = 0
        self.overlap = False
        # None until physical contact readings are supplied; legacy runs had none.
        self.contact_max = None
        self.contact_steps = None
        self.settled = 0
        self.max_lift = 0.0
        self.steps = 0
        self.final_error = float("inf")

    def update(self, xyz, yaw, speed, angular_speed=0.0, tilt=0.0, contacts=None):
        xyz = np.asarray(xyz)
        if not np.isfinite(xyz).all() or not np.isfinite([yaw,speed,angular_speed,tilt]).all():
            raise ValueError("Nonfinite state in evaluator")
        self.steps += 1
        if contacts is not None:
            if self.contact_max is None:
                self.contact_max = dict.fromkeys(CONTACT_KEYS, 0.0)
                self.contact_steps = dict.fromkeys(CONTACT_KEYS, 0)
            for key in CONTACT_KEYS:
                force = float(contacts[key])
                if not np.isfinite(force): raise ValueError("Nonfinite contact force")
                self.contact_max[key] = max(self.contact_max[key], force)
                self.contact_steps[key] += int(force > CONTACT_THRESHOLD_N)
        self.overlap |= square_overlaps_obstacle(xyz[:2], yaw, self.task)
        self.max_lift = max(self.max_lift, float(xyz[2] - self.task.cube_size / 2))
        if self.next_waypoint < len(self.waypoints):
            error = np.linalg.norm(xyz[:2] - self.waypoints[self.next_waypoint])
            if error <= self.task.waypoint_tolerance:
                self.next_waypoint += 1
        self.final_error = float(np.linalg.norm(xyz[:2] - self.task.goal))
        supported=abs(xyz[2]-self.task.cube_size/2)<0.01 and tilt<0.2
        at_goal = (self.final_error <= self.task.goal_tolerance and speed < 0.02
                   and angular_speed<0.1 and supported)
        self.settled = self.settled + 1 if at_goal else 0

    def result(self):
        route_complete = self.next_waypoint == len(self.waypoints)
        success = (self.settled >= self.task.settle_steps and not self.overlap
                   and self.max_lift < 0.03)
        result = dict(contact_scored=self.contact_max is not None)
        if self.contact_max is not None:
            # Robot or cube touching the obstacle invalidates success; robot-table
            # contact is allowed (fingers may brush the tabletop) but reported.
            obstacle_contact = any(self.contact_max[k] > CONTACT_THRESHOLD_N
                                   for k in ("robot_obstacle", "cube_obstacle"))
            success = success and not obstacle_contact
            result.update(obstacle_contact=bool(obstacle_contact),
                          **{f"max_force_{k}_n": v for k, v in self.contact_max.items()},
                          **{f"contact_steps_{k}": v for k, v in self.contact_steps.items()})
        return dict(success=bool(success), route_success=bool(success and route_complete),
                    waypoints_reached=self.next_waypoint, obstacle_overlap=bool(self.overlap),
                    final_error_m=self.final_error, max_lift_m=self.max_lift,
                    control_steps=self.steps, simulated_seconds=self.steps*self.task.control_dt,
                    **result)
