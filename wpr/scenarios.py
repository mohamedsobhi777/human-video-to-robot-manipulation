"""Seeded task variation, disjoint splits, and reproducible disturbance assignment.

A scenario fixes everything an episode depends on before it runs: layout,
physical parameters, route side and disturbance. Methods compared on the same
scenario therefore face identical conditions.
"""
from dataclasses import dataclass, asdict, replace
import numpy as np
from wpr.geometry import Task, route_waypoints, route_polyline, square_overlaps_obstacle

# Independent random streams per split: the split index is part of the RNG seed,
# so equal integer seeds in different splits never share samples.
SPLITS = {"train": 0, "dev": 1, "val": 2, "test": 3, "shifted": 4}

# In-distribution ranges (metres, kg, friction coefficient). "shifted" widens
# physical parameters beyond these ranges to measure generalization separately.
# Route offsets keep larger-x routes at x <= 0.69 m: the palm-down hand cannot
# push from beyond a planar radius of about 0.76 m (see decision 0001).
RANGES = dict(
    obstacle_dx=(-0.02, 0.02), obstacle_dy=(-0.02, 0.02),
    endpoint_dx=(-0.03, 0.03), endpoint_dy=(-0.02, 0.02),
    route_offset=(0.13, 0.17), friction=(0.35, 0.60), cube_mass=(0.06, 0.12))
SHIFTED_FRICTION = ((0.25, 0.35), (0.60, 0.75))
SHIFTED_MASS = ((0.12, 0.16),)

# Disturbances trigger at a sampled fraction of route arc length, so every
# method is disturbed at the same place on the route, not the same time.
DISTURBANCE_PROGRESS = (0.3, 0.7)
SHOVE_DELTA_V = (0.25, 0.45)  # m/s, nominal impulse / mass
SHOVE_DURATION_PHYSICS_STEPS = 6  # 0.05 s at 120 Hz
DISPLACE_DISTANCE = (0.02, 0.04)


@dataclass(frozen=True)
class Scenario:
    split: str
    seed: int
    route: str
    task: Task
    disturbance: dict  # kind: none | shove | displace

    @property
    def id(self):
        return f"{self.split}-{self.seed:05d}-{self.route}-{self.disturbance['kind']}"

    def to_dict(self):
        return dict(id=self.id, split=self.split, seed=self.seed, route=self.route,
                    task=asdict(self.task), disturbance=self.disturbance)

    @classmethod
    def from_dict(cls, data):
        task = Task(**{k: tuple(v) if isinstance(v, list) else v for k, v in data["task"].items()})
        return cls(data["split"], data["seed"], data["route"], task, data["disturbance"])


def _uniform_union(rng, intervals):
    widths = np.array([b - a for a, b in intervals])
    i = rng.choice(len(intervals), p=widths / widths.sum())
    return float(rng.uniform(*intervals[i]))


def sample_task(rng, split="dev", base=Task()):
    r = RANGES
    u = lambda key: float(rng.uniform(*r[key]))
    obstacle = (base.obstacle[0] + u("obstacle_dx"), base.obstacle[1] + u("obstacle_dy"))
    start = (obstacle[0] + u("endpoint_dx"), base.start[1] + u("endpoint_dy"))
    goal = (obstacle[0] + u("endpoint_dx"), base.goal[1] + u("endpoint_dy"))
    if split == "shifted":
        friction, mass = _uniform_union(rng, SHIFTED_FRICTION), _uniform_union(rng, SHIFTED_MASS)
    else:
        friction, mass = u("friction"), u("cube_mass")
    return replace(base, obstacle=obstacle, start=start, goal=goal, route_offset=u("route_offset"),
                   friction=friction, cube_mass=mass)


def sample_disturbance(rng, kind, task):
    if kind == "none":
        return dict(kind="none")
    angle = float(rng.uniform(-np.pi, np.pi))
    direction = [float(np.cos(angle)), float(np.sin(angle))]
    progress = float(rng.uniform(*DISTURBANCE_PROGRESS))
    if kind == "shove":
        dv = float(rng.uniform(*SHOVE_DELTA_V))
        steps = SHOVE_DURATION_PHYSICS_STEPS
        force = task.cube_mass * dv / (steps * task.control_dt / 3)
        return dict(kind="shove", progress_fraction=progress, direction=direction,
                    nominal_delta_v_mps=dv, force_n=force, physics_steps=steps)
    if kind == "displace":
        distance = float(rng.uniform(*DISPLACE_DISTANCE))
        return dict(kind="displace", progress_fraction=progress,
                    delta_xy_m=[distance*direction[0], distance*direction[1]])
    raise ValueError(f"Unknown disturbance kind: {kind}")


def make_scenario(split, seed, route, disturbance="none"):
    if split not in SPLITS: raise ValueError(f"Unknown split: {split}")
    if route not in ("left", "right"): raise ValueError(f"Unknown route: {route}")
    # Layout depends on (split, seed) only, so the two routes and all disturbance
    # conditions of one seed share the same layout and are directly paired.
    task = sample_task(np.random.default_rng([SPLITS[split], seed]), split)
    kinds = ("none", "shove", "displace")
    dist_rng = np.random.default_rng([SPLITS[split], seed, kinds.index(disturbance), route == "right"])
    return Scenario(split, seed, route, task, sample_disturbance(dist_rng, disturbance, task))


def nominal_scenario(route, disturbance=None):
    """The fixed development layout used by the recorded baseline runs."""
    return Scenario("dev", -1, route, Task(), disturbance or dict(kind="none"))


def validate_scenario(s: Scenario):
    """Reject sampled layouts whose straight route segments touch the obstacle."""
    points = route_polyline(s.task, route_waypoints(s.task, s.route))
    for a, b in zip(points[:-1], points[1:]):
        for t in np.linspace(0, 1, 50):
            if square_overlaps_obstacle(a*(1-t) + b*t, np.pi/4, s.task):
                return False
    return True
