"""Receding-horizon pushing with a learned dynamics model.

``MPCPusher`` shares every non-contact behaviour with ``RoutePusher`` (lifting,
obstacle-aware approach planning, lowering, retraction, waypoint switching and
all safety checks), so a comparison between the two isolates one change: during
contact the planar hand command is chosen by sampling short action sequences,
predicting their outcome with the learned ensemble, and executing the first
action of the best sequence (cross-entropy method), instead of the geometric
steering rule.

Each candidate holds a push-direction offset (relative to the route-tracking
direction) and a speed fraction for the horizon. Cost: progress and cross-track
error along the route segment, overshoot past the segment end, cube yaw change,
cube clearance from the obstacle, loss of hand-cube contact, and an optional
penalty on ensemble disagreement (``uncertainty_weight``).

Planning is batched across all environments currently pushing: controllers
register requests during ``command`` and the runner calls ``BatchPlanner.solve``
once per control step, then overwrites those environments' planar targets.
"""
import time
import numpy as np
import torch
from wpr.dynamics import load_model, features_t, apply_t, wrap_t, CUBE_SYMMETRY
from wpr.route_controller import RoutePusher, PUSH_Z, PUSH_LEAD, CONTACT_OFFSET, CUBE_CLEARANCE

_PLANNERS = {}


def shared_planner(model_path, **kw):
    key = (str(model_path), tuple(sorted(kw.items())))
    if key not in _PLANNERS:
        _PLANNERS[key] = BatchPlanner(model_path, **kw)
    return _PLANNERS[key]


class BatchPlanner:
    def __init__(self, model_path, horizon=8, samples=128, iterations=2, uncertainty_weight=1.0,
                 seed=0, device=None):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model, self.meta = load_model(model_path, self.device)
        self.horizon, self.samples, self.iterations = horizon, samples, iterations
        self.uncertainty_weight = uncertainty_weight
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.requests = []
        self.solve_seconds = []

    def request(self, controller, state, direction, yaw, max_step, a, b):
        self.requests.append((controller, state, direction, yaw, max_step, a, b))
        return len(self.requests)-1

    @torch.no_grad()
    def solve(self):
        """Plan every pending request in one batch; returns planar hand targets."""
        if not self.requests: return []
        tic = time.perf_counter()
        dev, R, N, H = self.device, len(self.requests), self.samples, self.horizon
        T = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)
        st = [r[1] for r in self.requests]
        s0 = dict(cube=T([s["object"][:2] for s in st]), cube_yaw=T([s["yaw"] for s in st]),
                  cube_vel=T([[s["object_velocity"][0], s["object_velocity"][1], s["object_velocity"][5]] for s in st]),
                  hand=T([s["hand"] for s in st]), hand_yaw=T([s["hand_yaw"] for s in st]),
                  q=T([s["joint_positions"][:7] for s in st]), qd=T([s["joint_velocities"][:7] for s in st]))
        heading = T([np.arctan2(r[2][1], r[2][0]) for r in self.requests])
        yaw = T([r[3] for r in self.requests])
        max_step = T([r[4] for r in self.requests])
        a, b = T([r[5] for r in self.requests]), T([r[6] for r in self.requests])
        tasks = [r[0].task for r in self.requests]
        obstacle = T([t.obstacle for t in tasks])
        half = T([np.asarray(t.obstacle_size[:2])/2 for t in tasks])
        mean = T([r[0].mean if r[0].mean is not None else (0.0, 1.0) for r in self.requests])
        std = torch.tensor([0.35, 0.3], device=dev).expand(R, 2).clone()
        rep = lambda x: x.repeat_interleave(N, 0)
        s0b = {k: rep(v) for k, v in s0.items()}
        seg = b - a
        length = seg.norm(dim=1).clamp_min(1e-9)
        u = seg/length[:, None]
        normal = torch.stack([-u[:, 1], u[:, 0]], 1)
        for _ in range(self.iterations):
            params = mean[:, None] + std[:, None]*torch.randn(R, N, 2, device=dev, generator=self.gen)
            params[:, 0] = mean  # always evaluate the incumbent
            params[..., 0] = params[..., 0].clamp(-np.pi/2, np.pi/2)
            params[..., 1] = params[..., 1].clamp(0.0, 1.0)
            angle = heading[:, None] + params[..., 0]
            step = torch.stack([angle.cos(), angle.sin()], -1)*(max_step[:, None]*params[..., 1])[..., None]
            step = step.reshape(R*N, 2)
            s, spread = s0b, torch.zeros(R*N, device=dev)
            z = torch.full((R*N, 1), PUSH_Z, device=dev)
            for _ in range(H):
                action = torch.cat([s["hand"][:, :2] + step, z, rep(yaw)[:, None]], 1)
                out = self.model(features_t(s, action, rep(obstacle)))
                spread += out[:, :, 0:2].std(0).sum(-1)
                s = apply_t(s, out.mean(0))
            rel = s["cube"] - rep(a)
            progress = (rel*rep(u)).sum(1)
            cross = (rel*rep(normal)).sum(1)
            gap = ((s["cube"] - rep(obstacle)).abs() - rep(half)).clamp_min(0).norm(dim=1)
            contact = (s["cube"] - s["hand"][:, :2]).norm(dim=1)
            cost = (-progress.clamp(max=rep(length)) + 3.0*cross.abs() + 4.0*(progress - rep(length)).clamp_min(0)
                    + 0.02*wrap_t(s["cube_yaw"] - s0b["cube_yaw"], CUBE_SYMMETRY).abs()
                    + 5.0*(CUBE_CLEARANCE - gap).clamp_min(0)
                    + 2.0*(contact - (CONTACT_OFFSET+0.02)).clamp_min(0)
                    + self.uncertainty_weight*spread).reshape(R, N)
            k = max(8, N//10)
            elite = torch.gather(params, 1, cost.argsort(1)[:, :k, None].expand(R, k, 2))
            mean, std = elite.mean(1), elite.std(1) + 0.02
        angle = heading + mean[:, 0]
        planar = s0["hand"][:, :2] + torch.stack([angle.cos(), angle.sin()], 1)*(max_step*mean[:, 1])[:, None]
        planar, mean = planar.cpu().numpy(), mean.cpu().numpy()
        for (controller, *_), m in zip(self.requests, mean):
            controller.mean = m
        self.requests = []
        self.solve_seconds.append(time.perf_counter()-tic)
        return list(planar)


class MPCPusher(RoutePusher):
    def __init__(self, waypoints, task, model_path, **planner_kw):
        super().__init__(waypoints, task)
        self.planner = shared_planner(model_path, **planner_kw)
        self.mean = None      # CEM warm start: (direction offset, speed fraction)
        self.pending = None   # index of this step's planning request, if any

    def restart(self, phase="lift"):
        super().restart(phase)
        self.mean = None
        self.pending = None  # a request from an abandoned push must not be applied

    def push_step(self, state, hand, direction, yaw, distance, contact_vector):
        """Register a planning request; return the geometric command as a
        placeholder that the runner replaces with the planned target."""
        fallback = super().push_step(state, hand, direction, yaw, distance, contact_vector)
        # Close the gap from the approach standoff geometrically: before contact no
        # short-horizon candidate makes progress, and the uncertainty term would
        # then steer the hand away from the cube.
        if float(contact_vector @ direction) - CONTACT_OFFSET > 0.008:
            return fallback
        target_index = min(self.index+1, len(self.points)-1)
        max_step = max(0.003, PUSH_LEAD*min(1.0, distance/0.08))
        self.pending = self.planner.request(self, state, direction, yaw, max_step,
                                            self.points[target_index-1], self.points[target_index])
        return fallback
