"""Closed-loop keyframe controller for FruitToBowl-v1, optionally styled by a human clip.

Batched state machine over `pd_ee_pose` actions (absolute TCP position and
XYZ-Euler orientation in the robot base frame, then gripper: +1 open, -1 closed):
approach -> above fruit -> down -> close -> lift -> carry -> lower -> open -> retreat.

Without a style it is the privileged scripted expert. With a style (one per
environment, see `style_from_plan`) the keyframes come from what the person did
in their clip, measured relative to the objects: where the hand came from
before the grasp, the grasp yaw, how high the fruit was lifted and carried,
where over the bowl and from what height it was released, and where the hand
went afterwards. Positions are re-read from the scene every step, so the same
style works in any layout.
"""
import numpy as np
import torch

BASE = torch.tensor([-0.615, 0.0, 0.0])
TOP_DOWN = np.pi                      # Euler X of a gripper pointing straight down
HOVER_Z, CARRY_Z, RELEASE_Z = 0.14, 0.20, 0.13
ROUND = (True, False, True)           # persimmon, pear, mandarin: any grasp yaw works for round fruit


def quat_rotate(q, v):
    """Rotate vectors v (..., 3) by unit quaternions q (..., 4) in wxyz order."""
    w, xyz = q[..., :1], q[..., 1:]
    t = 2*torch.cross(xyz, v, dim=-1)
    return v + w*t + torch.cross(xyz, t, dim=-1)


def grasp_yaw(fruit_quat):
    """Gripper yaw closing across the fruit's narrow side: fingers close along the
    TCP y axis, so align that axis with the fruit's local x (YCB long axis is y)."""
    x_axis = quat_rotate(fruit_quat, torch.tensor([1.0, 0, 0], device=fruit_quat.device).expand_as(fruit_quat[..., 1:]))
    yaw = torch.atan2(x_axis[..., 1], x_axis[..., 0]) - np.pi/2
    return torch.remainder(yaw + np.pi/2, np.pi) - np.pi/2        # gripper is symmetric under 180 deg


def clamp_norm(v, max_norm):
    n = np.linalg.norm(v)
    return v if n <= max_norm else v*max_norm/n


def style_from_plan(pl, fps=30.0, fruit_z=0.03):
    """Object-relative keyframe parameters from a human plan (table-frame pinch path,
    grasp/release frames). Clamped to what the robot can do safely."""
    p, g, r = pl["pinch"], pl["g"], pl["r"]
    T = len(p)
    before = p[max(0, g - int(0.5*fps))] - p[g]
    approach = np.r_[clamp_norm(before[:2], 0.10), np.clip(before[2], 0.04, 0.12)]
    lift = np.clip(p[g:min(T, g + int(fps)), 2].max() - p[g, 2], 0.06, 0.18)
    carry = np.clip(p[g:r+1, 2].max() - p[g, 2] + fruit_z, 0.16, 0.28)
    release_z = np.clip(p[r, 2] - p[g, 2] + fruit_z, 0.11, 0.18)
    bowl = pl.get("bowl_xy", p[r, :2])
    release_off = clamp_norm(p[r, :2] - bowl, 0.035)
    after = p[min(T - 1, r + int(0.5*fps))] - p[r]
    retreat = np.r_[clamp_norm(after[:2], 0.10), np.clip(after[2], 0.03, 0.12)]
    return dict(approach=approach, lift=lift, carry=carry, release_z=release_z, release_off=release_off,
                retreat=retreat, yaw=float(pl["yaw"][g]))


class Expert:
    PHASES = ["approach", "above", "down", "close", "lift", "carry", "lower", "open", "retreat", "done"]
    TIMEOUT = dict(approach=60, above=40, down=40, close=12, lift=30, carry=60, lower=30, open=10, retreat=20, done=10**9)
    CLOSE, OPEN = 3, 7

    def __init__(self, env, styles=None, arm=None):
        """styles: optional list (one per env) of dicts from style_from_plan.
        arm: optional embodiments.Arm; turns the gripper command into another robot's action."""
        self.u = env.unwrapped
        self.arm = arm
        n, dev = self.u.num_envs, self.u.device
        self.phase = torch.zeros(n, dtype=torch.long, device=dev)
        self.timer = torch.zeros(n, dtype=torch.long, device=dev)
        self.yaw = torch.zeros(n, device=dev)
        self.base = BASE.to(dev)
        T = lambda key, default: torch.tensor(np.array([s[key] if s else default for s in (styles or [None]*n)]),
                                              dtype=torch.float32, device=dev)
        self.approach = T("approach", [0.0, 0.0, HOVER_Z - 0.03])
        self.lift = T("lift", CARRY_Z - 0.03)
        self.carry = T("carry", CARRY_Z)
        self.release_z = T("release_z", RELEASE_Z)
        self.release_off = T("release_off", [0.0, 0.0])
        self.retreat = T("retreat", [0.0, 0.0, 0.05])
        self.human_yaw = T("yaw", np.nan)
        # Embodiment adaptation: a smaller arm keeps the human style but caps its heights.
        for key, cap in ((arm.cfg.get("heights") or {}).items() if arm is not None else ()):
            setattr(self, key, getattr(self, key).clamp(max=cap))
        self.release_pt = torch.zeros(n, 3, device=dev)

    def reset(self, idx=None):
        idx = slice(None) if idx is None else idx
        self.phase[idx] = 0
        self.timer[idx] = 0

    def act(self):
        u = self.u
        tcp = u.agent.tcp.pose.p
        target = u.target_pose()
        fruit, quat = target[:, :3], target[:, 3:]
        bowl = u.bowl.pose.p
        ph = self.phase
        # Grasp yaw, fixed before descending: the person's for round fruit, across the
        # pear's narrow side for the pear.
        round_fruit = torch.tensor(ROUND, device=tcp.device)[u.target]
        yaw_now = torch.where(round_fruit & ~torch.isnan(self.human_yaw), torch.nan_to_num(self.human_yaw), grasp_yaw(quat))
        self.yaw = torch.where(ph <= 1, yaw_now, self.yaw)
        col = lambda x, y, z: torch.stack([x, y, z], 1)
        drop_xy = bowl[:, :2] + self.release_off
        goals = [
            col(fruit[:, 0] + self.approach[:, 0], fruit[:, 1] + self.approach[:, 1], (fruit[:, 2] + self.approach[:, 2]).clamp(min=0.07)),
            col(fruit[:, 0], fruit[:, 1], fruit[:, 2] + 0.07),
            col(fruit[:, 0], fruit[:, 1], (fruit[:, 2] - 0.005).clamp(min=0.015)),
            col(fruit[:, 0], fruit[:, 1], (fruit[:, 2] - 0.005).clamp(min=0.015)),
            col(tcp[:, 0], tcp[:, 1], self.lift + 0.03),
            col(drop_xy[:, 0], drop_xy[:, 1], self.carry),
            col(drop_xy[:, 0], drop_xy[:, 1], self.release_z),
            col(drop_xy[:, 0], drop_xy[:, 1], self.release_z),
            self.release_pt + self.retreat,
            self.release_pt + self.retreat,
        ]
        goal = torch.stack(goals, 1)[torch.arange(len(ph), device=ph.device), ph]
        gripper = torch.where((ph >= self.CLOSE) & (ph < self.OPEN), -1.0, 1.0)
        # Limit each step to 2.5 cm so the PD controller tracks smoothly.
        step = goal - tcp
        step = step*torch.clamp(0.025/step.norm(dim=1, keepdim=True).clamp_min(1e-9), max=1.0)
        if self.arm is not None:
            action = self.arm.action(tcp + step, self.yaw, gripper)
        else:
            cmd = tcp + step - self.base
            rot = torch.stack([torch.full_like(self.yaw, TOP_DOWN), torch.zeros_like(self.yaw), -self.yaw], 1)
            action = torch.cat([cmd, rot, gripper[:, None]], 1)
        # Phase transitions: reach the goal, or wait a fixed time for gripper phases.
        timed_phase = (ph == self.CLOSE) | (ph == self.OPEN)
        reached = (goal - tcp).norm(dim=1) < 0.01
        timed = timed_phase & (self.timer >= 8)
        timeout = self.timer >= torch.tensor([self.TIMEOUT[p] for p in self.PHASES], device=ph.device)[ph]
        advance = ((reached & ~timed_phase) | timed | timeout) & (ph < len(self.PHASES) - 1)
        entering_retreat = advance & (ph == self.OPEN)
        self.release_pt = torch.where(entering_retreat[:, None], tcp, self.release_pt)
        self.phase = ph + advance.long()
        self.timer = torch.where(advance, 0, self.timer + 1)
        return action
