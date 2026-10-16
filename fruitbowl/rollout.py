"""Execute retargeted human plans in FruitToBowl-v1, in batch, optionally recording
camera images, robot state and actions for policy training.

A human plan is object-relative (pinch path, yaw, grasp and release frames,
target fruit), so the same plan can run in any layout: that is how one
recorded clip yields many simulated demonstrations.

Policy interface (6-D, robot base frame):
  state  = [tcp x, y, z, cos 2*yaw, sin 2*yaw, gripper opening (m)]
  action = [target x, y, z, cos 2*yaw, sin 2*yaw, gripper (+1 open, -1 closed)]
The doubled yaw angle makes the encoding continuous for a two-finger gripper,
which looks the same after a 180-degree turn.
"""
import numpy as np
import torch
from fruitbowl import retarget
from fruitbowl.expert import BASE, TOP_DOWN


def tcp_yaw(u):
    """Yaw of the TCP about the vertical, from its x axis (top-down gripper)."""
    from fruitbowl.expert import quat_rotate
    q = u.agent.tcp.pose.q
    x = quat_rotate(q, torch.tensor([1.0, 0, 0], device=q.device).expand(len(q), 3))
    return torch.atan2(x[:, 1], x[:, 0])


def robot_state(u):
    yaw = tcp_yaw(u)
    width = u.agent.robot.get_qpos()[:, -2:].sum(1)
    return torch.cat([u.agent.tcp.pose.p - BASE.to(u.device), torch.cos(2*yaw)[:, None], torch.sin(2*yaw)[:, None], width[:, None]], 1)


def to_env_action(a, device):
    """6-D policy action -> pd_ee_pose action (position, XYZ Euler, gripper)."""
    a = torch.as_tensor(a, dtype=torch.float32, device=device)
    yaw = 0.5*torch.atan2(a[:, 4], a[:, 3])
    rot = torch.stack([torch.full_like(yaw, TOP_DOWN), torch.zeros_like(yaw), -yaw], 1)
    # Euler Z = -yaw gives a TCP x axis at +yaw for the top-down gripper (X = pi flips z).
    return torch.cat([a[:, :3], rot, a[:, 5:6]], 1)


def plan_actions(plans, u, anchor_noise=None, **retarget_kw):
    """Retarget each plan onto its env's current fruit and bowl -> (N, L, 6) actions, lengths.
    anchor_noise: optional (N, 6) offsets added to the grasp and release anchors
    (used to generate near-miss plans for world-model training)."""
    fruit = u.target_pose()[:, :3].cpu().numpy()
    bowl = u.bowl.pose.p.cpu().numpy(); bowl[:, 2] = 0.0
    if anchor_noise is not None:
        fruit, bowl = fruit + anchor_noise[:, :3], bowl + anchor_noise[:, 3:]
    tcp0 = u.agent.tcp.pose.p.cpu().numpy()
    seqs = []
    for i, pl in enumerate(plans):
        P, Y, G = retarget.retarget(pl["pinch"], pl["yaw"], pl["g"], pl["r"], fruit[i], bowl[i], tcp0[i], **retarget_kw)
        seqs.append(np.concatenate([P - BASE.numpy(), np.cos(2*Y)[:, None], np.sin(2*Y)[:, None], G[:, None]], 1))
    lengths = np.array([len(s) for s in seqs])
    L = lengths.max()
    acts = np.stack([np.concatenate([s, np.repeat(s[-1:], L - len(s), 0)]) for s in seqs])
    return acts, lengths


def settle(env, steps=10):
    u = env.unwrapped
    hold = torch.cat([u.agent.tcp.pose.p - BASE.to(u.device), torch.tensor([[TOP_DOWN, 0, 0, 1.0]], device=u.device).expand(u.num_envs, 4)], 1)
    for _ in range(steps): env.step(hold)


def run_plans(env, plans, layouts, seed=0, extra_steps=30, record=False, cameras=("base_camera", "hand_camera"), record_every=1,
              action_noise=0.0, **retarget_kw):
    """action_noise: std (m) of slowly varying noise added to the executed TCP target
    while the clean plan action is recorded as the label (DART-style), so the
    policy also sees slightly-off states paired with the correction."""
    """Reset with `layouts`, retarget `plans` (one per env) and execute.
    Returns per-env success, and if record: dict of per-env lists (images, state, action)."""
    u = env.unwrapped
    env.reset(seed=seed, options=dict(layouts=layouts))
    settle(env)
    acts, lengths = plan_actions(plans, u, **retarget_kw)
    n, L = acts.shape[:2]
    T = L + extra_steps
    rec = dict(state=[], action=[], **{c: [] for c in cameras}) if record else None
    done_at = np.full(n, -1)
    rng = np.random.default_rng(seed)
    drift = np.zeros((n, 3))
    for k in range(T):
        a = acts[:, min(k, L - 1)]
        if record and k % record_every == 0:
            obs = env.unwrapped.get_obs()
            rec["state"].append(robot_state(u).cpu().numpy())
            rec["action"].append(a.copy())
            for c in cameras:
                rec[c].append(obs["sensor_data"][c]["rgb"].cpu().numpy())
        executed = a
        if action_noise > 0:
            drift = 0.9*drift + rng.normal(0, action_noise*np.sqrt(1 - 0.9**2), size=drift.shape)
            executed = a.copy(); executed[:, :3] += drift
        obs, rew, term, trunc, info = env.step(to_env_action(executed, u.device))
        s = info["success"].cpu().numpy()
        done_at[(done_at < 0) & s & (k >= lengths - 1)] = k
    success = info["success"].cpu().numpy()
    if record:
        rec = {key: np.stack(v, 1) for key, v in rec.items()}       # (N, T, ...)
    return success, lengths, rec


def scene_state(u):
    """World-model state (N, 22), robot base frame: tcp xyz, cos/sin 2*yaw, gripper
    opening, target fruit xyz + quaternion, bowl xyz, the two other fruits' xyz."""
    idx = torch.arange(u.num_envs, device=u.device)
    fr = torch.stack([f.pose.raw_pose for f in u.fruits], 1)              # (N, 3, 7)
    base = BASE.to(u.device)
    others = torch.stack([fr[idx, (u.target + k) % 3, :3] for k in (1, 2)], 1) - base
    tgt = fr[idx, u.target]
    return torch.cat([robot_state(u), tgt[:, :3] - base, tgt[:, 3:], u.bowl.pose.p - base, others.flatten(1)], 1)


def run_plans_states(env, plans, layouts, seed=0, anchor_noise=None, extra_steps=30, every=2):
    """Execute plans and log (state, action) every `every` control steps, holding each
    logged action for `every` steps (the 10 Hz interface the world model learns)."""
    u = env.unwrapped
    env.reset(seed=seed, options=dict(layouts=layouts))
    settle(env)
    acts, lengths = plan_actions(plans, u, anchor_noise)
    L = acts.shape[1]
    S, A = [], []
    for k in range(0, L + extra_steps, every):
        a = acts[:, min(k, L - 1)]
        S.append(scene_state(u).cpu().numpy()); A.append(a.copy())
        for _ in range(every):
            obs, rew, term, trunc, info = env.step(to_env_action(a, u.device))
    S.append(scene_state(u).cpu().numpy())
    return np.stack(S, 1), np.stack(A, 1), info["success"].cpu().numpy(), lengths
