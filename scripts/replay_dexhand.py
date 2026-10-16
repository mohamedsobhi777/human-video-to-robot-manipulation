"""Replay EgoDex hands (full skeleton) on the floating Inspire hand in FruitToBowl-v1.

Go/no-go test for dexterous retargeting: each clip's wrist and fingers drive the
robot hand, anchored to the simulated fruit and bowl (dexhand.retarget_hand).
Reports per clip whether the fruit was lifted and whether the episode succeeded.
"""
import argparse, json, sys
import numpy as np, torch, gymnasium as gym
sys.path.insert(0, ".")
import fruitbowl
from fruitbowl import dexhand, egodex
from fruitbowl.env import HAND, HAND_START

p = argparse.ArgumentParser()
p.add_argument("--episodes", type=int, nargs="*")
p.add_argument("--video", help="mp4 of the first 4 clips")
p.add_argument("--out", default="results/dexhand_replay.json")
p.add_argument("--release-height", type=float, default=0.11)
args = p.parse_args()

task = egodex.Task()
eps = args.episodes if args.episodes is not None else list(range(len(task.episodes)))
kin = dexhand.InspireKinematics.cached()
env = gym.make("FruitToBowl-v1", num_envs=len(eps), robot_uids=HAND, obs_mode="state", control_mode="pd_joint_pos",
               sim_backend="physx_cuda", render_mode="rgb_array" if args.video else None)
u = env.unwrapped
env.reset(seed=7, options=dict(layouts=[dict(target=e % 3) for e in eps]))
start = np.array(HAND_START)
hold = torch.tensor(np.r_[start, 0, 0, np.zeros(6)], dtype=torch.float32, device=u.device).repeat(len(eps), 1)
for _ in range(10): env.step(hold)

fruit = u.target_pose()[:, :3].cpu().numpy()
bowl = u.bowl.pose.p.cpu().numpy()
plans = []
for i, e in enumerate(eps):
    ep = task.episode(e, joints=egodex.HAND_JOINTS)
    H = dexhand.human_hand(ep, egodex.active_side(ep))
    root, q, gs, rs = dexhand.retarget_hand(kin, H, fruit[i], bowl[i], release_height=args.release_height,
                                           start_angles=start[3:])
    root, q, n_lead = dexhand.lead_in(start, root, q)
    plans.append(np.concatenate([root, np.zeros((len(root), 2)), q], 1))
T = max(len(pl) for pl in plans) + 30
actions = np.stack([np.concatenate([pl, np.repeat(pl[-1:], T - len(pl), 0)]) for pl in plans])
actions = torch.tensor(actions, dtype=torch.float32, device=u.device)

rest_z = fruit[:, 2].copy()
max_lift = np.zeros(len(eps))
frames = []
for k in range(T):
    obs, rew, term, trunc, info = env.step(actions[:, k])
    max_lift = np.maximum(max_lift, u.target_pose()[:, 2].cpu().numpy() - rest_z)
    if args.video and k % 2 == 0:
        frames.append(np.concatenate(list(env.render()[:4].cpu().numpy()), 1))
res = [dict(episode=e, target=int(u.target[i]), lifted=bool(max_lift[i] > 0.05), max_lift=float(max_lift[i]),
            in_bowl=bool(info["target_in_bowl"][i]), success=bool(info["success"][i]),
            others_in_bowl=bool(info["others_in_bowl"][i]), bowl_upset=bool(info["bowl_upset"][i]))
       for i, e in enumerate(eps)]
summary = {k: float(np.mean([r[k] for r in res])) for k in ("lifted", "in_bowl", "success", "others_in_bowl", "bowl_upset")}
summary["episodes"] = len(res)
print(json.dumps(summary, indent=1))
json.dump(dict(summary=summary, episodes=res), open(args.out, "w"), indent=1)
if args.video:
    import imageio; imageio.mimwrite(args.video, frames, fps=10)
