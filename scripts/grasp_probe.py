"""Can the floating Inspire hand grasp and lift each fruit at all? Grid search over grasp geometry.

A scripted top grasp: palm down, fingers along +x pitched down by `pitch`, palm-frame
origin (the wrist) placed at fruit + (dx along +x, 0, dz); thumb pre-opposed; then
close all fingers and lift 15 cm. One environment per (fruit, dx, dz, pitch).
"""
import argparse, itertools, json, sys
import numpy as np, torch, gymnasium as gym
sys.path.insert(0, ".")
import fruitbowl
from fruitbowl import dexhand
from fruitbowl.env import HAND, HAND_START, FRUITS

p = argparse.ArgumentParser()
p.add_argument("--out", default="results/grasp_probe.json")
p.add_argument("--video", help="mp4 of the best configuration per fruit")
args = p.parse_args()
kin = dexhand.InspireKinematics.cached()
dxs, dzs, pitches = np.arange(-0.13, -0.03, 0.02), np.arange(0.04, 0.13, 0.02), np.deg2rad([0, 20, 40, 60])
grid = list(itertools.product(range(3), dxs, dzs, pitches))
env = gym.make("FruitToBowl-v1", num_envs=len(grid), robot_uids=HAND, obs_mode="state", control_mode="pd_joint_pos",
               sim_backend="physx_cuda", render_mode="rgb_array" if args.video else None)
u = env.unwrapped
# Fruit at a fixed spot, bowl and other fruits out of the way.
layouts = [dict(target=f, bowl=[0.06, -0.12], fruits=[[0.06, 0.12], [-0.06, 0.12], [0.06, 0.0]]) for f, *_ in grid]
for l, (f, *_) in zip(layouts, grid):
    l["fruits"] = [list(x) for x in l["fruits"]]; l["fruits"][f] = [-0.05, -0.02]
env.reset(seed=0, options=dict(layouts=layouts))
start = np.array(HAND_START)
open_q = np.zeros(6)
act = lambda root, q: torch.tensor(np.concatenate([root, np.zeros((len(root), 2)), q], 1), dtype=torch.float32, device=u.device)
for _ in range(10): env.step(act(np.repeat(start[None], len(grid), 0), np.repeat(open_q[None], len(grid), 0)))
fruit = u.target_pose()[:, :3].cpu().numpy()

def palm_R(pitch):
    x = np.array([np.cos(pitch), 0, -np.sin(pitch)]); y = np.array([0, 1.0, 0])
    return np.stack([x, y, np.cross(x, y)], -1)

R = np.stack([palm_R(pt) for _, _, _, pt in grid])
grasp_o = fruit + np.array([[dx, 0, dz] for _, dx, dz, _ in grid])
above_o = grasp_o + [0, 0, 0.12]
lift_o = grasp_o + [0, 0, 0.15]
root = lambda o: kin.continuous(kin.root_from_palm(R, o).reshape(-1, 6)[:, None].repeat(1, 1).reshape(-1, 6), prev=None)
r_above, r_grasp, r_lift = (kin.root_from_palm(R, o) for o in (above_o, grasp_o, lift_o))
for r in (r_above, r_grasp, r_lift):         # same Euler branch as the start pose
    for i in range(len(r)): r[i] = kin.continuous(r[i:i+1], prev=start[3:])[0]
thumb_yaw = kin.thumb_q[:, 0].min()
preshape = np.r_[thumb_yaw, 0, 0, 0, 0, 0]
closed = np.r_[thumb_yaw, kin.thumb_q[:, 1].min(), [kin.finger_q[f].min() for f in dexhand.FINGERS]]
lerp = lambda a, b, n: [a + (b - a)*k/n for k in range(1, n + 1)]
plan = ([(r, open_q) for r in lerp(np.repeat(start[None], len(grid), 0), r_above, 40)] +
        [(r_above, q) for q in lerp(open_q, preshape, 10)] +
        [(r, preshape) for r in lerp(r_above, r_grasp, 30)] +
        [(r_grasp, q) for q in lerp(preshape, closed, 20)] +
        [(r, closed) for r in lerp(r_grasp, r_lift, 40)] + [(r_lift, closed)]*20)
frames = []
for k, (r, q) in enumerate(plan):
    env.step(act(r, np.broadcast_to(q, (len(grid), 6)) if q.ndim == 1 else q))
    if args.video and k % 2 == 0: frames.append(env.render().cpu().numpy())
lift = (u.target_pose()[:, 2].cpu().numpy() - fruit[:, 2])
ok = lift > 0.10
res = [dict(fruit=FRUITS[f], dx=float(dx), dz=float(dz), pitch_deg=float(np.rad2deg(pt)), lift=float(l))
       for (f, dx, dz, pt), l in zip(grid, lift)]
summary = {FRUITS[f]: dict(success=float(ok[[g[0] == f for g in grid]].mean()),
                           best=max((r for r in res if r["fruit"] == FRUITS[f]), key=lambda r: r["lift"])) for f in range(3)}
print(json.dumps(summary, indent=1))
json.dump(dict(summary=summary, configs=res), open(args.out, "w"), indent=1)
if args.video:
    import imageio
    best = [int(np.argmax(np.where([g[0] == f for g in grid], lift, -1))) for f in range(3)]
    imageio.mimwrite(args.video, [np.concatenate([fr[i] for i in best], 1) for fr in frames], fps=10)
