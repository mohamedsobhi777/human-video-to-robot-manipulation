"""One human clip, several robots: replay the keyframe styles from my clips on other arms.

For each robot: every clip in its own measured layout (moved as a whole into the work
area), and `--random` episodes in random layouts with a random clip's style.
  python scripts/replay_robots.py ~/data/session2 --robots panda xarm6_robotiq widowxai
"""
import argparse, json, pickle, sys
from pathlib import Path
import imageio, numpy as np, torch, gymnasium as gym
sys.path.insert(0, ".")
import fruitbowl
from fruitbowl.embodiments import Arm
from fruitbowl.env import fit_layout
from fruitbowl.expert import Expert, style_from_plan

p = argparse.ArgumentParser()
p.add_argument("session")
p.add_argument("--robots", nargs="*", default=["panda", "xarm6_robotiq", "widowxai"])
p.add_argument("--random", type=int, default=128)
p.add_argument("--steps", type=int, default=260)
p.add_argument("--video", default="results/robots")
p.add_argument("--out", default="results/replay_robots.json")
args = p.parse_args()
plans = pickle.load(open(Path(args.session).expanduser()/"processed/plans.pkl", "rb"))
styles = [style_from_plan(pl, fps=pl["fps"]) for pl in plans]
Path(args.video).mkdir(parents=True, exist_ok=True)


def run(robot, layouts, sty, seed, video=None):
    env = gym.make("FruitToBowl-v1", num_envs=len(layouts), obs_mode="state", control_mode="pd_ee_pose" if robot != "widowxai" else "pd_joint_pos",
                   robot_uids=robot, sim_backend="physx_cuda", render_mode="rgb_array" if video else None)
    u = env.unwrapped
    env.reset(seed=seed, options=dict(layouts=layouts))
    arm = Arm(env)
    for _ in range(10): env.step(arm.hold())
    ex = Expert(env, styles=sty, arm=arm)
    frames, lifted = [], torch.zeros(len(layouts), dtype=torch.bool, device=u.device)
    z0 = u.target_pose()[:, 2].clone()
    for k in range(args.steps):
        obs, r, term, trunc, info = env.step(ex.act())
        lifted |= u.target_pose()[:, 2] - z0 > 0.04
        if video and k % 2 == 0: frames.append(np.concatenate(list(env.render()[:4].cpu().numpy()), 1))
    if video: imageio.mimwrite(video, frames, fps=10)
    res = dict(success=info["success"].cpu().numpy().astype(bool), lifted=lifted.cpu().numpy())
    env.close()
    return res


out = {}
for robot in args.robots:
    own = []
    for pl in plans:
        b, f = fit_layout(pl["bowl_xy"], pl["fruits_xy"])
        own.append(dict(bowl=b, fruits=f, target=pl["target"]) if not pl.get("in_bowl_at_start") else dict(target=pl["target"]))
    r1 = run(robot, own, styles, 0, video=f"{args.video}/{robot}_my_layouts.mp4")
    rng = np.random.default_rng(1)
    pick = rng.integers(len(plans), size=args.random)
    r2 = run(robot, [dict(target=plans[i]["target"]) for i in pick], [styles[i] for i in pick], 500)
    out[robot] = dict(my_layouts=float(r1["success"].mean()), my_layouts_lifted=float(r1["lifted"].mean()),
                      random_layouts=float(r2["success"].mean()), random_lifted=float(r2["lifted"].mean()),
                      per_clip={pl["clip"]: bool(s) for pl, s in zip(plans, r1["success"])})
    print(robot, {k: v for k, v in out[robot].items() if k != "per_clip"}, flush=True)
json.dump(out, open(args.out, "w"), indent=1)
