"""Replay the person's processed clips in the simulator, each in a copy of the layout
measured from its own video, with the robot following that clip's style
(object-relative keyframes from the hand). Success per clip + optional video."""
import argparse, json, pickle, sys
import numpy as np, torch, gymnasium as gym
sys.path.insert(0, ".")
import fruitbowl
from fruitbowl import rollout
from fruitbowl.expert import Expert, style_from_plan

p = argparse.ArgumentParser()
p.add_argument("plans")
p.add_argument("--video")
p.add_argument("--steps", type=int, default=240)
p.add_argument("--out", default="results/replay_session.json")
args = p.parse_args()

plans = pickle.load(open(args.plans, "rb"))
from fruitbowl.env import HALF
clampxy = lambda v, r: np.clip(v, -(HALF - r), HALF - r)
layouts = []
for pl in plans:
    # The measured layout, kept inside the simulated work area. A clip that started with
    # another fruit already in the bowl replays in a sampled layout (same target).
    if pl.get("in_bowl_at_start"):
        l = dict(target=pl["target"])
    else:
        l = dict(bowl=clampxy(pl["bowl_xy"], 0.09), fruits=clampxy(pl["fruits_xy"], 0.04), target=pl["target"])
    if pl.get("instruction"): l["instruction"] = pl["instruction"]
    layouts.append(l)
env = gym.make("FruitToBowl-v1", num_envs=len(plans), obs_mode="state", control_mode="pd_ee_pose",
               render_mode="rgb_array" if args.video else None, sim_backend="physx_cuda")
env.reset(seed=0, options=dict(layouts=layouts))
rollout.settle(env)
ex = Expert(env, styles=[style_from_plan(pl, fps=pl.get("fps", 30.0)) for pl in plans])
frames = []
for k in range(args.steps):
    obs, r, term, trunc, info = env.step(ex.act())
    if args.video and k % 2 == 0:
        frames.append(np.concatenate(list(env.render()[:min(4, len(plans))].cpu().numpy()), 1))
res = {pl["clip"]: bool(s) for pl, s in zip(plans, info["success"].cpu().numpy())}
print(json.dumps(res, indent=1)); print("success", np.mean(list(res.values())))
json.dump(dict(per_clip=res, success=float(np.mean(list(res.values())))), open(args.out, "w"), indent=1)
if args.video:
    import imageio; imageio.mimwrite(args.video, frames, fps=10)
