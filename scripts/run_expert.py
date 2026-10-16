"""Run the scripted expert on N random episodes; report success per fruit; optional video."""
import argparse, sys, time
import numpy as np, torch, gymnasium as gym
sys.path.insert(0, ".")
import fruitbowl
from fruitbowl.env import FRUITS
from fruitbowl.expert import Expert

p = argparse.ArgumentParser()
p.add_argument("--num-envs", type=int, default=256)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--steps", type=int, default=200)
p.add_argument("--video", help="write an mp4 of the first 4 envs here")
args = p.parse_args()

env = gym.make("FruitToBowl-v1", num_envs=args.num_envs, obs_mode="state", control_mode="pd_ee_pose",
               render_mode="rgb_array" if args.video else None, sim_backend="physx_cuda")
env.reset(seed=args.seed)
u = env.unwrapped
expert = Expert(env)
frames, ever = [], torch.zeros(args.num_envs, dtype=torch.bool, device=u.device)
t = time.time()
for step in range(args.steps):
    obs, r, term, trunc, info = env.step(expert.act())
    ever |= info["success"]
    if args.video and step % 2 == 0:
        frames.append(np.concatenate(list(env.render()[:4].cpu().numpy()), 1))
final = info["success"]
print(f"{args.num_envs} episodes, {time.time()-t:.1f}s")
for k, name in enumerate(FRUITS):
    m = u.target == k
    print(f"  {name:10s} n={int(m.sum()):4d}  success at end {final[m].float().mean():.3f}  ever {ever[m].float().mean():.3f}")
print(f"  all        success at end {final.float().mean():.3f}  others knocked in {info['others_in_bowl'].float().mean():.3f}"
      f"  bowl upset {info['bowl_upset'].float().mean():.3f}")
print("  phase histogram:", torch.bincount(expert.phase, minlength=9).tolist())
if args.video:
    import imageio
    imageio.mimwrite(args.video, frames, fps=15)
