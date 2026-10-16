"""Smoke test: build FruitToBowl-v1, render, drop each fruit into the bowl, check success."""
import argparse, sys, time
from pathlib import Path
import cv2, gymnasium as gym, numpy as np, torch
sys.path.insert(0, ".")
import fruitbowl
from mani_skill.utils.structs.pose import Pose

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("out", nargs="?", default="artifacts/smoke")
p.add_argument("--throughput-envs", type=int, default=0)
args = p.parse_args()
out = args.out
Path(out).mkdir(parents=True, exist_ok=True)
env = gym.make("FruitToBowl-v1", num_envs=3, obs_mode="state", render_mode="rgb_array", sim_backend="physx_cuda")
u = env.unwrapped
obs, info = env.reset(seed=0, options=dict(layouts=[dict(target=k) for k in range(3)]))
cv2.imwrite(f"{out}/fruitbowl_reset.png", cv2.cvtColor(np.concatenate(list(env.render().cpu().numpy()), 1), cv2.COLOR_RGB2BGR))
print("instructions:", u.instructions)
zero = torch.zeros(env.action_space.shape)
for _ in range(30): obs, r, term, trunc, info = env.step(zero)
print("settled: success", info["success"].tolist(), "in_bowl", u.in_bowl().tolist())
assert not info["success"].any(), "reset layouts must not already solve the task"
# Teleport each env's target fruit 10 cm above its bowl and let it fall.
for k, f in enumerate(u.fruits):
    pose = f.pose.raw_pose.clone()
    mask = u.target == k
    pose[mask, :3] = u.bowl.pose.p[mask] + torch.tensor([0, 0, 0.10], device=u.device)
    f.set_pose(Pose.create_from_pq(pose[:, :3], pose[:, 3:]))
u.scene._gpu_apply_all()  # pose writes outside reset must be pushed to the GPU sim
for _ in range(60): obs, r, term, trunc, info = env.step(zero)
print("after drop: success", info["success"].tolist(), "in_bowl", u.in_bowl().tolist(), "reward", r.tolist())
assert info["success"].all(), "dropping each target into its bowl must solve all three episodes"
cv2.imwrite(f"{out}/fruitbowl_dropped.png", cv2.cvtColor(np.concatenate(list(env.render().cpu().numpy()), 1), cv2.COLOR_RGB2BGR))
env.close()
# Throughput
if args.throughput_envs:
    env = gym.make("FruitToBowl-v1", num_envs=args.throughput_envs, obs_mode="state", sim_backend="physx_cuda")
    env.reset(seed=1)
    a = torch.zeros((args.throughput_envs,) + env.unwrapped.single_action_space.shape)
    t = time.time()
    for _ in range(100): env.step(a)
    print(f"throughput: {args.throughput_envs*100/(time.time()-t):.0f} steps/s at {args.throughput_envs} envs")
    env.close()
