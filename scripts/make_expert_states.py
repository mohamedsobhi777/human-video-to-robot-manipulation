"""Scripted-expert rollouts logged in the world-model state/action format (10 Hz)."""
import sys
from pathlib import Path
import numpy as np, torch, gymnasium as gym
sys.path.insert(0, ".")
import fruitbowl
from fruitbowl import rollout
from fruitbowl.expert import Expert
env = gym.make("FruitToBowl-v1", num_envs=512, obs_mode="state", control_mode="pd_ee_pose", sim_backend="physx_cuda")
u = env.unwrapped
S, A, Y = [], [], []
for r in range(8):
    env.reset(seed=200_000 + r); rollout.settle(env)
    ex = Expert(env); s_log, a_log = [], []
    for k in range(200):
        a = ex.act()
        if k % 2 == 0:
            s_log.append(rollout.scene_state(u).cpu().numpy())
            a_log.append(torch.cat([a[:, :3], torch.cos(2*ex.yaw)[:, None], torch.sin(2*ex.yaw)[:, None], a[:, 6:7]], 1).cpu().numpy())
        env.step(a)
    s_log.append(rollout.scene_state(u).cpu().numpy())
    _, _, _, _, info = env.step(a)
    S.append(np.stack(s_log, 1)); A.append(np.stack(a_log, 1)); Y.append(info["success"].cpu().numpy())
    print(r, Y[-1].mean(), flush=True)
out = Path("data/fruitbowl/expert_states.npz")
np.savez_compressed(out, state=np.concatenate(S), action=np.concatenate(A), success=np.concatenate(Y))
