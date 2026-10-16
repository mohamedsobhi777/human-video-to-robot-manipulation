"""Transitions for the world model: human plans replayed in random layouts with
perturbed grasp/release anchors, so the data holds near misses and failures.
Logged at 10 Hz in the 6-D action interface. Train plans and held-out plans
are kept apart (held-out EgoDex episodes as in make_dataset.py)."""
import argparse, sys, time
from pathlib import Path
import numpy as np, gymnasium as gym
sys.path.insert(0, ".")
import fruitbowl
from fruitbowl import egodex, rollout

p = argparse.ArgumentParser()
p.add_argument("--rounds", type=int, default=12)
p.add_argument("--num-envs", type=int, default=512)
p.add_argument("--holdout", type=int, nargs="*", default=[3, 11, 19, 27, 35, 43])
p.add_argument("--out", default="data/fruitbowl/wm")
args = p.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
plans, _ = egodex.human_plans("wilor")
rng = np.random.default_rng(0)
env = gym.make("FruitToBowl-v1", num_envs=args.num_envs, obs_mode="state", control_mode="pd_ee_pose", sim_backend="physx_cuda")
for split in ("train", "test"):
    pool = [pl for pl in plans if (pl["episode"] in args.holdout) == (split == "test")]
    rounds = args.rounds if split == "train" else max(2, args.rounds//4)
    S, A, Y, M = [], [], [], []
    t = time.time()
    for r in range(rounds):
        batch = [pool[i] for i in rng.integers(len(pool), size=args.num_envs)]
        # Anchor noise: half the envs clean-ish, half strongly perturbed.
        scale = np.where(rng.random(args.num_envs) < 0.5, 0.005, 0.03)[:, None]
        noise = rng.normal(size=(args.num_envs, 6))*scale*np.array([1, 1, 0.5, 2, 2, 0.5])
        s, a, y, lengths = rollout.run_plans_states(env, batch, [dict(target=pl["target"]) for pl in batch],
                                                    seed=1000*(split == "test") + r, anchor_noise=noise)
        S.append(s); A.append(a); Y.append(y); M.append([pl["episode"] for pl in batch])
        print(f"{split} round {r}: success {y.mean():.2f}, T={s.shape[1]}, {time.time()-t:.0f}s", flush=True)
    T = max(x.shape[1] for x in S)
    padS = lambda x: np.concatenate([x, np.repeat(x[:, -1:], T - x.shape[1], 1)], 1)
    padA = lambda x: np.concatenate([x, np.repeat(x[:, -1:], T - 1 - x.shape[1], 1)], 1)
    np.savez_compressed(out/f"{split}.npz", state=np.concatenate([padS(x) for x in S]), action=np.concatenate([padA(x) for x in A]),
                        success=np.concatenate(Y), plan=np.concatenate(M))
