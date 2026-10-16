"""Human-styled keyframe demos in random layouts: success rate per style and a
state/action log (world-model format) for the learnability check (state_bc.py)."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, gymnasium as gym
sys.path.insert(0, ".")
import fruitbowl
from fruitbowl import egodex, rollout
from fruitbowl.expert import Expert, style_from_plan

p = argparse.ArgumentParser()
p.add_argument("--rounds", type=int, default=8)
p.add_argument("--holdout", type=int, nargs="*", default=[3, 11, 19, 27, 35, 43])
p.add_argument("--out", default="data/fruitbowl/styled_states.npz")
p.add_argument("--noise", type=float, nargs="*", default=[0.0],
               help="std (m) of slowly varying noise on executed commands; one value per round, cycled. "
                    "Gives the world model off-nominal states like those a learned policy visits.")
p.add_argument("--grip-flip", type=float, default=0.0, help="per-step probability of flipping the gripper command")
p.add_argument("--plans", help="plans.pkl of a recorded session instead of EgoDex clips")
p.add_argument("--clips", nargs="*", help="with --plans: use only these clips (e.g. the held-out ones)")
p.add_argument("--exclude", nargs="*", default=[], help="with --plans: leave these clips out")
p.add_argument("--seed", type=int, default=300_000)
args = p.parse_args()
if args.plans:
    import pickle
    plans = [pl for pl in pickle.load(open(args.plans, "rb")) if (not args.clips or pl["clip"] in args.clips) and pl["clip"] not in args.exclude]
    for k, pl in enumerate(plans): pl["episode"] = k
    styles = [style_from_plan(pl, fps=pl["fps"]) for pl in plans]
else:
    plans, _ = egodex.human_plans("wilor")
    plans = [pl for pl in plans if pl["episode"] not in args.holdout]
    styles = [style_from_plan(pl) for pl in plans]
print(len(plans), "plans", flush=True)
print(json.dumps({k: [float(np.round(np.percentile([np.atleast_1d(s[k])[0] for s in styles], q), 3)) for q in (10, 50, 90)]
                  for k in ("lift", "carry", "release_z", "yaw")}))
env = gym.make("FruitToBowl-v1", num_envs=512, obs_mode="state", control_mode="pd_ee_pose", sim_backend="physx_cuda")
u = env.unwrapped
rng = np.random.default_rng(0)
S, A, Y, M = [], [], [], []
for r in range(args.rounds):
    pick = rng.integers(len(plans), size=512)
    env.reset(seed=args.seed + r, options=dict(layouts=[dict(target=plans[i]["target"]) for i in pick]))
    rollout.settle(env)
    ex = Expert(env, styles=[styles[i] for i in pick])
    s_log, a_log = [], []
    sigma = args.noise[r % len(args.noise)]
    drift = torch.zeros(512, 3, device=u.device)
    flip_until = torch.zeros(512, dtype=torch.long, device=u.device)
    for k in range(240):
        a = ex.act()
        if sigma > 0:
            drift = 0.9*drift + torch.randn_like(drift)*sigma*np.sqrt(1 - 0.81)
            a = a.clone(); a[:, :3] += drift
        if args.grip_flip > 0:
            start = torch.rand(512, device=u.device) < args.grip_flip
            flip_until = torch.where(start, torch.full_like(flip_until, k + 10), flip_until)
            a = a.clone(); a[:, 6] = torch.where(flip_until > k, -a[:, 6], a[:, 6])
        if k % 2 == 0:
            s_log.append(rollout.scene_state(u).cpu().numpy())
            a_log.append(torch.cat([a[:, :3], torch.cos(2*ex.yaw)[:, None], torch.sin(2*ex.yaw)[:, None], a[:, 6:7]], 1).cpu().numpy())
        obs, rew, term, trunc, info = env.step(a)
    s_log.append(rollout.scene_state(u).cpu().numpy())
    S.append(np.stack(s_log, 1)); A.append(np.stack(a_log, 1)); Y.append(info["success"].cpu().numpy())
    M.append(np.array([plans[i]["episode"] for i in pick]))
    print(f"round {r}: success {Y[-1].mean():.3f}", flush=True)
Y_all, M_all = np.concatenate(Y), np.concatenate(M)
per_style = {int(e): float(Y_all[M_all == e].mean()) for e in np.unique(M_all)}
print("overall", Y_all.mean().round(3), "| worst styles", sorted(per_style.items(), key=lambda x: x[1])[:5])
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(args.out, state=np.concatenate(S), action=np.concatenate(A), success=Y_all, plan=M_all)
