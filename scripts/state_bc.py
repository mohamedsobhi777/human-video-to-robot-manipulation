"""Diagnostic: behaviour cloning from privileged scene state on human-derived plans.

Training data: successful rollouts in the world-model dataset (human plans with
perturbed anchors, scene state logged at 10 Hz). The state lists the target
fruit first, so the instruction is already resolved; this isolates whether the
demonstrated motions are learnable, separate from vision and language.
Policy: MLP, state -> next 10 actions (chunk), executes 5 then replans.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, gymnasium as gym
sys.path.insert(0, ".")
import fruitbowl
from fruitbowl import rollout

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/fruitbowl/wm_big/train.npz")
p.add_argument("--epochs", type=int, default=20)
p.add_argument("--chunk", type=int, default=10)
p.add_argument("--execute", type=int, default=5)
p.add_argument("--episodes", type=int, default=256)
p.add_argument("--out", default="results/state_bc.json")
p.add_argument("--save-dir", help="save checkpoints (after the epochs in --save-at) and evaluate each in the simulator")
p.add_argument("--save-at", type=float, nargs="*", default=[0.25, 1, 3, 20])
args = p.parse_args()

z = np.load(args.data)
keep = z["success"].astype(bool)
S, A = z["state"][keep], z["action"][keep]
N, T = A.shape[:2]
idx = [(i, t) for i in range(N) for t in range(T - args.chunk)]
X = np.stack([S[i, t] for i, t in idx]).astype(np.float32)
Y = np.stack([A[i, t:t+args.chunk].ravel() for i, t in idx]).astype(np.float32)
print(f"{N} successful episodes, {len(X)} samples")
xm, xs, ym, ys = X.mean(0), X.std(0) + 1e-4, Y.mean(0), Y.std(0) + 1e-4
dev = "cuda"
net = nn.Sequential(nn.Linear(X.shape[1], 512), nn.SiLU(), nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, 512), nn.SiLU(),
                    nn.Linear(512, Y.shape[1])).to(dev)
Xt, Yt = torch.tensor((X - xm)/xs, device=dev), torch.tensor((Y - ym)/ys, device=dev)
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
steps = args.epochs*len(Xt)//1024
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
save_steps = {int(e*len(Xt)/1024): e for e in args.save_at} if args.save_dir else {}
ckpts = []
for s in range(steps):
    if s in save_steps:
        ckpts.append((save_steps[s], {k: v.clone() for k, v in net.state_dict().items()}))
    b = torch.randint(len(Xt), (1024,), device=dev)
    loss = ((net(Xt[b]) - Yt[b])**2).mean()
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if s % (steps//10) == 0: print(f"step {s}/{steps} loss {loss.item():.4f}", flush=True)

ckpts.append((args.epochs, {k: v.clone() for k, v in net.state_dict().items()}))
env = gym.make("FruitToBowl-v1", num_envs=args.episodes, obs_mode="state", control_mode="pd_ee_pose", sim_backend="physx_cuda")
u = env.unwrapped
xm_t, xs_t, ym_t, ys_t = (torch.tensor(v, device=dev) for v in (xm, xs, ym, ys))
results = []
for epoch, sd in (ckpts if args.save_dir else ckpts[-1:]):
    net.load_state_dict(sd)
    env.reset(seed=777)
    rollout.settle(env)
    s0 = rollout.scene_state(u).float().cpu()
    with torch.no_grad():
        for k in range(0, 200, args.execute):
            s = rollout.scene_state(u).float()
            chunk = (net((s - xm_t)/xs_t)*ys_t + ym_t).reshape(-1, args.chunk, 6)
            for j in range(args.execute):
                for _ in range(2):
                    obs, r, term, trunc, info = env.step(rollout.to_env_action(chunk[:, j], u.device))
    succ = info["success"].float().cpu()
    results.append(dict(epoch=epoch, success=float(succ.mean())))
    print(json.dumps(results[-1]), flush=True)
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        torch.save(dict(state_dict=sd, xm=xm, xs=xs, ym=ym, ys=ys, chunk=args.chunk, execute=args.execute, s0=s0,
                        sim_success=succ, data=args.data, epoch=epoch), Path(args.save_dir)/f"epoch{epoch:g}.pt")
res = dict(success=results[-1]["success"], checkpoints=results, episodes=args.episodes, train_episodes=int(N))
print(json.dumps(res))
json.dump(res, open(args.out, "w"))
