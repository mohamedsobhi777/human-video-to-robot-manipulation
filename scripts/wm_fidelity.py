"""Where can the world model be trusted? Fidelity of open-loop predictions against horizon.

From every 5th step of every test episode (styles from held-out clips, new layouts),
roll the ensemble forward with the logged actions for up to H steps (0.1 s each)
and compare with what the physics simulator did:
  - fruit and gripper position error against horizon (median, 90th percentile);
  - trust horizon: the longest horizon with median fruit error under 1 cm;
  - disagreement as a trust signal: does ensemble spread predict the error?
    (rank correlation, and mean error per spread decile);
  - the same split by phase (fruit resting, carried, released).
Writes results/<name>.json and a figure; keeps example rollouts for videos.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, ".")
from fruitbowl import worldmodel as W

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/fruitbowl/wm_mine")
p.add_argument("--model", default="data/fruitbowl/wm_mine/model.pt")
p.add_argument("--horizon", type=int, default=40)
p.add_argument("--every", type=int, default=5)
p.add_argument("--episodes", type=int, default=600)
p.add_argument("--name", default="wm_fidelity_mine")
args = p.parse_args()
dev = "cuda"
te = np.load(Path(args.data)/"test.npz")
S = torch.as_tensor(te["state"][:args.episodes], dtype=torch.float32, device=dev)
A = torch.as_tensor(te["action"][:args.episodes], dtype=torch.float32, device=dev)
model = W.Ensemble(attach=True).to(dev)
model.load_state_dict(torch.load(args.model, map_location=dev, weights_only=True)); model.eval()
B, T1, _ = S.shape; T = T1 - 1; H = args.horizon

err_f, err_t, spread, phase = [], [], [], []       # each (n_starts*B, H)
held = W.held_labels(S.cpu()).to(dev) if hasattr(W, "held_labels") else None
with torch.no_grad():
    for t0 in range(0, T - H, args.every):
        pred = model.rollout(S[:, t0], A[:, t0:t0 + H])          # (M, B, H+1, 22)
        true = S[:, t0:t0 + H + 1]
        med = pred.median(0).values
        err_f.append((med[:, 1:, W.FRUIT] - true[:, 1:, W.FRUIT]).norm(dim=-1).cpu())
        err_t.append((med[:, 1:, W.TCP] - true[:, 1:, W.TCP]).norm(dim=-1).cpu())
        spread.append(pred[:, :, 1:, W.FRUIT].std(0).norm(dim=-1).cpu())
        # phase at the start: carried if the fruit is held (above the table and near the gripper)
        lifted = (S[:, t0, W.FRUIT][:, 2] - S[:, 0, W.FRUIT][:, 2]) > 0.02
        near = (S[:, t0, W.FRUIT] - S[:, t0, W.TCP]).norm(dim=-1) < 0.05
        moved = (S[:, t0, W.FRUIT] - S[:, 0, W.FRUIT]).norm(dim=-1) > 0.05
        phase.append(torch.where(lifted & near, 1, torch.where(moved, 2, 0)).cpu())
err_f, err_t, spread = torch.cat(err_f).numpy(), torch.cat(err_t).numpy(), torch.cat(spread).numpy()
phase = torch.cat(phase).numpy()
h = np.arange(1, H + 1)
curve = dict(horizon_s=(h*0.1).round(1).tolist(),
             fruit_median_cm=(np.median(err_f, 0)*100).round(2).tolist(), fruit_p90_cm=(np.percentile(err_f, 90, 0)*100).round(2).tolist(),
             tcp_median_cm=(np.median(err_t, 0)*100).round(2).tolist(), tcp_p90_cm=(np.percentile(err_t, 90, 0)*100).round(2).tolist())
trust = int(h[np.median(err_f, 0) < 0.01].max()) if (np.median(err_f, 0) < 0.01).any() else 0
rank = lambda x: np.argsort(np.argsort(x))
spearman = [float(np.corrcoef(rank(spread[:, k]), rank(err_f[:, k]))[0, 1]) for k in range(H)]
# Calibration at 1 s: mean error per spread decile.
k1 = min(9, H - 1)
edges = np.quantile(spread[:, k1], np.linspace(0, 1, 11))
dec = np.clip(np.digitize(spread[:, k1], edges[1:-1]), 0, 9)
calib = [float(err_f[dec == d, k1].mean()*100) for d in range(10)]
by_phase = {name: dict(n=int((phase == i).sum()), fruit_median_cm_at_1s=float(np.median(err_f[phase == i, k1])*100),
                       fruit_median_cm_at_3s=float(np.median(err_f[phase == i, min(29, H-1)])*100))
            for i, name in enumerate(("resting", "carried", "released"))}
res = dict(test_episodes=int(B), starts=int(len(err_f)), trust_horizon_s=trust*0.1, curve=curve,
           spread_error_spearman_by_horizon=[round(x, 3) for x in spearman], calibration_at_1s_cm_by_spread_decile=[round(x, 2) for x in calib],
           by_phase=by_phase)
print(json.dumps({k: v for k, v in res.items() if k != "curve"}, indent=1))
print("fruit median cm @0.5/1/2/3/4 s:", [curve["fruit_median_cm"][i] for i in (4, 9, 19, 29, H-1)])
json.dump(res, open(f"results/{args.name}.json", "w"), indent=1)

# Figure: error vs horizon (fruit, gripper) and calibration.
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for key, col, lab in (("fruit", "#D9601F", "fruit"), ("tcp", "#5E7A86", "gripper")):
    ax[0].plot(h*0.1, curve[f"{key}_median_cm"], color=col, label=f"{lab}, median")
    ax[0].plot(h*0.1, curve[f"{key}_p90_cm"], color=col, ls="--", lw=1, label=f"{lab}, 90th percentile")
ax[0].axhline(1, color="#9AA39C", lw=0.8); ax[0].set_xlabel("prediction horizon (s, open loop)"); ax[0].set_ylabel("position error (cm)")
ax[0].set_title(f"Open-loop error; trusted to {trust*0.1:.1f} s (median fruit error < 1 cm)", fontsize=10); ax[0].legend(fontsize=8, frameon=False)
ax[1].bar(range(1, 11), calib, color="#D9601F"); ax[1].set_xlabel("ensemble disagreement decile (1 = most agreement)")
ax[1].set_ylabel("mean fruit error at 1 s (cm)"); ax[1].set_title(f"Disagreement predicts error (Spearman {spearman[k1]:.2f} at 1 s)", fontsize=10)
for a in ax: a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(f"results/{args.name}.png", dpi=160)
# Example open-loop rollouts from the start of 6 episodes, for trajectory plots / videos.
with torch.no_grad():
    ex = model.rollout(S[:6, 0], A[:6, :T]).cpu().numpy()
np.savez_compressed(f"results/{args.name}_examples.npz", pred=ex, true=S[:6].cpu().numpy(), success=te["success"][:6])
print("wrote", f"results/{args.name}.json", f"results/{args.name}.png")
